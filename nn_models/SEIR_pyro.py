#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:01:20 2025

@author: abel
"""

from BNN import pyro_SVI
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import Transform
#from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import TransformedDistribution, Gamma, ExpTransform
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import torch
from torchdiffeq import odeint as torch_odeint


class SEIR_Model(torch.nn.Module) :
    def __init__(self, beta, sigma, gamma, N):
        super().__init__()
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N  
        self.labels = ['Susceptible', 'Exposed', 'Infectuous', 'Recovery']

    def forward(self, t, y):
        S, E, I, R = y
        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return torch.stack([dSdt, dEdt, dIdt, dRdt])



class SEIR_pyro :
    
    def __init__(self, **kwargs) :
        
        self.pyro_model = pyro_SVI(self.seir_model, self.guide) 
        self.pyro_model.guide = self.guide
        self.args = kwargs
        self.N = torch.tensor(self.args['N'], dtype=torch.float32)
        self.I0 = torch.tensor(self.args['I0'], dtype=torch.float32)

    def seir_model(self, data, **kwargs) :
        t = kwargs['t']
        # sample prior
        self.beta = pyro.sample("beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
        self.gamma = pyro.sample("gamma", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
        self.sigma = pyro.sample("sigma", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))

        #print(self.beta, self.sigma, self.gamma)

        seir_prediction = SEIR_Model(self.beta, self.sigma, self.gamma, self.N)
        x0 = self.compute_initial_conditions(seir_prediction, data, sigma=self.sigma)
        
        method = 'bdf' if (self.beta/self.sigma > 100 or self.beta/self.gamma > 100 ) else 'dopri5'

        y = torch_odeint(seir_prediction, x0, t,
                         rtol=1e-2,
                         atol=1e-3,
                         method=method,
                         options={'max_num_steps':1000})

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        incidence = self.compute_incidency(y.T, self.args['inc_method'], sigma=self.sigma)
        incidence = incidence.clamp(min=1e-8) 
        
        #print(f"Shape de incidence: {incidence.shape}, Shape de data: {data.shape}")
        likelihood_model = self.args['likelihood_model']

        if likelihood_model == 'Gaussian' :
            sigma = self.args['sigma']
            with pyro.plate('data', len(data)) :
                pyro.sample('obs', dist.Normal(incidence,sigma), obs=data)
        elif likelihood_model == 'NegBinomial' :
            p = self.args['p_negbinom']
            with pyro.plate('data', len(data)) :
                pyro.sample('obs', dist.NegativeBinomial(total_count=incidence,probs=p), obs=data)
        elif likelihood_model == 'Poisson' :
            with pyro.plate("data", len(data)):
                pyro.sample("obs", dist.Poisson(incidence), obs=data)
    
    def compute_incidency(self, y, method='susceptible', sigma=None) :

        S, E, I, R = y 

        if method == 'susceptible' :           

            N_tensor = self.N 
            N_tensor = N_tensor.expand_as(S[:1])  # Asegura compatibilidad de dimensiones
            incidency = -1 * (S - torch.cat([N_tensor, S[:-1]]))  # Diferencia manual

            incidency = torch.nn.functional.softplus(incidency)

            return incidency

        elif method == 'exposed' :
  
            sigma = self.sigma
            gamma = self.gamma
            incidency = sigma*E - gamma*I
            incidency = torch.nn.functional.softplus(incidency)

            return incidency

        elif method == 'roman' :
            if sigma == None :
                sigma = self.sigma
            Y = torch.cumsum(sigma*E,dim=0)

            Zero_tensor = torch.tensor([0], dtype=torch.float32, device=E.device)  # Si es constante
            Zero_tensor = Zero_tensor.expand_as(E[:1]) 

            incidency = (Y - torch.cat([Zero_tensor, Y[:-1]])) 
            
            return incidency

        else :

            print('Incidency method has not been implemented yet')
            return None

        return None

    def guide(self, data, **kwargs) :

        beta_loc = pyro.param("beta_loc", torch.tensor(0.5))
        beta_scale = pyro.param("beta_scale", torch.tensor(0.01), constraint=dist.constraints.positive)
        
        sigma_loc = pyro.param("sigma_loc", torch.tensor(0.5))
        sigma_scale = pyro.param("sigma_scale", torch.tensor(0.01), constraint=dist.constraints.positive)
        
        gamma_loc = pyro.param("gamma_loc", torch.tensor(0.5))
        gamma_scale = pyro.param("gamma_scale", torch.tensor(0.01), constraint=dist.constraints.positive)
        

        pyro.sample("beta", dist.Normal(beta_loc, beta_scale))
        pyro.sample("sigma", dist.Normal(sigma_loc, sigma_scale))
        pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale))



        '''
        prior_model = self.args['prior_model']

        if prior_model == 'Beta' :
            alpha_beta_prior = self.args['alpha_beta_prior']
            beta_beta_prior = self.args['beta_beta_prior']
            alpha_sigma_prior = self.args['alpha_sigma_prior']
            beta_sigma_prior = self.args['beta_sigma_prior']
            alpha_gamma_prior = self.args['alpha_gamma_prior']
            beta_gamma_prior = self.args['beta_gamma_prior']

            #logbeta_beta_dist = TransformedDistribution(dist.Beta(alpha_beta_prior, beta_beta_prior), [LogTransform()])
            pyro.sample('beta', dist.Beta(alpha_beta_prior, beta_beta_prior))

            logbeta_sigma_dist = TransformedDistribution(dist.Beta(alpha_sigma_prior, beta_sigma_prior), [LogTransform()])
            pyro.sample('sigma', dist.Beta(alpha_sigma_prior, beta_sigma_prior))

            #logbeta_gamma_dist = TransformedDistribution(dist.Beta(alpha_gamma_prior, beta_gamma_prior), [LogTransform()])
            pyro.sample('gamma', dist.Beta(alpha_gamma_prior, beta_gamma_prior))

        elif prior_model == 'Gamma' :

            shape_beta_prior = self.args['shape_beta_prior']
            shape_sigma_prior = self.args['shape_sigma_prior']
            shape_gamma_prior = self.args['shape_gamma_prior']
            scale_beta_prior = self.args['scale_beta_prior']
            scale_sigma_prior = self.args['scale_sigma_prior']
            scale_gamma_prior = self.args['scale_gamma_prior']

            #loggamma_beta_dist = TransformedDistribution(Gamma(shape_beta_prior,1.0/scale_beta_prior), [LogTransform()])
            pyro.sample('beta', dist.Gamma(shape_beta_prior,1.0/scale_beta_prior))

            #loggamma_sigma_dist = TransformedDistribution(Gamma(shape_sigma_prior,1.0/scale_sigma_prior), [LogTransform()])
            pyro.sample('sigma', dist.Gamma(shape_sigma_prior,1.0/scale_sigma_prior))

            #loggamma_gamma_dist = TransformedDistribution(Gamma(shape_gamma_prior,1.0/scale_gamma_prior), [LogTransform()])
            pyro.sample('gamma', dist.Gamma(shape_gamma_prior,1.0/scale_gamma_prior))

        elif prior_model == 'Gaussian' :

            beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
            beta_scale = pyro.param("beta_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
            gamma_loc = pyro.param("gamma_loc", torch.tensor(0.0))
            gamma_scale = pyro.param("gamma_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
            sigma_loc = pyro.param("sigma_loc", torch.tensor(0.0))
            sigma_scale = pyro.param("sigma_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        
            # Aproximación a la posterior de los parámetros
            pyro.sample("beta", dist.Normal(beta_loc, beta_scale))
            pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale))
            pyro.sample("sigma", dist.Normal(sigma_loc, sigma_scale))

        else :
            print('Something is wrong here!!')'''

    
    def compute_initial_conditions(self, y_pred, data, sigma=None) :

        if self.args['init_cond'] == 'fixed' :
            S0 = torch.tensor(self.args['S0'], dtype=torch.float32, requires_grad=False)
            E0 = torch.tensor(self.args['E0'], dtype=torch.float32, requires_grad=False)
            I0 = torch.tensor(self.args['I0'], dtype=torch.float32, requires_grad=False)
            R0 = torch.tensor(self.args['R0'], dtype=torch.float32, requires_grad=False)
        elif self.args['init_cond'] == 'estimated':

            with torch.no_grad() :
                sigma = self.sigma
                gamma = self.gamma
                N = self.N
                I0 = self.I0

                hat_I0 = I0  
                k = ((1.0+gamma)*data[1] - data[0])/sigma
                hat_E1 = (sigma*k + gamma*hat_I0 + data[0])/sigma
                hat_E0 = hat_E1 - k
                hat_R0 = gamma*data[1]

                hat_S0 = N - hat_E0 - hat_I0 - hat_R0

                S0 = hat_S0
                E0 = hat_E0
                I0 = hat_I0
                R0 = hat_R0
        
        elif self.args['init_cond'] == 'estimated_roman':
            if sigma == None :
                sigma = self.sigma

            I0 = self.I0
            N = self.N

            E0 = data[0]/sigma
            R0 = torch.tensor([0], dtype=torch.float32)
            S0 = N - E0 - I0 - R0
            
        else :
            print('Initial conditions method has not been implemented')
            print('init_cond should be fixed or estimated in the configuration file')
            return None
        
        return torch.tensor([S0, I0, E0, R0])
    
    def get_params(self) :
        print(pyro.get_param_store().keys())
        return {name : pyro.param(name).detach().cpu().clone() for name in pyro.get_param_store().keys()}
    
    '''def infer_params(self, t, data, num_interations) :

        optimizer = optim.Adam({'lr':0.01})
        
        t = torch.tensor(t, dtype=torch.float32)
        data = torch.tensor(data, dtype=torch.float32)

        svi = SVI(self.seir_model, self.guide, optimizer, loss=Trace_ELBO())

        for it in range(num_interations) :
            loss = svi.step(data)
            if step % 500 == 0 :
                print(f"it : {it}: loss = {loss:.4f}")'''

