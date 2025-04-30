#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:01:20 2025

@author: abel
"""

from BNN import pyro_SVI
from BNN import BNN
from PINN import PINN
from .SEIR_PINN import SEIR_PINN
from samplers.SEIR_mcmc_base import SEIR_mcmc_base
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import Transform
#from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import TransformedDistribution, Gamma, ExpTransform
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import torch
from torchdiffeq import odeint as torch_odeint
from utils import Normalizer

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



class SEIR_PINN_pyro :
    
    def __init__(self, **kwargs) :
        
        self.bnn_model = BNN(SEIR_PINN, SEIR_mcmc_base, **kwargs)
        N = kwargs['N']
        min_val = torch.tensor(0, dtype=torch.float32)
        max_val = torch.tensor(N, dtype=torch.float32)
        self.bnn_model.model.normalizer = Normalizer(min_val, max_val)
        self.bnn_model.ndim = 3
        self.pyro_model = pyro_SVI(self.seir_model, self.guide) 
        self.pyro_model.guide = self.guide
        self.args = kwargs
        self.bnn_model.model.args = kwargs
        self.N = torch.tensor(self.args['N'], dtype=torch.float32)
        self.I0 = torch.tensor(self.args['I0'], dtype=torch.float32)


    def seir_model(self, data, **kwargs) :
        t = kwargs['t']
        N = self.N
        lambda_data = self.args['lambda_data']
        lambda_cond = self.args['lambda_cond']
        lambda_eq = self.args['lambda_eq']
        likelihood_model = self.args['likelihood_model']
        
        pinn_params_np = self.bnn_model.get_params_vector()

        # sample prior
        self.beta = pyro.sample("beta", dist.Gamma(torch.tensor(0.35), torch.tensor(0.1)))
        self.gamma = pyro.sample("gamma", dist.Gamma(torch.tensor(0.2), torch.tensor(0.1)))
        self.sigma = pyro.sample("sigma", dist.Gamma(torch.tensor(0.05), torch.tensor(0.1)))
        self.weights = pyro.sample("weights",dist.Normal(torch.tensor([0.0]*len(pinn_params_np)),torch.tensor([1.0]*len(pinn_params_np))).to_event(1))

        with torch.no_grad() :
            self.bnn_model.set_params_vector(self.weights)

        t_data = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
        model_prediction = self.bnn_model.model.forward(t_data) 

        self.bnn_model.model.log_beta = torch.log(torch.tensor(self.beta, dtype=torch.float32))#, requires_grad=True))
        self.bnn_model.model.log_sigma = torch.log(torch.tensor(self.sigma, dtype=torch.float32))#, requires_grad=True))
        self.bnn_model.model.log_gamma = torch.log(torch.tensor(self.gamma, dtype=torch.float32))#, requires_grad=True))

        eq_loss, dsystem_dt = self.bnn_model.model.compute_eq_loss(model_prediction, t_data)
        cond_loss = self.bnn_model.model.compute_cond_loss(model_prediction, data)

        S_pred, I_pred, E_pred = model_prediction.T

        S_pred_denormalized = self.bnn_model.model.normalizer.denormalize(S_pred)
        I_pred_denormalized = self.bnn_model.model.normalizer.denormalize(I_pred)
        E_pred_denormalized = self.bnn_model.model.normalizer.denormalize(E_pred)
        R_pred_denormalized = N - S_pred_denormalized - E_pred_denormalized - I_pred_denormalized

        y = [S_pred_denormalized, E_pred_denormalized, I_pred_denormalized, R_pred_denormalized]

        # Corrección IC trick para que S_pred solo decrezca
        #beta = torch.exp(self.log_beta)  # Aquí usas tu parámetro
        #S_pred_denormalized = N - torch.cumsum( beta * S_pred_denormalized * I_pred_denormalized, dim=0)
        
        #data_denormalized = self.normalizer.denormalize(data)#.requires_grad_()
        if self.args['inc_method'] == 'exposed' : 
            incidence = self.compute_incidency(y, method='exposed')
        elif self.args['inc_method'] == 'susceptible' : 
            incidence = self.compute_incidency(y)
        elif self.args['inc_method'] == 'roman' : 
            incidence = self.compute_incidency(y, method='roman')

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

        return incidence, (lambda_eq*eq_loss + lambda_cond*cond_loss)
    
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

            incidency = torch.nn.functional.softplus(incidency)

            return incidency

        else :

            print('Incidency method has not been implemented yet')
            return None

        return None

    def guide(self, data, **kwargs) :

        beta_loc = pyro.param("beta_loc", torch.tensor(0.35), constraint=dist.constraints.positive)
        beta_scale = torch.clamp(pyro.param("beta_scale", torch.tensor(0.1), constraint=dist.constraints.positive), min=1e-4, max=5e-2)
        
        #sigma_loc = pyro.param("sigma_loc", torch.tensor(0.5), constraint=dist.constraints.positive)
        #sigma_scale = pyro.param("sigma_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        
        sigma_loc = torch.clamp(pyro.param("sigma_loc", torch.tensor(0.05), constraint=dist.constraints.positive), min=1e-3, max=5e-2)
        sigma_scale = torch.clamp(pyro.param("sigma_scale", torch.tensor(0.1), constraint=dist.constraints.positive), min=1e-4, max=5e-2)


        gamma_loc = pyro.param("gamma_loc", torch.tensor(0.2), constraint=dist.constraints.positive)
        gamma_scale = torch.clamp(pyro.param("gamma_scale", torch.tensor(0.1), constraint=dist.constraints.positive), min=1e-4, max=5e-2)
        

        # pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
        # pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))
        # pyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_scale))

        pyro.sample("beta", dist.Gamma(beta_loc, beta_scale))
        pyro.sample("sigma", dist.Gamma(sigma_loc, sigma_scale))
        pyro.sample("gamma", dist.Gamma(gamma_loc, gamma_scale))

        params = self.bnn_model.get_params_vector()
        sigma_weight = self.args['sigma_weight_prior']
        sigma_weights = pyro.param("sigma_weights", torch.tensor([sigma_weight]*len(params)), constraint=dist.constraints.positive)
        mean_weights = torch.tensor([0.0]*len(params)) #pyro.param("mean_weights", torch.tensor([0.0]*len(params)))

        pyro.sample("weights",dist.Normal(mean_weights,sigma_weights).to_event(1))

    
    def compute_initial_conditions(self, y_pred, data, sigma=None) :

        if self.args['init_cond'] == 'fixed' :
            S0 = torch.tensor(self.args['S0'], dtype=torch.float32, requires_grad=False)
            E0 = torch.tensor(self.args['E0'], dtype=torch.float32, requires_grad=False)
            I0 = torch.tensor(self.args['I0'], dtype=torch.float32, requires_grad=False)
            R0 = torch.tensor(self.args['R0'], dtype=torch.float32, requires_grad=False)
        elif self.args['init_cond'] == 'estimated':

            with torch.no_grad() :
                sigma = y_pred.sigma
                gamma = y_pred.gamma
                N = y_pred.N
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
                sigma = y_pred.sigma

            I0 = self.args['I0'] #self.I0
            N = y_pred.N
            #print('** ',data[0], sigma)
            E0 = data[0]/sigma
            R0 = torch.tensor([0], dtype=torch.float32)
            S0 = N - E0 - I0 - R0
            
        else :
            print('Initial conditions method has not been implemented')
            print('init_cond should be fixed or estimated in the configuration file')
            return None
        
        return torch.tensor([S0, E0, I0, R0])
    
    def get_params(self) :
        print(pyro.get_param_store().keys())
        return {name : pyro.param(name).detach().cpu().clone() for name in pyro.get_param_store().keys()}


