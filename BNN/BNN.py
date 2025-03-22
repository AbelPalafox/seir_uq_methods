#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:35:47 2025

@author: abel
"""

#from PINN.PINN import PINN
import torch
import numpy as np
from pytwalk import pytwalk
import emcee
from tqdm import tqdm
from samplers.pyhmc import pyhmc

class BNN :
    
    def __init__(self, pinn_model, n_input, n_output, n_hidden, n_flayers, **kwargs) :
        
        self.model = pinn_model(n_input, n_output, n_hidden, n_flayers)
        self.args = kwargs
        self.call_counter = 0

        return 
    
    def set_weights(self, weights) :
        
        with torch.no_grad() :
            for param, new_value in zip(self.model.parameters(), weights):
                param.copy_(torch.tensor(new_value, dtype=param.dtype))     
        return

    def get_params_vector(self) :
        
        params = []
        shapes = []
        for param in self.model.parameters() :
            shapes.append(param.shape)
            params.append(param.view(-1))
            
        return torch.cat(params), shapes
    
    def set_params_vector(self, flat_params, shapes) :
        
        with torch.no_grad() :
            index = 0
            for param, shape in zip(self.model.parameters(), shapes) :
                size = torch.prod(torch.tensor(shape)).item()
                new_value = flat_params[index: index + size].view(shape)
                param.copy_(new_value)
                index += size
        
        return 
        
    def likelihood(self, theta_) :
        
        t = torch.tensor(self.args['t'], dtype=torch.float32).view(-1, 1)
        data = torch.tensor(self.args['data'], dtype=torch.float32).view(-1, 1)
        lambda_data = self.args['lambda_data']
        lambda_cond = self.args['lambda_cond']
        lambda_eq = self.args['lambda_eq']
        
        
        theta = torch.tensor(theta_[:-self.nparams])
        params = [torch.tensor(_) for _ in theta_[-self.nparams:]]
        
        #storing current weights
        original_weights = self.model.state_dict()
        
        # using proposal weights
        self.set_params_vector(theta, self.shapes)
        
        # computing loss
        with torch.no_grad() :
            model_prediction = self.model.forward(t) 
            
        eq_loss, dsystem_dt = self.model.compute_eq_loss(model_prediction, t, params)
        data_loss = self.model.compute_data_loss(model_prediction, data, t)
        cond_loss = self.model.compute_cond_loss(model_prediction)
    
        loss = lambda_eq*eq_loss + lambda_data*data_loss + lambda_cond*cond_loss
        
        # restoring the current weights
        self.model.load_state_dict(original_weights)
        
        return loss.item()
        
    def prior(self, theta_) :
        
        theta = torch.tensor(theta_[:-self.nparams])
        params = torch.tensor(theta_[-self.nparams:])
        
        sigma = self.args['sigma']
        
        # Gaussian prior
        log_p = 0.0
        for param in theta :
            log_p += 0.5*torch.sum(param**2) / (sigma**2)
        
        ## evaluando la prior para los parámetros del modelo SEIR
        log_beta, log_sigma, log_gamma = params
        
        # Media y desviación estándar de las priors normales
        mu_beta, sigma_beta = np.log(self.args['mu_prior_beta']), self.args['sigma_prior_beta']
        mu_sigma, sigma_sigma = np.log(self.args['mu_prior_sigma']), self.args['sigma_prior_sigma']
        mu_gamma, sigma_gamma = np.log(self.args['mu_prior_gamma']), self.args['sigma_prior_gamma']
        
        
    
        # Evaluación del logaritmo de la prior
        log_p += 0.5 * torch.sum((log_beta - mu_beta) ** 2) / (sigma_beta ** 2)
        log_p += 0.5 * torch.sum((log_sigma - mu_sigma) ** 2) / (sigma_sigma ** 2)
        log_p += 0.5 * torch.sum((log_gamma - mu_gamma) ** 2) / (sigma_gamma ** 2)
    
        # Agregar el término de normalización
        log_p += 0.5 * (torch.log(torch.tensor(2 * np.pi * sigma_beta**2)) +
                          torch.log(torch.tensor(2 * np.pi * sigma_sigma**2)) +
                          torch.log(torch.tensor(2 * np.pi * sigma_gamma**2)))
        return log_p
        
    def support(self, theta_) :
        
        params = torch.tensor(theta_[-self.nparams:])
        
        bounds = self.args['bounds']
        
        for p, bound in zip(params, bounds) :
            if torch.exp(p) < bound[0] :
                return False
            if torch.exp(p) > bound[1] :
                return False
        
        return True
    
    def lnprob(self, theta_) :
        
        if not self.support(theta_) :
            print('out of  support')
            return -torch.inf
        
        loss_likelihood = self.likelihood(theta_)
        loss_prior = self.prior(theta_)
        
        loss = -(loss_likelihood + loss_prior)
        if self.call_counter % 1000 == 0:
            print(f'loss: {loss:.6g}, {loss_likelihood:.6g}, {loss_prior:.6g} ')
        self.call_counter += 1
        
        if np.isnan(loss) :
            print(f'loss: {loss:.6g}, {loss_likelihood:.6g}, {loss_prior:.6g} ')
        
        return loss
    
    
    def infer_parameters(self, num_iterations, initial_weights=None, step_size=0.01, sampler=None, **kwargs):
        """
        Inferir los parámetros de la red neuronal bayesiana utilizando MCMC.
        
        Args:
            model: Red neuronal PINN (instancia de torch.nn.Module).
            sampler: Función de sampler que genera nuevas muestras de parámetros.
            num_iterations: Número de iteraciones de MCMC.
            initial_weights: Pesos iniciales de la red. Si no se proporciona, se inicializa aleatoriamente.
            step_size: Tamaño de paso para el sampler (perturbación).
            sigma: Desviación estándar para la prior gaussiana.
        
        Returns:
            Lista con los parámetros inferidos en cada iteración.
        """
        self.args.update(kwargs)

        self.model.args.update(kwargs)

        self.params = kwargs['params']
        self.nparams = len(self.params)
        
        flat_params, shapes = self.get_params_vector()
        self.ndim = len(flat_params) + self.nparams
        self.shapes = shapes
        
        # 1. Inicialización
        if initial_weights is None:
            initial_weights = self.get_params_vector()[0]  # Usamos los parámetros iniciales del modelo
        current_weights = np.array([w.item() for w in initial_weights] + [np.log(_) for _ in self.params]) # juntamos los pesos y sesgos, con los parámetros de la ED
        
        self.sampler_name = sampler
        
        print(f'Running on sampler: {sampler}')

        
        if sampler=='twalk' :
            self.sampler = pytwalk(self.ndim, k=1, w=self.likelihood, Supp=self.support, u=self.prior)
            
            xp0 = current_weights
            xp1 = current_weights + np.random.normal(0,1,len(current_weights))
            self.sampler.Run(num_iterations, xp0, xp1)
            
            self.Output = self.sampler.Output

        elif sampler=='emcee' :
            
            self.nwalkers = int(2*self.ndim)
            print(f'using {self.nwalkers} walkers')
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob)

            theta_0 =[current_weights + 0.5*np.random.randn(len(current_weights)) for _ in range(self.nwalkers)]
            
            with tqdm(total=num_iterations) as pbar:
                for i, _ in enumerate(self.sampler.sample(theta_0, iterations=num_iterations)):
                    pbar.update(1)

            self.Output = self.sampler.get_chain()

        elif sampler=='pyhmc' :
            self.sampler = pyhmc(self.likelihood,self.prior,self.support,**kwargs)
            
            self.Run(num_iterations, current_weights)
            self.Output = self.sampler.Output
                       
        else :
            print('sampler must be one of available')
            return
    
        return
    
        