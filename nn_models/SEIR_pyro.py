#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:01:20 2025

@author: abel
"""

from epidemic_models import SEIR_Model 
from BNN import pyro_SVI
import pyro
import pyro.distributions as dist
import torch


class SEIR_pyro(SEIR_Model) :
    
    def __init__(self, beta, sigma, gamma, N, **kwargs) :
        
        super().__init__(beta, sigma, gamma, N)
        self.pyro = pyro_SVI(**kwargs)
        self.pyro.guide = self.guide
        
    def seir_model(self, x0, t, data) :
        
        self.beta = self.pyro.sample("beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
        self.gamma = self.pyro.sample("gamma", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
        self.sigma = self.pyro.sample("sigma", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0)))
    
        seir_prediction = self.run(x0, t)
        
        with self.pyro.plate('data', len(data)) :
            self.pyro.sample('obs', dist.Normal(seir_prediction,0.1), obs=data)
            
    def guide(self, data) :
        
        beta_loc = self.pyro.param("beta_loc", torch.tensor(0.0))
        beta_scale = self.pyro.param("beta_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        gamma_loc = self.pyro.param("gamma_loc", torch.tensor(0.0))
        gamma_scale = self.pyro.param("gamma_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
        sigma_loc = self.pyro.param("sigma_loc", torch.tensor(0.0))
        sigma_scale = self.pyro.param("sigma_scale", torch.tensor(0.1), constraint=dist.constraints.positive)
    
        # Aproximación a la posterior de los parámetros
        self.pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
        self.pyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_scale))
        self.pyro.sample("sigma", dist.LogNormal(sigma_loc, sigma_scale))

    
        
        