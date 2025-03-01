#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:45:10 2025

@author: abel
"""

from tqdm import tqdm
#import emcee
#import corner
#import numpy as np
from .SEIR_mcmc_base import SEIR_mcmc_base
from .pyhmc import pyhmc

class SEIR_pyhmc(pyhmc, SEIR_mcmc_base) :
    
    def __init__(self, *argv, **kwargs) :
        
        SEIR_mcmc_base.__init__(self, *argv, **kwargs)
        
        if self.prior_model == 'Beta' :
            self.PriorEnergy = self.PriorBeta
        else :
            self.PriorEnergy = self.PriorUniform

        if self.likelihood_model == 'Poisson' :    
            super().__init__(loglikelihood=self.LikelihoodEnergyPoisson, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        elif self.likelihood_model == 'NegBinomial':
            super().__init__(loglikelihood=self.LikelihoodEnergyNegBinom, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        else :
            super().__init__(loglikelihood=self.LikelihoodEnergyGaussian, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        
        self.instance = 'hmc'
    
    def run(self, T, theta_0) :
        
        with tqdm(total=T) as pbar :
            for i, _ in enumerate(self.Run(T, theta_0)) :
                pbar.update(1)
        
        self.nsamples = T
        # put the output in a dataframe
        
        self.create_dictionary()
        
        return True