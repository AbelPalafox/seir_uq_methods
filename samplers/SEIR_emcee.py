#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:36:28 2025

@author: abel
"""
from tqdm import tqdm
import emcee
#import corner
import numpy as np
from .SEIR_mcmc_base import SEIR_mcmc_base

class SEIR_emcee(SEIR_mcmc_base) :
    
    def __init__(self, *argv, **kwargs) :
        
        super().__init__(*argv, **kwargs)
        
        self.nwalkers = kwargs['nwalkers']
        self.ndim = kwargs['ndim']
        print(self.nwalkers, self.ndim)
        self.sampler = emcee.EnsembleSampler(self.nwalkers, 
                                             self.ndim, 
                                             self.lnprob,
                                             moves=[
                                                    emcee.moves.DEMove(),
                                                    emcee.moves.StretchMove()
                                                ])
        
        self.instance = 'emcee'

        
    def lnprob(self, theta) :
        
        
        if not self.Supp(theta) :
            #print('Out of support!')
            return -np.inf
        
        if self.likelihood_model == 'Poisson' :
            lnlike = self.LikelihoodEnergyPoisson(theta)
        elif self.likelihood_model == 'NegBinomial' :
            lnlike = self.LikelihoodEnergyNegBinom
        else :
            lnlike = self.LikelihoodEnergyGaussian(theta)
        
        if self.prior_model == 'Beta' :
            lnprior = self.PriorBeta(theta)
        elif self.prior_model == 'Logarithmic' :
            lnprior = self.PriorLogarithmic(theta)
        else :
            lnprior = self.PriorUniform(theta)
            
        #lnprior = self.PriorEnergy(theta)
        print(-(lnlike + lnprior))
        return -(lnlike + lnprior)
    
    def run(self, T, theta_0) :
        #print(theta_0)
        
        self.nsamples = T 
        with tqdm(total=T) as pbar:
            for i, _ in enumerate(self.sampler.sample(theta_0, iterations=T)):
                pbar.update(1)
        
        
        self.Output = self.sampler.get_chain()
        
        self.create_dictionary()
        
        return True