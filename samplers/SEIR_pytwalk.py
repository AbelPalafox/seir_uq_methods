#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:33:27 2025

@author: abel
"""
from pytwalk import pytwalk
from .SEIR_mcmc_base import SEIR_mcmc_base

# defining the class for the SEIR_twalk model
class SEIR_pytwalk(pytwalk,SEIR_mcmc_base) :

    def __init__(self, *argv, **kwargs) :
        
        SEIR_mcmc_base.__init__(self, *argv, **kwargs)  
        
        if self.prior_model == 'Beta' :
            self.PriorEnergy = self.PriorBeta
        elif self.prior_model == 'Logarithmic' :
            self.PriorEnergy = self.PriorLogarithmic
        elif self.prior_model == 'Gamma' :
            self.PriorEnergy = self.PriorGamma
        else :
            self.PriorEnergy = self.PriorUniform
        
        if self.likelihood_model == 'Poisson' :    
            super().__init__(self.ndim,k=1,u=self.LikelihoodEnergyPoisson,Supp=self.Supp,w=self.PriorEnergy)
        elif self.likelihood_model == 'NegBinomial' :
            super().__init__(self.ndim,k=1,u=self.LikelihoodEnergyNegBinom,Supp=self.Supp,w=self.PriorEnergy)
        else :
            super().__init__(self.ndim,k=1,u=self.LikelihoodEnergyGaussian,Supp=self.Supp,w=self.PriorEnergy)

        self.instance = 'pytwalk'
        
    def run(self, T, xp0, xp1) :
        
        # calling the pytwalk run function
        self.Run(T,xp0,xp1, save_xp=True)
        
        self.nsamples = T
        # put the output in a dataframe
        
        self.create_dictionary()
        
        return True