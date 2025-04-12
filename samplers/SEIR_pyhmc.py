#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:45:10 2025

@author: abel
"""

from tqdm import tqdm
from .SEIR_mcmc_base import SEIR_mcmc_base
from .pyhmc import pyhmc
import numpy as np

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
        
        self.instance = 'pyhmc'
        self.energy_hmc = None
    
    def gradient_(self, theta) :

        print('using the gradient of SEIR model')

        beta,sigma,gamma = theta

        ######################################    ######################################
        #Solució del ODE

        t = self.time
        dt = self.params['dt']
        N = self.params['N']

        npoints = int(t.size/dt)+1

        S = np.zeros([npoints,])
        E = np.zeros([npoints,])
        I = np.zeros([npoints,])
        Yn = np.zeros([npoints,])
        #Valores inciiales de los estados
        y0 = self.get_initial_conditions(theta)

        S[0] = y0[0]
        E[0] = y0[1]
        I[0] = y0[2]
        
        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a beta
        dSdb = np.zeros([npoints,])
        dEdb = np.zeros([npoints,])
        dIdb = np.zeros([npoints,])

        dSpdb = np.zeros([npoints,])
        dEpdb = np.zeros([npoints,])
        dIpdb = np.zeros([npoints,])
        
        #Valores iniciales 
        dSpdb[0] = -S[0]*I[0]/N
        dEpdb[0] = -dSpdb[0]

        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a sigma
        dSds = np.zeros([npoints,])
        dEds = np.zeros([npoints,])
        dIds = np.zeros([npoints,])

        dSpds = np.zeros([npoints,])
        dEpds = np.zeros([npoints,])
        dIpds = np.zeros([npoints,])

        #Valores iniciales 
        dSpds[0] = 0
        dEpds[0] = -E[0]
        dIpds[0] = -dEpds[0]
        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a gamma
        dSdg = np.zeros([npoints,])
        dEdg = np.zeros([npoints,])
        dIdg = np.zeros([npoints,])

        dSpdg = np.zeros([npoints,])
        dEpdg = np.zeros([npoints,])
        dIpdg = np.zeros([npoints,])

        #Valores iniciales 
        dSpdg[0] = 0
        dEpdg[0] = 0
        dIpdg[0] = -I[0]
        ######################################    ######################################
        #En este ciclo se calculan los valores de las parciales para cada valor en el tiempo
        for i in range(1,npoints):
            S[i] = S[i-1] - dt*beta*S[i-1]*I[i-1]/N
            E[i] = E[i-1] + dt*(beta*S[i-1]*I[i-1]/N - sigma*E[i-1])
            I[i] = I[i-1] + dt*(sigma*E[i-1] - gamma*I[i-1])
            if S[i] < 0:
                S[i] = 0
            if E[i] < 0:
                E[i] = 0
            if I[i] < 0:
                I[i] = 0
            
            Yn[i-1] = sigma*E[i]
            
            dSdb[i] = dSdb[i-1] + dt*dSpdb[i-1]
            dEdb[i] = dEdb[i-1] + dt*dEpdb[i-1]
            dIdb[i] = dIdb[i-1] + dt*dIpdb[i-1]
        
            dSpdb[i] = -(S[i]*I[i] + S[i]*beta*dIdb[i] + I[i]*beta*dSdb[i])/N
            dEpdb[i] = -dSpdb[i] - sigma*dEdb[i]
            dIpdb[i] = -(dSpdb[i]+dEpdb[i]) -gamma*dIdb[i]


            dSds[i] = dSds[i-1] + dt*dSpds[i-1]
            dEds[i] = dEds[i-1] + dt*dEpds[i-1]
            dIds[i] = dIds[i-1] + dt*dIpds[i-1]

            dSpds[i] = -beta*(I[i]*dSds[i] + S[i]*dIds[i])/N
            dEpds[i] = -dSpds[i] - E[i] -sigma*dEds[i]
            dIpds[i] = -(dSpds[i]+dEpds[i]) -gamma*dIds[i]


            dSdg[i] = dSdg[i-1] + dt*dSpdg[i-1]
            dEdg[i] = dEdg[i-1] + dt*dEpdg[i-1]
            dIdg[i] = dIdg[i-1] + dt*dIpdg[i-1]

            dSpdg[i] = -beta*(I[i]*dSdg[i] + S[i]*dIdg[i])/N
            dEpdg[i] = -dSpdg[i] - sigma*dEdg[i]
            dIpdg[i] = -(dSpdg[i]+dEpdg[i]) - gamma*dIdg[i] -I[i]
        
        dE = np.zeros([60,3])
        dE[:,0] = sigma*dEdb[2::2]
        dE[:,1] = sigma*dEds[2::2] + E[2::2]
        dE[:,2] = sigma*dEdg[2::2]
      
        Energy = (self.data-Yn[1::2])@(self.data-Yn[1::2])
        Grad = -2*((self.data-Yn[1::2])@dE)
        
        self.energy_hmc = Energy

        return Grad

    def run(self, T, theta_0) :
        
        with tqdm(total=T) as pbar :
            for i, _ in enumerate(self.Run(T, theta_0)) :
                pbar.update(1)
        
        self.nsamples = T
        # put the output in a dataframe
        self.Output = np.column_stack([self.Output[1:],self.U_samples])

        self.Outputp = self.Output 
        
        self.create_dictionary()
        
        return True