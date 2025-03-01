#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:12:53 2025

@author: abel
"""
from epidemic_model.SEIR_Model import SEIR_Model
import scipy
import numpy as np
from scipy.special import gammaln
from utils.AnalysisTools import AnalysisTools
import scipy.special as sp
import matplotlib.pyplot as plt

counter = 0
class SEIR_mcmc_base(AnalysisTools) :

    def __init__(self, *argv, **kwargs) :
        
        super().__init__()
        
        try :
            self.N = kwargs['N']
            self.time = kwargs['time']
            self.x0  = kwargs['x0']
            self.data = kwargs['data']
            self.ndim = kwargs['ndim']
            self.likelihood_model = kwargs['likelihood_model']
            self.prior_model = kwargs['prior_model']
            
        except :
            print('Warning. Something is strange here!')
        
        if not 'labels' in kwargs :
            labels = [str(i) for i in range(self.ndim)]
            kwargs['labels'] = labels
        self.labels = kwargs['labels']
        self.params = kwargs
        

    def LikelihoodEnergyPoisson(self, theta) :
        
        #print('evaluating likelihood poisson')
        
        N = self.N
        t = self.time
                
        beta, sigma, gamma = theta
        
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        
        delta_t = t[1] - t[0]
        x = seir_model.run(self.x0,t)
        
        S, E, I, R = x[:]

        incidency = -seir_model.incidency(np.hstack([[N],S]),t,1)
        
        epsilon = 1e-8 ## numerical regularization to avoid log 0
        #p_i = -incidency + self.data[:-1]*np.log(np.abs(incidency))
        
        if (incidency < 0).any() :
            incidency = np.maximum(incidency, 0)
            print('Warning: negative incidency')
        
        # computing scaling constant for avoid the chain gets stucked
        C = np.sum(self.data) / np.sum(incidency)
        
        p_i = C*incidency - self.data*np.log(C*incidency + epsilon)
        global counter 
        
        #if counter %100 :
        #    print(np.sum(p_i))
        counter += 1
        return np.sum(p_i)
    
    def LikelihoodEnergyGaussian(self,theta) :
        
        #print('evaluating likelihood gaussian')
        N = self.N
        t = self.time
                
        beta, sigma, gamma = theta
                
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        x = seir_model.run(self.x0,t)
        
        S, E, I, R = x[:]
        
        incidency = -seir_model.incidency(np.hstack([[N],S]),t,1)
        
        #plt.figure()
        #plt.plot(incidency, label='incidency')
        #plt.plot(self.data, label='data')
        #plt.legend()
        #plt.grid()
        #plt.show()
        
        # assuming 
        val = (np.linalg.norm(incidency-self.data)**2) 
           
        return val
    
    def LikelihoodEnergyNegBinom(self,theta) :
                
        data = self.data
        N = self.N
        t = self.time
        p_negbinom = self.params['p_negbinom']        
        
        beta, sigma, gamma = theta
                
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        x = seir_model.run(self.x0,t)
        
        S, E, I, R = x[:]
        
        incidency = -seir_model.incidency(np.hstack([[N],S]),t,1)
        
        mu = np.round(incidency)
        
        p = mu / (mu + theta)
        log_binom = sp.gammaln(data + p_negbinom) - sp.gammaln(data + 1) - sp.gammaln(p_negbinom)
        
        logL = -np.sum(log_binom + p_negbinom*np.log(1-p) + data*np.log(p))
        
        #log_binomial_coeff = gammaln(data + incidency) - gammaln(incidency) - gammaln(data+1)
        
        #logL = np.sum(log_binomial_coeff + incidency * np.log(p) + data*np.log(1-p))
        
        return logL
    
    def PriorBeta(self,theta) :
        #print('evaluating prior***')
        
        beta, sigma, gamma = theta
        
        alpha_beta_prior = self.params['alpha_beta_prior']
        beta_beta_prior = self.params['beta_beta_prior']
        alpha_sigma_prior = self.params['alpha_sigma_prior']
        beta_sigma_prior = self.params['beta_sigma_prior']
        alpha_gamma_prior = self.params['alpha_gamma_prior']
        beta_gamma_prior = self.params['beta_gamma_prior']

        log_pri_beta = scipy.stats.beta.logpdf(beta,alpha_beta_prior,beta_beta_prior)
        log_pri_sig = scipy.stats.beta.logpdf(sigma,alpha_sigma_prior,beta_sigma_prior)
        log_pri_gam = scipy.stats.beta.logpdf(gamma,alpha_gamma_prior,beta_gamma_prior)
        
        return (float(log_pri_beta+log_pri_sig+log_pri_gam))

    def PriorUniform(self, theta) :
        
        beta, sigma, gamma = theta
        
        beta_prior_loc = self.params['beta_min']
        beta_prior_scale = self.params['beta_max'] - beta_prior_loc
        sigma_prior_loc = self.params['sigma_min']
        sigma_prior_scale = self.params['sigma_max'] - sigma_prior_loc
        gamma_prior_loc = self.params['gamma_min']
        gamma_prior_scale = self.params['gamma_max'] - gamma_prior_loc
        
        log_pri_beta = scipy.stats.uniform.logpdf(beta,loc=beta_prior_loc,scale=beta_prior_scale)
        log_pri_sigma = scipy.stats.uniform.logpdf(sigma,loc=sigma_prior_loc,scale=sigma_prior_scale)
        log_pri_gamma = scipy.stats.uniform.logpdf(gamma,loc=gamma_prior_loc,scale=gamma_prior_scale)
        
        return (float(log_pri_beta+log_pri_sigma+log_pri_gamma))
   
    def PriorLogarithmic(self, theta) :
        
        beta, sigma, gamma = np.exp(theta)
        
        return -0.5*np.sum(theta**2) + np.sum(theta)
    
    def Supp(self, theta) :
    
        beta, sigma, gamma = theta
        
        beta_min = self.params['beta_min']
        beta_max = self.params['beta_max']
        sigma_min = self.params['sigma_min']
        sigma_max = self.params['sigma_max']
        gamma_min = self.params['gamma_min']
        gamma_max = self.params['gamma_max'] 

        if beta < beta_min :
            #print('*')
            return False
        
        if beta > beta_max :
            #print('**')
            return False
        
        if sigma < sigma_min :
            #print('-')
            return False
        
        if sigma > sigma_max :
            #print('--')
            return False
        
        if gamma < gamma_min :
            #print('#')
            return False
        
        if gamma > gamma_max :
            #print('##')
            return False
        
        # if (theta <= 0).any() :
        #     return False
            
        # if (theta > 1).any() :
        #     return False
        
        return True
    

