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
        
    def forward_map(self, theta) : 

        N = self.N
        t = self.time
                
        beta, sigma, gamma = theta
        
        seir_model = SEIR_Model(beta,sigma,gamma,N)
        
        x0 = []

        if self.params['init_cond'] == 'fixed' :
            x0 = self.x0
        elif self.params['init_cond'] == 'estimated' :
            s0, e0, I0, r0 = self.x0
            k = ((1+gamma)*self.data[1] - self.data[0])/sigma
            E1 = (sigma*k + gamma*I0 + self.data[0])/sigma
            E0 = E1-k
            R0 = gamma*self.data[1]

            S0 = N - E0 - I0 - R0
            x0 = [S0,E0,I0,R0]
        elif self.params['init_cond'] == 'estimated_roman' :
            s0, e0, I0, r0 = self.x0

            E0 = self.data[0]/sigma
            R0 = 0
            S0 = N - E0 - I0 - R0
            x0 = [S0,E0,I0,R0]


        else :
            print('Warning: initial condition not defined. Using default')
            x0 = self.x0

        x = seir_model.run(x0,t)
        
        S, E, I, R = x[:]

        #print(' --- ', x0, sigma, gamma )

        if self.params['inc_method'] == 'susceptible' :
            incidency = -seir_model.incidency(np.hstack([[N],S]),t,1,method='susceptible')
        elif self.params['inc_method'] == 'exposed' :
            incidency = seir_model.incidency([E,I],t,1,method='exposed',sigma=sigma,gamma=gamma)
        elif self.params['inc_method'] == 'roman' :
            incidency = seir_model.incidency(np.hstack([[0],E]),t,1,method='roman',sigma=sigma,gamma=gamma)

        return incidency

    def LikelihoodEnergyPoisson(self, theta) :

        incidency = self.forward_map(theta)
        #print(incidency[0]) #### 
        epsilon = 1e-8 ## numerical regularization to avoid log 0
        
        if (incidency < 0).any() :
            incidency = np.maximum(incidency, 0)
        
        p_i = incidency - self.data*np.log(incidency + epsilon)

        if np.isnan(np.sum(p_i)) :
            print('Warning: nan values')   

        return np.sum(p_i)
    
    def LikelihoodEnergyGaussian(self,theta) :
        
        incidency = self.forward_map(theta)
        
        val = (np.linalg.norm(incidency-self.data)**2) 
           
        return val
    
    def LikelihoodEnergyNegBinom(self,theta) :
        #print('Warning. This has not been tested yet!')
        data = self.data

        p_negbinom = self.params['p_negbinom']        
        
        incidency = self.forward_map(theta)

        r = np.round(incidency)
        
        #p = mu / (mu + theta)
        #log_binom = sp.gammaln(data + p_negbinom) - sp.gammaln(data + 1) - sp.gammaln(p_negbinom)
        
        #logL = -np.sum(log_binom + p_negbinom*np.log(1-p) + data*np.log(p))
        
        log_binom = sp.gammaln(data + r) - sp.gammaln(data+1) - sp.gammaln(r)

        return -np.sum(log_binom + r - np.log(1-p_negbinom) + data*np.log(p_negbinom))

    
    def PriorBeta(self,theta) :
        
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
    
    def PriorGamma(self, theta) :

        beta, sigma, gamma = theta

        shape_beta_prior = self.params['shape_beta_prior']
        shape_sigma_prior = self.params['shape_sigma_prior']
        shape_gamma_prior = self.params['shape_gamma_prior']
        scale_beta_prior = self.params['scale_beta_prior']
        scale_sigma_prior = self.params['scale_sigma_prior']
        scale_gamma_prior = self.params['scale_gamma_prior']

        log_pri_beta = scipy.stats.gamma.logpdf(beta,shape_beta_prior,scale=scale_beta_prior)
        log_pri_sigma = scipy.stats.gamma.logpdf(sigma,shape_sigma_prior,scale=scale_sigma_prior)
        log_pri_gamma = scipy.stats.gamma.logpdf(gamma,shape_gamma_prior,scale=scale_gamma_prior)

        return (float(log_pri_beta+log_pri_sigma+log_pri_gamma))
    
    def Supp(self, theta) :
    
        beta, sigma, gamma = theta
        
        beta_min = self.params['beta_min']
        beta_max = self.params['beta_max']
        sigma_min = self.params['sigma_min']
        sigma_max = self.params['sigma_max']
        gamma_min = self.params['gamma_min']
        gamma_max = self.params['gamma_max'] 

        if beta < beta_min :
            return False
        
        if beta > beta_max :
            return False
        
        if sigma < sigma_min :
            return False
        
        if sigma > sigma_max :
            return False
        
        if gamma < gamma_min :
            return False
        
        if gamma > gamma_max :
            return False

        return True
    

    def get_prior_sample(self, n) :

        print(f'Generating {n} prior samples. Prior model: {self.prior_model}')

        prior_curves = {}
        if self.prior_model == 'Beta' :

            for label in self.labels :
                _min = self.params[f'{label}_min']
                _max = self.params[f'{label}_max']
                alpha_prior = self.params[f'alpha_{label}_prior']
                beta_prior = self.params[f'beta_{label}_prior']
                                
                prior_curves[label] = scipy.stats.beta.rvs(alpha_prior, beta_prior, size=n)

        elif self.prior_model == 'Uniform': 
            
            for label in self.labels :
                _min = self.params[f'{label}_min']
                _max = self.params[f'{label}_max']
                
                prior_curves[label] = scipy.stats.uniform.rvs(loc=_min, scale=_max-_min, size=n)
                
        elif self.prior_model == 'Gamma' :

            for label in self.labels :
                _min = self.params[f'{label}_min']
                _max = self.params[f'{label}_max']
                shape_prior = self.params[f'shape_{label}_prior']
                scale_prior = self.params[f'scale_{label}_prior']
                
                prior_curves[label] = scipy.stats.gamma.rvs(shape_prior, scale=scale_prior, size=n)

        return prior_curves