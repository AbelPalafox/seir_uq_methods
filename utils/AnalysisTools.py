#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:58:49 2025

@author: abel
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy  as np
import scipy.stats

class AnalysisTools :
    
    def __init__(self) :
    
        #plt.style.use('bmh')
        
        return 
        
    def trace_plot(self, burnin=0, subsample=1, end=-1) :
        
        df_samples = {}
        if self.instance == 'emcee' :
            # the samples of the emcee are in an array
            # of shape [..., nwalkers, ndim]
            
            # creating a dataframe for plotting
            
            for i, label in enumerate(self.labels):
                df_samples[label] = self.Output[burnin:end:subsample,:,i].T
        
            idata = az.from_emcee(self.sampler)
            stats = idata['sample_stats']
            #print(stats['lp'])
            
            az.plot_trace(stats, var_names=['lp'])
            az.plot_forest(idata)
            
            
        if self.instance == 'pytwalk' :
            
            for i, label in enumerate(self.labels):
                df_samples[label] = self.Output[burnin:end:subsample,i]
        
            az.plot_trace(self.Output_u)
            #az.plot_forest(df_samples,var_names=self.labels)
        
        az.plot_trace(df_samples,var_names=self.labels, compact=False)
        
        
        
    def create_dictionary(self, burnin=0, subsample=1, end=-1) :
        
        df_samples = {}
        if self.instance == 'emcee' :
            # the samples of the emcee are in an array
            # of shape [..., nwalkers, ndim]
            
            # creating a dataframe for plotting
            
            for i, label in enumerate(self.labels):
                df_samples[label] = self.Output[burnin:end:subsample,:,i]
                #print(i, self.Output[:,:,i].shape, df_samples[label].shape)
                
        if self.instance == 'pytwalk' or self.instance == 'pymhc':
            
            for i, label in enumerate(self.labels):
                df_samples[label] = self.Output[burnin:end:subsample,i]   
            
        #self.samples = pd.DataFrame(df_samples)
        
        #for key in df_samples :
        #    print('*** ', df_samples[key].shape)
        self.samples = df_samples
        
        return
    
    def plot_densities(self, burnin=0, subsample=1, end=-1) :
        
        az.plot_density(self.samples, var_names=self.labels, backend='bokeh')
        
        return
    
    def plot_prior_vs_posterior(self) :
        ## solo funciona para el modelo SEIR ---- actualizar para más general
        
        if self.prior_model == 'Beta' :

            for label in self.labels :
                _min = self.params[f'{label}_min']
                _max = self.params[f'{label}_max']
                alpha_prior = self.params[f'alpha_{label}_prior']
                beta_prior = self.params[f'beta_{label}_prior']
                x_lin = np.linspace(_min, _max, 200)    

                plt.figure()
                plt.hist(self.samples[label], density=True)
                plt.plot(x_lin, scipy.stats.beta.pdf(x_lin, alpha_prior, beta_prior))
                plt.show()
        elif self.prior_model == 'Uniform':
            
            
            
            for label in self.labels :
                _min = self.params[f'{label}_min']
                _max = self.params[f'{label}_max']
                x_lin = np.linspace(_min, _max, 200)    

                plt.figure()
                plt.hist(self.samples[label], density=True)
                plt.plot(x_lin, scipy.stats.uniform.pdf(x_lin, _min, _max - _min))
                plt.title(label)
                plt.grid()
                plt.show()    
            
        return
                        
                
    
    def plot_histograms(self) :
        
        #print(self.samples)
        
        '''if self.instance == 'emcee' :
            samples = []
            for label in self.labels :
                
                samples.append(self.samples[label].reshape(-1))
            
            print('I am here!')
            
            az.plot_dist(samples,kind='kde')
            return 
        
        az.plot_dist(self.samples,kind='kde')
        
        return'''
        
        for label in self.labels :
            az.plot_dist(self.samples[label],rug=True, label=label)
            plt.show()
        
        
    
    def plot_posterior(self) :
        
        az.plot_posterior(self.samples)
        
        return
    
    def plot_pairs(self) :
        
        if self.instance == 'emcee' :
            
            # collapsing the chains 
            samples = {}
            for label in self.labels :
                
                samples[label] = self.samples[label].reshape(-1)
            
            az.plot_pair(samples, kind='kde', marginals=True)
            
            return 
        
        az.plot_pair(self.samples, kind='kde', marginals=True)
        
        return
            
    def plot_dist_comparison(self) :
         
        # creating a prior sample
        n = self.nsamples
        
        prior_samples = self.PriorSample(n)
        
        df_prior = {}
        for i,label in enumerate(self.labels) :
            df_prior[label] = prior_samples[:,i]
        
        df_group = {}
        df_group['prioir'] = df_prior
        df_group['posterior'] = self.samples
        az.plot_dist_comparison(df_group)
        
        return
            
    
    def plot_energy(self) :
        
        log_prob_vals = self.sampler.get_log_probs()
        
        mean_log_prob = self.log_prob_samples.mean(axis=0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    