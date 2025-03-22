#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 09:58:49 2025

@author: abel
"""

from collections import defaultdict
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
import scipy.stats
import arviz.labels as azl
from tqdm import tqdm



class AnalysisTools :
    
    def __init__(self) :
    
        #plt.style.use('bmh')
        self.labeller = azl.MapLabeller(var_name_map={"beta": r"$\beta$", "sigma": r"$\sigma$", "gamma": r"$\gamma$"})

        return 
        
    def plot_trace(self, burnin=0, end=-1, subsample=1) :
        
        dic = defaultdict(list)
        if self.instance == 'emcee' :
            for i, label in enumerate(self.labels):
                dic[label] = self.Output[burnin:end:subsample,:,i].T

        elif self.instance == 'pytwalk' or self.instance == 'pymhc':
            for i, label in enumerate(self.labels):
                dic[label] = self.Output[burnin:end:subsample,i]

        az.plot_trace(self.idata,var_names=self.labels, labeller=self.labeller, backend='bokeh', compact=True)

        return
        
        
        
    def create_dictionary(self, burnin=0, subsample=1, end=-1) :
        
        df_samples = {}
        stats_samples = {}
        if self.instance == 'emcee' :
            # the samples of the emcee are in an array
            # of shape [..., nwalkers, ndim]
            
            # creating a dataframe for plotting
            df_samples = {var: self.Output[burnin:end:subsample, :, i].T for i, var in enumerate(self.labels)}

            log_prob = self.sampler.get_log_prob()
            log_prob = log_prob[burnin:end:subsample]
            stats_samples['energy'] = log_prob.T
            acceptance_fraction = self.sampler.acceptance_fraction
            stats_samples['acceptance_fraction'] = acceptance_fraction
            stats_samples['diverging'] = np.zeros_like(log_prob.T, dtype=bool)
               
        if self.instance == 'pytwalk' or self.instance == 'pymhc':
            
            Output = np.stack((self.Output, self.Outputp), axis=1)

            df_samples = {var: Output[burnin:end:subsample,:,i].T for i, var in enumerate(self.labels)}
            log_prob = Output[burnin:end:subsample,:,-1]
            stats_samples['energy'] = log_prob.T
            stats_samples['acceptance_fraction'] = np.array([1])
            stats_samples['diverging'] = np.zeros_like(log_prob.T, dtype=bool)
            stats_samples['acceptance_fraction'] = self.Acc[5]

            iat = self.IAT()

            stats_samples['acor'] = iat
            
        self.samples = df_samples
        self.stats_samples = stats_samples

        self.idata = az.from_dict(self.samples, 
                        sample_stats=self.stats_samples,
                        coords={"param": self.labels},  # Definir coordenadas explícitamente
                        dims={ val:["chain", "draw"] for val in self.labels}, # Definir dimensiones explícitamente
                        )
        
        return
    
    def plot_density(self) :
        
        az.plot_density(self.samples, var_names=self.labels, labeller=self.labeller, shade=0.1, backend='bokeh')
        
        return
    
    def plot_prior_vs_posterior(self) :
        
        prior_curves = self.get_prior_curves()

        for label in self.labels :
            plt.figure()
            az.plot_density(self.idata, shade=0.1, var_names=[label], labeller=self.labeller, backend='bokeh')
            sns.plot(prior_curves[label][:,0], prior_curves[label][:,1], label='Prior')
            plt.legend()
            plt.grid()
            plt.show()  

        return
                        
                
    
    def plot_histograms(self) :

        for label in self.labels :
            az.plot_dist(self.samples[label], 
                         rug=True, 
                         label=self.labeller.var_name_map[label], 
                         show=True)
            
        az.plot_dist(self.stats_samples['energy'],
                     rug=True,
                     label='Energy',
                     show=True) 
            #plt.show()
        
        
    
    def plot_posterior(self) :
        az.plot_posterior(self.idata, var_names=self.labels, kind='kde', labeller=self.labeller, backend='bokeh')
        
        return
    
    def plot_pair(self) :
        
        az.plot_pair(self.idata, 
                     var_names=self.labels, 
                     kind='kde', 
                     marginals=True, 
                     labeller=self.labeller,
                     kde_kwargs={
                        "hdi_probs": [0.3, 0.6, 0.9],  # Plot 30%, 60% and 90% HDI contours
                        "contourf_kwargs": {"cmap": "Blues"},
                        })
        
        return
            
    def plot_dist_comparison(self) :
         
        # creating a prior sample
        n = 500
        
        prior_samples = self.get_prior_sample(n)
        
        idata_combined = az.from_dict(prior=prior_samples,
                                   posterior={var: self.samples[var] for var in self.labels},
                                   coords={"param": self.labels},
                                   dims={val:["chain", "draw"] for val in self.labels})



        az.plot_dist_comparison(idata_combined, 
                                var_names=self.labels, 
                                labeller=self.labeller)
        
        return
            
    
    def plot_energy(self) :
        
        az.plot_energy(self.idata, backend='bokeh')

        return
    
    
    def plot_rank_bars(self) :
        
        az.plot_trace(self.idata, var_names=self.labels, kind='rank_bars', labeller=self.labeller, backend='bokeh')
        
        return
    
    def plot_forest(self) :
        
        az.plot_forest(self.idata,
                       var_names=self.labels,
                       ess=True,
                       hdi_prob=0.95,
                       labeller=self.labeller,
                       backend='bokeh')
        
        return
    
    def plot_parallel(self) :
        
        az.plot_parallel(self.idata, var_names=self.labels, labeller=self.labeller, backend='bokeh')
        
        return
    
    def plot_autocorr(self) :
        
        az.plot_autocorr(self.idata, var_names=self.labels, labeller=self.labeller, backend='bokeh')
        
        return 
    
    def summary(self) :
        
        print(az.summary(self.idata))
        
        return
    
    def generate_from_simulated_data(self, num_samples) :
        
        draw_idx = np.random.choice(self.idata.posterior.draw.size, num_samples, replace=False)

        idata_subsample = self.idata.posterior.isel(draw=("draw", draw_idx))

        idata_subsample_dict = {key: idata_subsample[key].values for key in idata_subsample.keys()}

        idata_subsample = az.from_dict(
            posterior=idata_subsample_dict,
            coords={"param": self.labels},
            dims={val: ["chain", "draw"] for val in self.labels}
        )

        param_vectors = np.stack([idata_subsample.posterior[label].values.flatten() for label in self.labels], axis=1)  # Cada fila es una simulación
        
        simulations = []
        
        def run_loop(param_vector) :

            for theta in param_vectors :
                out = self.forward_map(theta)
                simulations.append(out)
                yield        

        with tqdm(total=param_vectors.shape[0]) as pbar:
            for i, _ in enumerate(run_loop(param_vectors)):
                pbar.update(1)

        return np.array(simulations)
    
    
    def plot_ppc(self, n=0.1) :

        if n < 1 :
            n = int(n * self.idata.posterior.draw.size)

        posterior_predictive = self.generate_from_simulated_data(n)

        mean_predictive = np.mean(posterior_predictive, axis=0)

        plt.figure()
        plt.plot(self.time, self.data, label='Data')
        plt.plot(self.time, posterior_predictive.T, alpha=0.1, lw=3, color='gray')
        plt.plot(self.time, mean_predictive, color='tab:orange', lw=2, label='Mean predictive')   
        plt.grid()
        plt.legend()
        plt.show()

        return
        

    def report_results(self) :
        
        self.summary()
        self.plot_rank_bars()
        self.plot_trace()
        self.plot_forest()
        self.plot_density()
        self.plot_histograms()
        self.plot_pair()
        self.plot_posterior()
        self.plot_parallel()
        self.plot_autocorr()
        self.plot_energy()
        self.plot_dist_comparison()
        self.plot_ppc()
        
        return
    
    
    
    
    