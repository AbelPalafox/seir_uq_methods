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
import matplotlib.ticker as ticker
import pickle

class AnalysisTools :
    
    def __init__(self) :
    
        #plt.style.use('bmh')
        self.labeller = azl.MapLabeller(var_name_map={"beta": r"$\beta$", "sigma": r"$\sigma$", "gamma": r"$\gamma$"})
        #self.labeller ={"beta": r"$\beta$", "sigma": r"$\sigma$", "gamma": r"$\gamma$"}
        self.backend = 'matplotlib'
        return 
        
    def plot_trace(self, burnin=0, end=-1, subsample=1) :

        # dic = defaultdict(list)
        # if self.instance == 'emcee' :
        #     for i, label in enumerate(self.labels):
        #         dic[label] = self.Output[burnin:end:subsample,:,i].T

        # elif self.instance == 'pytwalk' or self.instance == 'pymhc':
        #     for i, label in enumerate(self.labels):
        #         dic[label] = self.Output[burnin:end:subsample,i]

        #plt.figure(figsize=(4.5,3))
        import matplotlib.ticker as ticker
        
        ax = az.plot_trace(self.idata,
                           var_names=self.labels, 
                           labeller=self.labeller, 
                           figsize=(7.5,5.5), 
                           backend=self.backend, 
                           compact=True,
                           #rug_kwargs={'rug':True, 'quantiles':[.25,.5,.75]},
                           #plot_kwargs={"color": "tab:red", "linewidth": 1.2},  # Personalizar trazas
                           #hist_kwargs={"color": "tab:blue", "alpha": 0.6, 'hdi_prob':0.94, 'quantiles':[.25,.5,.75]},  # Personalizar histograma
                           #fill_kwargs={"alpha": 0.3},  # Sombreado para HDI
                           #hdi_prob=0.94,  # Intervalo de alta densidad al 94%
                           #hdi_markers=True,  # Muestra los cuantiles como líneas verticales
                           #quantiles=[.25,.5,.75],
                           #hdi_prob=0.94,
                           #point_estimate='mean'
                           #plot_kwargs={'quantiles':[.25,.5,.75],
                           #             'hdi_prob':0.94,
                           #             'point_estimate':'mean'},
                            )

        # Ajustar los ticks de todos los subgráficos
        '''for i,a in enumerate(ax.flat):  # Esto recorre todos los subgráficos
            #a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')  # Gira los números
            a.set_title(a.get_title(), fontsize=12) 
            a.set_ylabel(a.get_ylabel(),fontsize=8)
            a.set_xlabel(a.get_xlabel(),fontsize=8)
            
            start, end = a.get_xlim()
            if not i%2==0 :
                a.xaxis.set_ticks(np.linspace(start, end+1, 5, endpoint=True))
                a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            else :
                a.xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
                a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
            a.figure.tight_layout()  # Ajusta el layout'''

        plt.tight_layout() 
        plt.savefig(self.instance+'_trace_plot.png',dpi=300)

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
               
        if self.instance == 'pytwalk' or self.instance == 'pyhmc':
            
            Output = np.stack((self.Output, self.Outputp), axis=1)

            df_samples = {var: Output[burnin:end:subsample,:,i].T for i, var in enumerate(self.labels)}
            log_prob = Output[burnin:end:subsample,:,-1]
            stats_samples['energy'] = log_prob.T
            stats_samples['acceptance_fraction'] = np.array([1])
            stats_samples['diverging'] = np.zeros_like(log_prob.T, dtype=bool)

            if self.instance == 'pytwalk' :
                stats_samples['acceptance_fraction'] = self.Acc[5]

            #iat = self.IAT()

            #stats_samples['acor'] = iat
            
        self.samples = df_samples
        self.stats_samples = stats_samples

        self.idata = az.from_dict(self.samples, 
                        sample_stats=self.stats_samples,
                        coords={"param": self.labels},  # Definir coordenadas explícitamente
                        dims={ val:["chain", "draw"] for val in self.labels}, # Definir dimensiones explícitamente
                        )
        
        return
    
    def plot_density(self) :
        
        az.plot_density(self.samples, var_names=self.labels, labeller=self.labeller, shade=0.1, backend=self.backend)
        
        return
    
    def plot_prior_vs_posterior(self) :
        
        prior_curves = self.get_prior_curves()

        for label in self.labels :
            plt.figure()
            az.plot_density(self.idata, shade=0.1, var_names=[label], labeller=self.labeller, backend=self.backend)
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

        ax = az.plot_posterior(self.idata, var_names=self.labels, kind='kde', labeller=self.labeller, backend=self.backend)
        
        for i,a in enumerate(ax.flat):  # Esto recorre todos los subgráficos
            #a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')  # Gira los números
            a.set_title(a.get_title(), fontsize=14) 
            a.set_ylabel(a.get_ylabel(),fontsize=8)
            a.set_xlabel(a.get_xlabel(),fontsize=12)
            
            start, end = a.get_xlim()
            a.xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
            a.figure.tight_layout()  # Ajusta el layout

        plt.tight_layout() 
        plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_histograms.png',dpi=300)

        return
    
    def plot_posterior_and_traces(self) :

        ax_t = az.plot_trace(self.idata,
                    var_names=self.labels, 
                    labeller=self.labeller, 
                    figsize=(12,7.5), 
                    backend=self.backend, 
                    compact=False,
                    plot_kwargs={'lw':2, 'alpha':1},
                    trace_kwargs={'lw':2, 'alpha':1},
                    show=False)

        for i, label in enumerate(self.labels) :

            az.plot_posterior(self.idata, 
                                 var_names=label, 
                                 kind='kde', 
                                 labeller=self.labeller, 
                                 backend=self.backend, 
                                 ax=ax_t[i,0],
                                 hdi_prob=0.94,  # Intervalo de alta densidad (HDI)
                                 point_estimate="median",  # Muestra la mediana
                                 #ref_val=0,  # Línea vertical en 0 como referencia
                                 #rope=[0, 0.5],  # Opcional, agrega ROPE
                                 )
            
            ax_t[i,1].set_title(ax_t[i,1].get_title(), fontsize=14) 
            ax_t[i,1].tick_params(axis='both', labelsize=14) 
            ax_t[i,0].set_xlim(0,0.5)
            
        plt.tight_layout() 

        plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_trace_plots.png',dpi=300)    
        plt.show()


        #for i,a in enumerate(ax.flat):  # Esto recorre todos los subgráficos
        #    #a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')  # Gira los números
        #    a.set_title(a.get_title(), fontsize=14) 
        #    a.set_ylabel(a.get_ylabel(),fontsize=8)
        #    a.set_xlabel(a.get_xlabel(),fontsize=12)
        #    
        #    start, end = a.get_xlim()
        #    a.xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
        #    a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
        #    a.figure.tight_layout()  # Ajusta el layout




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
        import matplotlib.ticker as ticker
        # creating a prior sample
        n = 10
        
        prior_samples = self.get_prior_sample(n)
        prior_curves = self.get_prior_curves()
        
        idata_combined = az.from_dict(prior=prior_samples,
                                   posterior={var: self.samples[var] for var in self.labels},
                                   coords={"param": self.labels},
                                   dims={val:["chain", "draw"] for val in self.labels})



        ax = az.plot_dist_comparison(idata_combined, 
                                var_names=self.labels, 
                                labeller=self.labeller, figsize=(10,12))

        for i, label in enumerate(self.labels) :

            ax[i,0].clear()
            #for line in ax[i, 2].lines:
            #    if line.get_color() == 'C1':  # Identificamos las líneas de la prior
            #        line.remove()
            for line in ax[i, 2].lines:
                if line.get_color() == 'C1':  # Identificamos las líneas de la prior
                    line.remove()

            ax[i,0].plot(prior_curves[label][0],prior_curves[label][1],'C1', label='Prior')
            
            ymin = 0
            ymax = np.max(prior_curves[label][1])
            xmin, xmax = prior_curves[label][0][0], prior_curves[label][0][-1]

            for j in [0,2] :
                ax[i,j].set_xlim(xmin, xmax)

            for j in range(3) :
                ax[i,j].set_xlabel(ax[i,j].get_xlabel(),fontsize=12)
                ax[i,j].set_yticklabels(ax[i,j].get_yticklabels(), fontsize=10) 
                ax[i,j].set_xticklabels(ax[i,j].get_xticklabels(), fontsize=10)
                ax[i,j].figure.tight_layout()
                ax[i,j].grid()
            
            posterior_line, = ax[i,2].plot([],[], color='C0', label='Posterior')
            
            ax2 = ax[i,2].twinx()            
            prior_line, = ax2.plot(prior_curves[label][0],prior_curves[label][1],'C1', label='Prior')
            ax2.tick_params(labelcolor='C1')

            lines = [posterior_line, prior_line]  # Lista de líneas de ambos ejes
            labels = [line.get_label() for line in lines]
            ax[i, 2].legend(lines, labels, loc="best")

            #ax[i,0].legend()
            for j in [0,1] :
                 handles, labels = ax[i,j].get_legend_handles_labels()  # Obtener leyenda del subplot
                 if labels :
                     new_labels = [label.capitalize() for label in labels]
                     ax[i,j].legend(handles, new_labels, loc="best") 

            #ax[i,2].set_ylim(ymin, ymax)

            #start, end = ax[i,0].get_xlim()
            #ax[i,0].xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
            #ax[i,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
            #start, end = ax[i,2].get_xlim()
            #ax[i,2].xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
            #ax[i,2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))


        '''
        for i,a in enumerate(ax.flat):  # Esto recorre todos los subgráficos
            #a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')  # Gira los números
            #a.set_title(a.get_title(), fontsize=12) 
            #a.set_ylabel(a.get_ylabel(),fontsize=8)
            a.set_xlabel(a.get_xlabel(),fontsize=12)
            a.set_yticklabels(a.get_yticklabels(), fontsize=10) 
            a.set_xticklabels(a.get_xticklabels(), fontsize=10) 
            
            if i in [1,4,7] :
                start, end = a.get_xlim()
                a.xaxis.set_ticks(np.linspace(start, end, 5, endpoint=True))
                a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
            
            a.figure.tight_layout()  # Ajusta el layout
            a.grid()

            handles, labels = a.get_legend_handles_labels()  # Obtener leyenda del subplot
            if labels :
                new_labels = [label.capitalize() for label in labels]
                a.legend(handles, new_labels, loc="best")  # Asignar nuevos nombres'''



        plt.tight_layout() 
        plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_dist_comp.png',dpi=300)
        return
            
    
    def plot_energy(self) :
        
        az.plot_energy(self.idata, backend=self.backend)

        return
    
    
    def plot_rank_bars(self) :

        plt.figure()
        ax = az.plot_trace(self.idata, var_names=self.labels, kind='rank_bars', figsize=(7.5,5.5), labeller=self.labeller, backend=self.backend)

        # Ajustar los ticks de todos los subgráficos
        for a in ax.flat:  # Esto recorre todos los subgráficos
            #a.set_xticklabels(a.get_xticklabels(), rotation=45, ha='right')  # Gira los números
            a.set_title(a.get_title(), fontsize=16) 
            a.set_ylabel(a.get_ylabel(),fontsize=8)
            a.set_xlabel(a.get_xlabel(),fontsize=8)
            a.figure.tight_layout()  # Ajusta el layout

        plt.tight_layout() 
        plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_rank_bars.png',dpi=300)
        return
    
    def plot_forest(self) :
        
        az.plot_forest(self.idata,
                       var_names=self.labels,
                       ess=True,
                       hdi_prob=0.95,
                       labeller=self.labeller,
                       backend=self.backend)
        
        return
    
    def plot_parallel(self) :
        
        az.plot_parallel(self.idata, var_names=self.labels, labeller=self.labeller, backend=self.backend)
        
        return
    
    def plot_autocorr(self) :
        
        az.plot_autocorr(self.idata, var_names=self.labels, labeller=self.labeller, backend=self.backend)
        
        return 
    
    def summary(self) :
        
        print(az.summary(self.idata))
        
        return
    
    def generate_from_simulated_data(self, num_samples) :
        
        draw_idx = np.random.choice(self.idata.posterior.draw.size, num_samples, replace=False)
        #print(draw_idx)
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
    
    
    def plot_ppc(self, n=0.5) :

        if n < 1 :
            n = int(n * self.idata.posterior.draw.size)

        print(f'Using {n} samples')

        posterior_predictive = self.generate_from_simulated_data(n)

        q25 = np.quantile(posterior_predictive, 0.25, axis=0)
        q75 = np.quantile(posterior_predictive, 0.75, axis=0)
        #m = posterior_predictive.shape[1]

        #q25 = np.zeros(m)
        #q75 = np.zeros(m)

        #for t in range(m) :
        #    q25[t] = np.percentile(posterior_predictive[:,t],25)
        #    q75[t] = np.percentile(posterior_predictive[:,t],75)
        
        median = np.median(posterior_predictive, axis=0)
        mean_predictive = np.mean(posterior_predictive, axis=0)

        map_theta, map_estimate = self.get_map() 

        plt.figure(figsize=(6.,4.0))
        plt.plot(self.time, self.data, label='Data')
        plt.plot(self.time, posterior_predictive.T, alpha=0.2, lw=0.25, color='gray')
        #for curve in posterior_predictive :
        #    plt.plot(self.time, curve, alpha=0.2, lw=0.25, color='gray')
        plt.plot([], [], alpha=0.2, lw=1, color='gray',label='Probability region')
        #plt.plot(self.time, posterior_predictive, alpha=0.2, lw=3, color='gray',label='Probability region')
        plt.plot(self.time, mean_predictive, color='tab:orange', lw=2, alpha=0.8, label='Predictive mean',zorder=11)
        plt.plot(self.time, map_estimate, '--', color='tab:green', lw=2, label='MAP estimate',zorder=12)
        plt.fill_between(self.time, q25, q75, color="skyblue", alpha=0.5, label="IQR (25%-75%)", zorder=10)
        plt.grid()
        plt.legend()
        plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_probability_region.png',dpi=300)
        #plt.savefig(self.outpath+'/'+self.instance+'/'+self.instance+'_probability_region.jpg',dpi=300)
        plt.show()

        return
        
    def get_map(self) :

        energy = self.idata.sample_stats.energy.values
        if self.instance == 'emcee' :
            map_index = np.unravel_index(np.argmax(energy), energy.shape)
        elif self.instance == 'pytwalk' :
            map_index = np.unravel_index(np.argmin(energy), energy.shape)
        else :
            map_index = np.unravel_index(np.argmin(energy), energy.shape)

        #print(map_index)
        #print(energy[map_index])

        map_theta = [self.idata.posterior[var].values[map_index] for var in self.idata.posterior.keys()]

        map_estimate = self.forward_map(map_theta)

        return map_theta, map_estimate
    
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
        self.plot_posterior_and_traces()
        return
    

    