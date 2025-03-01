#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:37:08 2025

@author: abel
"""

import numpy as np
#from pandas import read_csv
from samplers.SEIR_emcee import SEIR_emcee
from samplers.SEIR_pytwalk import SEIR_pytwalk
from samplers.SEIR_pyhmc import SEIR_pyhmc
from epidemic_model import SEIR_Model
from utils.utils import load_data, parser
import sys
import matplotlib.pyplot as plt

if sys.gettrace() :
    args = ['configuration_file_test.cfg']
else :
    args = sys.argv[1:]

fname_conf = args[0]
params = parser(fname_conf)

times, data, dates = load_data(params['fname'])

#input_data_file = 'test_data.csv'
## reading data into a dataframe
#df_input = read_csv(input_data_file)

#print(df_input)

#times = np.asarray(df_input['time'])
#data= df_input['I']

print(f'Loaded data from file {params["fname"]}')

plt.figure()
plt.plot(dates,data)
plt.grid()
plt.show()

if 'N' not in params :
    N = np.sum(data)/.8

print(f'Data will be normalized by factor {params["N"]}')

N = params["N"]
#data = N_true

E0 = 0 
I0 = data[0]
R0 = 0
S0 = N - E0 - I0 - R0


x0 = [S0,E0,I0,R0]

option = params['option']
iterations = params['iterations']
ndim = 3 # beta, sigma, gamma

if option == 1 :
    print('Running the emcee implementation')
    
    nwalkers = 6
     # three parameters beta, sigma, gamma 
    print(f'setting walkers to 2 x dim = {nwalkers}')
    
    seir_emcee = SEIR_emcee(
        ndim = ndim,
        nwalkers = nwalkers,
        data=data,
        time=times,
        labels=['beta', 'sigma', 'gamma'],
        x0=x0,
        **params
        )
    
    # initial guess
    #p0 = np.array([0.36,0.98,0.2])
    p0 = np.array([0.25,0.5,0.1])
    theta_0 = p0+np.random.randn(nwalkers, ndim) * 1e-1
    
    seir_emcee.run(iterations, theta_0)
    
    #seir_emcee.create_dictionary(400,2,-1)
    seir_emcee.plot_densities()
    seir_emcee.plot_histograms()
    seir_emcee.plot_posterior()
    seir_emcee.plot_pairs()
    seir_emcee.trace_plot()
    seir_emcee.plot_prior_vs_posterior()
    #seir_emcee.plot_dist_comparison()
    
    plt.figure()
    plt.plot(times, data, 'o', label='data')
    #plt.plot(times,I,label='true')
    
    for i in range(1,150,5) :
        beta_ = seir_emcee.samples['beta'][-i,0]
        sigma_ = seir_emcee.samples['sigma'][-i,0]
        gamma_ = seir_emcee.samples['gamma'][-i,0]
    
        sample_seir = SEIR_Model(beta_, sigma_, gamma_, N)
        S_, E_, I_, R_ = sample_seir.run(x0, times)

        #plt.plot(times, I_, 'green', alpha=.2, lw=3)
        
        incidency = -sample_seir.incidency(np.hstack([[N],S_]),times,1)
        plt.plot(times, incidency, 'green', alpha=.2, lw=3)
        
        print(seir_emcee.lnprob([beta_, sigma_, gamma_]), beta_, sigma_, gamma_)
        
    plt.grid()
    plt.legend()
    plt.show()
    

elif option == 2 :

    print('Testing the twalk implementation')
    
    seir_twalk = SEIR_pytwalk(
        ndim=ndim,
        data=data,
        time=times,
        x0=x0,
        labels=['beta', 'sigma', 'gamma'],
        **params
        )
    
    xp0 = np.array([0.36,0.98,0.2])
    xp1 = np.array([0.5,1.00,0.15])
    
    seir_twalk.run(iterations,xp0,xp1)
    
    seir_twalk.plot_densities()
    seir_twalk.plot_histograms()
    seir_twalk.plot_posterior()
    seir_twalk.plot_pairs()
    seir_twalk.trace_plot()
    seir_twalk.plot_prior_vs_posterior()
    #seir_twalk.plot_energy()

    plt.figure()
    plt.plot(times, data, 'o', label='data')
    
    for i in range(1,150,5) :
        beta_ = seir_twalk.samples['beta'][-i]
        sigma_ = seir_twalk.samples['sigma'][-i]
        gamma_ = seir_twalk.samples['gamma'][-i]
    
        sample_seir = SEIR_Model(beta_, sigma_, gamma_, N)
        S_, E_, I_, R_ = sample_seir.run(x0, times)

        incidency = -sample_seir.incidency(np.hstack([[N],S_]),times,1)
        plt.plot(times, incidency, 'green', alpha=.2, lw=3)
        
        
    plt.grid()
    plt.legend()
    plt.show()
    
elif option == 3 :
        
    print('Testing the HMC implementation')
    
    seir_hmc = SEIR_pyhmc(
        N = N,
        ndim=ndim,
        data=data,
        time=times,
        x0=x0,
        alpha_beta_prior=1.0,
        beta_beta_prior = 1.0,
        alpha_sigma_prior = 1.0,
        beta_sigma_prior = 1.0,
        alpha_gamma_prior = 1.0,
        beta_gamma_prior = 1.0,
        labels=['beta', 'sigma', 'gamma'],
        likelihood_model = 'Gaussian',
        h = 1e-4,
        step_size = 0.8,
        nsteps = 4,
        prior_model = 'Beta',
        beta_min = 0.0,
        beta_max = 1.0,
        sigma_min = 0.0,
        sigma_max = 1.0,
        gamma_min = 0.0,
        gamma_max = 1.0,
        )
    
    xp0 = np.array([0.5,0.5,0.5])
    seir_hmc.run(iterations,xp0)
    
    seir_hmc.plot_densities()
    seir_hmc.plot_histograms()
    seir_hmc.plot_posterior()
    seir_hmc.plot_pairs()








