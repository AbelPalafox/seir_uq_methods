#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:50:48 2025

@author: abel
"""
from nn_models import SEIR_PINN
from BNN import BNN
import matplotlib.pyplot as plt
import numpy as np
from epidemic_model import SEIR_Model
from utils.utils import load_data, parser
import sys

if sys.gettrace() :
    args = ['configuration_file_bnn.cfg']
else :
    args = sys.argv[1:]

fname_conf = args[0]
params = parser(fname_conf)

times, data, dates = load_data(params['fname'])

print(f'Loaded data from file {params["fname"]}')

plt.figure()
plt.plot(dates,data)
plt.grid()
plt.show()

N = params["N"]

E0 = 0 
I0 = data[0]
R0 = 0
S0 = N - E0 - I0 - R0

x0 = [S0,E0,I0,R0]

model = BNN(SEIR_PINN,1,3,params['n_hidden'],params['n_flayers'])

beta = params['beta_0']
sigma = params['sigma_0']
gamma = params['gamma_0']

n_iterations = params['iterations']
sampler = params['sampler']
lambda_data = params['lambda_data']
lambda_cond = params['lambda_cond']
lambda_eq = params['lambda_eq']
sigma = params['sigma']
bounds = [[params['beta_min'], params['beta_max']], 
    [params['sigma_min'], params['sigma_max']], 
    [params['gamma_min'], params['gamma_max']]]


model.infer_parameters(n_iterations, 
                       sampler=sampler,
                       params=[beta,sigma,gamma],
                       data=data,
                       t=times,
                       lambda_data=lambda_data,
                       lambda_cond=lambda_cond,
                       lambda_eq=lambda_eq,
                       sigma=sigma,
                       bounds=bounds,
                       S0=S0,
                       E0=E0,
                       I0=I0,
                       R0=R0,
                       N=N,
                       likelihood='Poisson'
                       )


beta_sample = [np.exp(_) for _ in model.Output[:,:,-3].flatten()]
sigma_sample = [np.exp(_) for _ in model.Output[:,:,-2].flatten()]
gamma_sample = [np.exp(_) for _ in model.Output[:,:,-1].flatten()]



beta_mean = np.mean(beta_sample)
sigma_mean = np.mean(sigma_sample)
gamma_mean = np.mean(gamma_sample)

print(beta_mean, sigma_mean, gamma_mean)

plt.figure()
plt.plot(times, data, 'ko', label='Datos')

for beta_i, sigma_i, gamma_i in zip(beta_sample[-50:], sigma_sample[-50:], gamma_sample[-50:]) :
    
    seir = SEIR_Model((beta_i), (sigma_i), (gamma_i), N)
    
    x0 = np.array([S0, E0, I0, R0])
    
    x = seir.run(x0, times)
        
    plt.plot(times, x[2,:], 'g', alpha=0.2, lw=3)

    
plt.grid()
plt.ylim(0,1)
plt.show()


'''
plt.hist((beta_sample))
plt.show()
plt.hist(sigma_sample)
plt.show()
plt.hist(gamma_sample)
plt.show()


plt.plot(beta_sample)
plt.show()

plt.plot(sigma_sample)
plt.show()

plt.plot(gamma_sample)
plt.show()


beta_mean = np.mean(beta_sample)
sigma_mean = np.mean(sigma_sample)
gamma_mean = np.mean(gamma_sample)

print(beta_mean, sigma_mean, gamma_mean)

plt.figure()
plt.plot(t_data, I_data, 'ko', label='Datos')

for beta_i, sigma_i, gamma_i in zip(beta_sample[-50:], sigma_sample[-50:], gamma_sample[-50:]) :
    
    seir = SEIR_Model((beta_i), (sigma_i), (gamma_i), N)
    
    x0 = np.array([S0, E0, I0, R0])
    
    x = seir.run(x0, t_data)
        
    plt.plot(t_data, x[2,:], 'g', alpha=0.2, lw=3)
    #plt.plot(t_data, x[1,:], 'r', alpha=0.2, lw=3)
    #plt.plot(times, x[0,:], 'steelblue', label='Estimación S')
    #plt.plot(times, x[3,:], 'k', label='Estimación R')
    
plt.grid()
#plt.legend()
plt.ylim(0,1)
plt.show()

'''