#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:44:23 2025

@author: abel
"""
import numpy as np
#import matplotlib.pyplot as plt
from pandas import read_csv
from nn_models.SEIR_PINN import SEIR_PINN
from epidemic_model.SEIR_Model import SEIR_Model
import matplotlib.pyplot as plt
import torch

input_data_file = 'data/Smoothed_Cases_Third_Wave.csv'
# reading data into a dataframe
df_input = read_csv(input_data_file)

print(df_input)

#times = np.asarray(df_input['time'])
#data= df_input['I']
dates = df_input['Date']
data = df_input['Smoothed_Daily_Cases']
times = np.arange(0,len(dates))

# setting initial values
N = 170000

E0 = 0 
I0 = data[0]
R0 = 0
S0 = (N - E0 - I0 - R0)

x0 = np.array([S0,E0,I0,R0])

print(x0)

#times = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.float32).view(-1, 1)
#data = torch.tensor([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4], dtype=torch.float32).view(-1, 1)

#times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#data = [1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4]


# Condiciones iniciales
#S0, E0, I0, R0 = 762, 0, 1, 0
#N = S0 + E0 + I0 + R0  # Población total

#x0 = [S0,E0,I0,R0]

model = SEIR_PINN(RFF=False)

beta, sigma, gamma = model.run(times,
                               data,
                               x0,
                               params_0 = [],
                               #params_0 = [-0.00020526449952740222, -0.011816229671239853, 0.0008839285583235323],
                               epochs=50000,
                               lr=1e-3,
                               N=N, inc_method='susceptible', likelihood='Poisson', init_cond='fixed')



plt.figure()
plt.plot(model.data_losses)
plt.grid()
plt.title('Data losses')
plt.show()

plt.figure()
plt.plot(model.eq_losses)
plt.grid()
plt.title('Equation losses')
plt.show()

plt.figure()
plt.plot(model.cond_losses)
plt.grid()
plt.title('Initial conditions losses')
plt.show()

params_track = np.exp(np.array(model.params_track))

plt.figure()
plt.plot(params_track)
plt.grid()
plt.title('params track')
plt.show()



print(beta, sigma, gamma)
#beta, sigma, gamma = [0.4910216478152676, 0.2800019137444663, 0.22897669226675946]
#beta, sigma, gamma = [0.4976092079426683, 0.16907448091279353, 0.18131287897970613]
seir = SEIR_Model(beta*N, sigma*N, gamma*N, N)

x = seir.run(x0, times)

incidency = -seir.incidency(np.hstack([[N],x[0,:]]),times,1,method='susceptible')

#seir.plot(times,x)
plt.figure()
plt.plot(times, data, label='Datos')
plt.plot(times,incidency, label='incidency')
#plt.plot(times, x[2,:], 'g', label='Estimación Exp')
plt.plot(times, x[1,:], 'r', label='Estimación Inf')
plt.plot(times, x[0,:], 'steelblue', label='Estimación S')
#plt.plot(times, x[3,:], 'k', label='Estimación R')
plt.title('odeint sol vs Data')
plt.grid()
plt.legend()
plt.show()

plt.figure()
times_tensor = torch.tensor(times, dtype=torch.float32).view(-1, 1)
y_pred = model.forward(times_tensor)

S_pred_denormalized = model.normalizer.denormalize(y_pred[:,0])

incidency = -model.compute_incidency(S_pred_denormalized, N)

plt.plot(times, data, label='Data')
plt.plot(times,N*y_pred[:,0], label='susceptible')
#plt.plot(times,N*y_pred[:,1], label='exposed')
#plt.plot(times,N*y_pred[:,2], label='infected')
with torch.no_grad() :
    #plt.plot(times,S_pred_denormalized,label='susceptible')
    plt.plot(times,incidency,label='incidency')
plt.title('Model vs Data')
plt.grid()
plt.legend()
plt.show()


