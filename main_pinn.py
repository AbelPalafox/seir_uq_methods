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

input_data_file = 'test_data.csv'
# reading data into a dataframe
df_input = read_csv(input_data_file)

print(df_input)

times = np.asarray(df_input['time'])
data= df_input['I']

# setting initial values
N = 100000

E0 = 0 
I0 = data[0]
R0 = 0
S0 = (N - E0 - I0 - R0)

x0 = [S0/N,E0/N,I0/N,R0/N]


#times = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.float32).view(-1, 1)
#data = torch.tensor([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4], dtype=torch.float32).view(-1, 1)

#times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#data = [1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4]


# Condiciones iniciales
#S0, E0, I0, R0 = 762, 0, 1, 0
#N = S0 + E0 + I0 + R0  # Población total

#x0 = [S0,E0,I0,R0]

model = SEIR_PINN()

beta, sigma, gamma = model.run(times,
                               data/N,
                               x0,
                               params_0 = [0.5, 0.5, 0.5],
                               #params_0 = [-0.00020526449952740222, -0.011816229671239853, 0.0008839285583235323],
                               epochs=200000,
                               lr=1e-3)



print(beta, sigma, gamma)
beta, sigma, gamma = [0.4910216478152676, 0.2800019137444663, 0.22897669226675946]
seir = SEIR_Model(beta, sigma, gamma, 1.0)

x = seir.run(x0, times)

seir.plot(times,x)

plt.plot(times, data/N, label='Datos')
plt.plot(times, x[2,:], 'g', label='Estimación Exp')
plt.plot(times, x[1,:], 'r', label='Estimación Inf')
#plt.plot(times, x[0,:], 'steelblue', label='Estimación S')
#plt.plot(times, x[3,:], 'k', label='Estimación R')

plt.grid()
plt.legend()
plt.show()
