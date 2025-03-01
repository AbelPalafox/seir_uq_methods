#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:19:12 2025

@author: abel
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy.stats as stats
#from scipy.stats import multivariate_normal
from numpy import linalg 
#import matplotlib.pyplot as plt
#from functools import partial
import scipy.linalg
from time import time
#from numba import jit


# @jit(nopython=True, fastmath=True)        
# def fd_gradient(U,theta,dim,h) :

#     grad = np.zeros(dim)
    
#     for i in range(dim) :
#         theta_fw = theta.copy()
#         theta_bw = theta.copy()
        
#         theta_fw[i] += h
#         theta_bw[i] -= h
        
#         grad[i] = (U(theta_fw) - U(theta_bw)) / (2.0 * h)
        
#     return grad



class pyhmc :
    
    def __init__(self, loglikelihood, logprior, support, *argv, **kwargs) :
        
        self.ndim = kwargs['ndim'] 
        self.h = kwargs['h']
        self.step_size = kwargs['step_size']
        self.nsteps = kwargs['nsteps']
        self.params = kwargs
        self.loglikelihood = loglikelihood
        self.logprior = logprior
        self.support = support
        
        return
    
    def U(self, theta) :
        
        if not self.support(theta):
            return np.inf 
        
        return self.loglikelihood(theta) + self.logprior(theta)
    
    
    def gradient(self, theta) :
        
        theta = np.asarray(theta,dtype=float)
        dim = self.ndim
        
        h = self.h
        
        diff_matrix = np.eye(dim)*h

        U_fw = np.apply_along_axis(self.U, 1, theta+diff_matrix)
        U_bw = np.apply_along_axis(self.U, 1, theta-diff_matrix)
        
        grad = (U_fw - U_bw) / (2.0*h)
        
        return grad
    
    def hessian(self, theta) :

        theta = np.asarray(theta, dtype=float)

        dim = self.ndim
        h = self.h
        
        H = np.zeros((dim, dim))
            
        diff_matrix = np.eye(dim)*h
        
        # Evaluaciones para segundas derivadas cruzadas (H[i, j], i ≠ j)
        #f_pp = np.array([self.U(theta + diff_matrix[i] + diff_matrix[j]) for i in range(dim) for j in range(i, dim)]).reshape(dim, -1)
        #f_pm = np.array([self.U(theta + diff_matrix[i] - diff_matrix[j]) for i in range(dim) for j in range(i, dim)]).reshape(dim, -1)
        #f_mp = np.array([self.U(theta - diff_matrix[i] + diff_matrix[j]) for i in range(dim) for j in range(i, dim)]).reshape(dim, -1)
        #f_mm = np.array([self.U(theta - diff_matrix[i] - diff_matrix[j]) for i in range(dim) for j in range(i, dim)]).reshape(dim, -1)
        f_pp = np.array([self.U(theta + diff_matrix[i] + diff_matrix[j]) for i in range(dim) for j in range(i, dim)])
        f_pm = np.array([self.U(theta + diff_matrix[i] - diff_matrix[j]) for i in range(dim) for j in range(i, dim)])
        f_mp = np.array([self.U(theta - diff_matrix[i] + diff_matrix[j]) for i in range(dim) for j in range(i, dim)])
        f_mm = np.array([self.U(theta - diff_matrix[i] - diff_matrix[j]) for i in range(dim) for j in range(i, dim)])
    
    
        # Cálculo de segundas derivadas cruzadas
        H[np.triu_indices(dim)] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
        H += np.triu(H, 1).T  # Rellenar la parte inferior por simetría
    
        # Evaluaciones para segundas derivadas puras (H[i, i])
        f_p = np.array([self.U(theta + diff_matrix[i]) for i in range(dim)])
        f_m = np.array([self.U(theta - diff_matrix[i]) for i in range(dim)])
        f_0 = self.U(theta)
    
        # Cálculo de derivadas puras
        H[np.diag_indices(dim)] = (f_p - 2 * f_0 + f_m) / (h**2)
    
        return H
    

    def leapfrog(self, q_,p_,Weight) :
        
        #print('entering leapfrog')
        step_size = self.step_size
        
        q = np.asarray(q_,dtype=float).copy()
        p = np.asarray(p_,dtype=float).copy()
        
        p -= step_size * self.gradient(q) / 2.0         # half step
        
        L = scipy.linalg.cho_factor(Weight, lower=True) 
        
        for _ in range(self.nsteps) :
            q += step_size * scipy.linalg.cho_solve(L, p) # whole step
            p -= step_size * self.gradient(q)           # whole step
        
        q += step_size * scipy.linalg.cho_solve(L, p)     # whole step
        p -= step_size * self.gradient(q) / 2.0         # half step
    
        Start_log_p = self.U(q_) + 0.5*p_.dot(scipy.linalg.cho_solve(L, p_))
        New_log_p = self.U(q) + 0.5*p.dot(scipy.linalg.cho_solve(L, p))
        
        if np.log(np.random.rand()) < Start_log_p - New_log_p:
            return q, 1
        else:
            return q, 0
        
    def Run(self, n_samples, theta_0) :
        
        dim = self.ndim
        
        samples = []
        U_samples = [] 
        
        samples.append(theta_0)
        U_samples.append(self.U(theta_0))
        
        Opt = samples[0]
        Value = U_samples[0]
        
        Weight = self.get_weight_matrix(theta_0)
        
        p0 = np.random.multivariate_normal(np.zeros(dim), Weight, 1)
        p0 = p0.reshape(dim,)
        
        print('* ', theta_0)
        q_new, salida = self.leapfrog(
            theta_0,
            p0,
            Weight,
            )
        print('** ', theta_0)
        
        samples.append(q_new)
        U_new = self.U(q_new)
        
        if U_new < Value :
            Opt = q_new
            Value = U_new
        
        
        for i in range(n_samples) :
            
            #print(f'iteration: {i}')
            
            theta = samples[-1]
            
            ini_time = time()
            Weight = self.get_weight_matrix(theta)
            end_time = time()
            #print(f'elapsed time weight matrix: {end_time - ini_time}')
            p0 = np.random.multivariate_normal(np.zeros(dim), Weight, 1)
            p0 = p0.reshape(dim,)
            
            ini_time = time()
            q_new, salida = self.leapfrog(
                theta,
                p0,
                Weight,
                )
            end_time = time()
            #print(f'elapsed time leapfrog: {end_time - ini_time}')
            U_new = self.U(q_new)
            
            if U_new < Value :
                Opt = q_new
                Value = U_new

            if salida == 1 :
                samples.append(q_new)
                U_samples.append(U_new)
            else :
                samples.append(theta)
                U_samples.append(U_samples[-1])
            
            print(q_new)
            yield q_new
            
        self.Output = np.asarray(samples)
        self.U_samples = np.asarray(U_samples)
        
        return np.asarray(samples), np.asarray(U_samples), Opt, Value
        
    def get_weight_matrix(self, theta) :
        
        H = self.hessian(theta)
        
        AutoVal, AutoVect = linalg.eig(H)
        
        if np.any(AutoVal == 0) :
            return np.eye(self.ndim)
        
        if np.any(AutoVal < 0) :
            AutoVal = np.abs(AutoVal)
            return AutoVect @ np.diag(AutoVal) @ AutoVect.T
        
        return H
        
if __name__ == '__main__' :

    def neg_log(x):
        return np.sum(x**2)

    #np.random.seed(0)

    n_samples = 1000
    initial_position=np.ones([4,])*50
    negative_log_prob = neg_log
    NumberSteps = 8
    step_size = 0.8


    hm_mcmcm_instance = pyhmc(negative_log_prob,ndim=1,h=1e-4,step_size=step_size,nsteps=NumberSteps)
    
    sample, Values, Opt, OptVal = hm_mcmcm_instance.Run(n_samples, initial_position)
    
    plt.plot(sample[:,0],sample[:,1],'.')

    print(Opt)

       
        
        
        
        
        
        
        
        
        
        
        