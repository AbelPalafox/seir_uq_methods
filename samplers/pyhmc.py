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
from joblib import Parallel, delayed

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

        self.out_support = 0
        
        return
    
    def U(self, theta) :
        
        if not self.support(theta):
            return 1e8 
        
        return 2*(self.loglikelihood(theta) + self.logprior(theta))/self.params['N']
    
    
    def gradient(self, theta) :
        
        theta = np.asarray(theta,dtype=float)
        dim = self.ndim
        
        h = self.h
        
        diff_matrix = np.eye(dim)*h

        #
        #U_fw = np.apply_along_axis(self.U, 1, theta+diff_matrix)
        #U_bw = np.apply_along_axis(self.U, 1, theta-diff_matrix)
        def eval_U_shift(i) :
            return self.U(theta + diff_matrix[i]), self.U(theta - diff_matrix[i])
        
        results = Parallel(n_jobs=dim)(delayed(eval_U_shift)(i) for i in range(dim))

        U_fw, U_bw = zip(*results)

        grad = (np.array(U_fw) - np.array(U_bw)) / (2.0*h)
        
        #print(grad)

        return grad
    
    def hessian(self, x0) :

        tam = np.shape(x0)[0]
        Hess = np.zeros([tam,tam])
        h = self.h
        Hj = np.zeros([tam])
        Hi = np.zeros([tam])
        
        f_x0 = self.U(x0)
        #print('*** ', x0, f_x0)
        for i in range(tam):    
            Hi[i] = h
            for j in range(i):
                Hj[j] = h
                Hess[i,j] = (self.U(x0+Hi+Hj)+self.U(x0-Hi-Hj)-self.U(x0-Hi+Hj)-self.U(x0+Hi-Hj))/(4*h**2)
                Hess[j,i] = Hess[i,j]
                Hj[j] = 0.0           
            Hess[i,i] = (self.U(x0+Hi)-2*f_x0+self.U(x0-Hi))/(h**2)
            Hi[i] = 0.0
    
        return Hess
    

    def leapfrog(self, q_,p_,WeightInv, U_q_) :

        step_size = self.step_size
        dim = len(q_)
        
        q = np.asarray(q_,dtype=float).copy()
        p = np.asarray(p_,dtype=float).copy()

        Start_log_p = U_q_  + 0.5* p_ @ WeightInv @ p_

        for _ in range(self.nsteps) :
            p -= 0.5*step_size*self.gradient(q) 

            q += step_size * WeightInv @ p 
            if not self.support(q) :
                q = self.project_params(q)
            
            p -= 0.5 * step_size * self.gradient(q) 

            U_q = self.U(q)
            New_log_p = U_q + 0.5* p @ WeightInv @ p

            if np.log(np.random.rand()) < Start_log_p - New_log_p:
                return q, U_q, 1
            
        U_q = self.U(q)
        New_log_p = U_q + 0.5* p @ WeightInv @ p

        if np.log(np.random.rand()) < Start_log_p - New_log_p:
            return q, U_q, 1
        
        return q, U_q, 0
        

    def leapfrog_(self, q_,p_,WeightInv, U_q_) :
        
        #print('entering leapfrog')
        step_size = self.step_size
        dim = len(q_)
        
        q = np.asarray(q_,dtype=float).copy()
        p = np.asarray(p_,dtype=float).copy()
        
        p -= step_size * self.gradient(q) / 2.0         # half step
        
        #L = scipy.linalg.cho_factor(Weight, lower=True) 
        Start_log_p = U_q_ + 0.5* p_ @ WeightInv @ p_
        
        if not np.isfinite(p).all() :
            print(-1)
        
        for _ in range(self.nsteps) :
            q += step_size * WeightInv @ p # whole step
            if not np.isfinite(p).all() or not np.isfinite(q).all():
                print(0.1, q, WeightInv, p)
            # check for points out of support
            while not self.support(q) :
                q = q_ + step_size*np.random.randn(dim)
                self.out_support += 1

            p -= step_size * self.gradient(q)           # whole step

            if not np.isfinite(p).all() or not np.isfinite(q).all() :
                print(0.2, q, WeightInv, p, self.gradient(q))

            U_q = self.U(q)
            New_log_p = U_q + 0.5* p @ WeightInv @ p
            #print(f'*** {_}: ', Start_log_p - New_log_p, Start_log_p, New_log_p)
            if np.log(np.random.rand()) < Start_log_p - New_log_p:
                return q, U_q, 1

        if not np.isfinite(p).all() :
            print(0)

        q += step_size * WeightInv @ p     # whole step
        while not self.support(q) :
            q = q_ + step_size*np.random.randn(dim)
            self.out_support += 1

        p -= step_size * self.gradient(q) / 2.0         # half step
    
        U_q = self.U(q)
        Start_log_p = U_q_ + 0.5* p_ @ WeightInv @ p_
        New_log_p = U_q + 0.5* p @ WeightInv @ p
        
        if not np.isfinite(q).all() :
            print(1)

        if not np.isfinite(WeightInv).all() :
            print(2)

        if np.log(np.random.rand()) < Start_log_p - New_log_p:
            return q, U_q, 1
        else:
            
            return q, U_q, 0
        
    def Run(self, n_samples, theta_0) :
        
        import sys

        sys.stdout.flush()

        dim = self.ndim
        
        samples = []
        U_samples = [] 
        
        samples.append(theta_0)
        U_samples.append(self.U(theta_0))
        
        Opt = samples[0]
        Value = U_samples[0]
        
        Weight, WeightInv = self.get_weight_matrix_inv(theta_0)
        
        p0 = np.random.multivariate_normal(np.zeros(dim), Weight, 1)
        p0 = p0.reshape(dim,)
        
        #print('* ', theta_0)
        q_new, U_q_new, salida = self.leapfrog(
            theta_0,
            p0,
            WeightInv,
            Value
            )
        #print('** ', theta_0)
        self.reject_counter = 0
    
        samples.append(q_new)
        U_new = U_q_new
        
        if U_new < Value :
            Opt = q_new
            Value = U_new
        
        
        for i in range(n_samples) :
            
            #print(f'iteration: {i}')
            
            theta = samples[-1].copy()
            
            #ini_time = time()
            Weight, WeightInv = self.get_weight_matrix_inv(theta)
            #end_time = time()
            #print(f'elapsed time weight matrix: {end_time - ini_time}')
            p0 = np.random.multivariate_normal(np.zeros(dim), Weight, 1)
            p0 = p0.reshape(dim,)
            
            ini_time = time()
            q_new, U_q_new, salida = self.leapfrog(
                theta,
                p0,
                WeightInv,
                U_samples[-1].copy()
                )
            end_time = time()
            #print(f'elapsed time leapfrog: {end_time - ini_time}')
            U_new = U_q_new
            #print(U_new, Value)
            if U_new < Value :
                Opt = q_new
                Value = U_new

            if salida == 1 :
                samples.append(q_new)
                U_samples.append(U_new)
            else :
                self.reject_counter += 1
                samples.append(samples[-1])
                U_samples.append(U_samples[-1])
            
            if i%50 == 0 :
                print(samples[-1], U_samples[-1], self.reject_counter, self.out_support)
            yield 
            
        self.Output = np.asarray(samples)
        self.U_samples = np.asarray(U_samples)
        
        return #np.asarray(samples), np.asarray(U_samples), Opt, Value
        
    def get_weight_matrix_inv(self, theta) :
        
        H = self.hessian(theta)
        
        if not np.isfinite(H).all() :
            return np.eye(self.ndim), np.eye(self.ndim)
        #print(H)

        AutoVal, AutoVect = linalg.eig(H)
        
        if np.iscomplex(AutoVal).any() :
            print('1')
            AutoVal = np.abs(AutoVal)
            AutoVal = np.clip(AutoVal, 1e-6, None)
            return AutoVal*np.eye(self.ndim), np.eye(self.ndim)/AutoVal

        AutoVal = np.abs(AutoVal)

        if np.any(np.abs(AutoVal) <1e-6) :
            print(2)
            AutoVal = np.clip(AutoVal, 1e-6, None)
            return AutoVal*np.eye(self.ndim), np.eye(self.ndim)/AutoVal
        
        return AutoVect @ np.diag(AutoVal) @ AutoVect.T, AutoVect @ np.diag(1.0/AutoVal) @ AutoVect.T
        

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

       
        
        
        
        
        
        
        
        
        
        
        