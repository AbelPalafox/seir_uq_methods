#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:35:47 2025

@author: abel
"""

#from PINN.PINN import PINN
import torch
import numpy as np
from pytwalk import pytwalk
import emcee
from tqdm import tqdm
from samplers.pyhmc import pyhmc
from utils import Normalizer
import copy
import pickle

class BNN :
    
    def __init__(self, pinn_model, system_model, n_input, n_output, **kwargs) :
        
        n_hidden = kwargs['n_hidden']
        n_flayers = kwargs['n_flayers']
        self.model = pinn_model()#pinn_model(n_input, n_output, n_hidden, n_flayers)
        self.system_model = system_model(**kwargs)
        self.args = kwargs
        self.call_counter = 0

        self.data_losses = []
        self.cond_losses = []
        self.eq_losses = []
        self.losses = []


        return 
    
    # def set_weights(self, weights) :
        
    #     with torch.no_grad() :
    #         for param, new_value in zip(self.model.parameters(), weights):
    #             param.copy_(torch.tensor(new_value, dtype=param.dtype))     
    #     return

    def get_params_vector(self) :
        
        # params = []
        # shapes = []
        # for param in self.model.parameters() :
        #     shapes.append(param.shape)
        #     params.append(param.view(-1))
        
        return np.concatenate([p.detach().numpy().flatten() for p in self.model.parameters()])

            
        #return torch.cat(params), shapes
    
    # def set_params_vector(self, flat_params, shapes) :
        
    #     with torch.no_grad() :
    #         index = 0
    #         for param, shape in zip(self.model.parameters(), shapes) :
    #             size = torch.prod(torch.tensor(shape)).item()
    #             new_value = flat_params[index: index + size].view(shape)
    #             param.copy_(new_value)
    #             index += size
        
    #     return 

    def set_params_vector(self, params_vector) :

        start = 0
        with torch.no_grad():  # Desactivar gradientes al actualizar pesos manualmente
            for p in self.model.parameters():
                size = p.numel()  # Número de elementos en el tensor
                new_values = params_vector[start:start + size].reshape(p.shape)  # Ajustar forma
                p.copy_(torch.tensor(new_values, dtype=torch.float32))  # Copiar valores
                start += size

        
    def likelihood(self, theta_) :
        
        lambda_data = self.model.args['lambda_data']
        lambda_cond = self.model.args['lambda_cond']
        lambda_eq = self.model.args['lambda_eq']

        theta = theta_[:-self.nparams]
        params = theta_[-self.nparams:]

        # computing loss
        with torch.no_grad() :
            # using proposal weights
            self.set_params_vector(theta)
        
        model_prediction = self.model.forward(self.t) 

        self.model.log_beta = torch.log(torch.tensor(params[0], dtype=torch.float32))#, requires_grad=True))
        self.model.log_sigma = torch.log(torch.tensor(params[1], dtype=torch.float32))#, requires_grad=True))
        self.model.log_gamma = torch.log(torch.tensor(params[2], dtype=torch.float32))#, requires_grad=True))
        eq_loss, dsystem_dt = self.model.compute_eq_loss(model_prediction, self.t)
        data_loss = self.model.compute_data_loss(model_prediction, self.data, self.t)
        cond_loss = self.model.compute_cond_loss(model_prediction, self.data)
    
        loss = 1e8*(lambda_eq*eq_loss + lambda_data*data_loss + lambda_cond*cond_loss)
        
        self.data_losses.append(data_loss.item())
        self.cond_losses.append(cond_loss.item())
        self.eq_losses.append(eq_loss.item())
        self.losses.append(loss.item())

        return loss.item()


    def prior(self, theta_) :
        
        theta = theta_[:-self.nparams]
        params = theta_[-self.nparams:]
        
        sigma = self.model.args['sigma_weight_prior']
        
        # Gaussian prior
        log_p = 0.0
        for param in theta :
            log_p += 0.5*np.sum(param**2) / (sigma**2)
        
        #print(self.system_model.params)

        if self.prior_model == 'Beta' :
            lnprior = self.system_model.PriorBeta(params)
        elif self.prior_model == 'Logarithmic' :
            lnprior = self.system_model.PriorLogarithmic(params)
        elif self.prior_model == 'Gamma' :
            lnprior = self.system_model.PriorGamma(params)
        else :
            lnprior = self.system_model.PriorUniform(params)

        return (log_p + lnprior)/len(theta_)
        
    def support(self, theta_) :
        
        params = theta_[-self.nparams:]
        theta = theta_[:-self.nparams]

        for param, label in zip(params, self.model.labels) :
            param = float(param)
            if param < self.model.args[f'{label}_min'] or param > self.model.args[f'{label}_max'] :
                return False
        
        if (theta > 1e3).any() or (theta < -1e3).any() :
            return False

        return True
    
    def lnprob(self, theta_) :
        
        if not self.support(theta_) :
            #print('out of  support')
            return -1e9
        
        #model = copy.deepcopy(self.model)

        loss_likelihood = self.likelihood(theta_)
        loss_prior = self.prior(theta_)
        
        loss = -(loss_likelihood + loss_prior)
        if self.call_counter % 1000 == 0:
            print(f'loss: {loss:.6g}, {loss_likelihood:.6g}, {loss_prior:.6g} ')
        self.call_counter += 1
        
        #if np.isnan(loss) or np.isinf(loss) :
        #    print(f'loss: {loss:.6g}, {loss_likelihood:.6g}, {loss_prior:.6g} ')
        
        return loss
    
    
    def infer_parameters(self, num_iterations, initial_weights=None, sampler=None, **kwargs):
        """
        Inferir los parámetros de la red neuronal bayesiana utilizando MCMC.
        
        Args:
            model: Red neuronal PINN (instancia de torch.nn.Module).
            sampler: Función de sampler que genera nuevas muestras de parámetros.
            num_iterations: Número de iteraciones de MCMC.
            initial_weights: Pesos iniciales de la red. Si no se proporciona, se inicializa aleatoriamente.
            step_size: Tamaño de paso para el sampler (perturbación).
            sigma: Desviación estándar para la prior gaussiana.
        
        Returns:
            Lista con los parámetros inferidos en cada iteración.
        """
        #self.args.update(kwargs)

        self.model.args.update(kwargs)

        self.params = np.array(kwargs['params'])
        self.nparams = len(self.params)

        N = kwargs['N']
        min_val = torch.tensor(0, dtype=torch.float32)
        max_val = torch.tensor(N, dtype=torch.float32)
        self.model.normalizer = Normalizer(min_val, max_val)

        pinn_params_np = self.get_params_vector()
        self.ndim = pinn_params_np.size + self.nparams
                
        self.t = torch.tensor(kwargs['t'], dtype=torch.float32, requires_grad=True).view(-1,1) ## require_grad
        self.data = torch.tensor(kwargs['data'], dtype=torch.float32).view(-1,1)

        self.prior_model = kwargs['prior_model']

        # 1. Inicialization
        #if initial_weights is None:
        #    initial_weights = self.get_params_vector()[0]  # Usamos los parámetros iniciales del modelo
        
        if 'f_init_name' in kwargs :
            print(f'Loading pinn weights and initial parameters from file {kwargs["f_init_name"]}')

            with open(kwargs["f_init_name"], 'rb') as fin :
                data_init = pickle.load(fin)
                pinn_params = data_init['pinn_params']
                
            pinn_params_np = pinn_params[self.nparams:]
            self.params = pinn_params[:self.nparams]

        current_weights = np.concatenate([pinn_params_np, self.params]) # juntamos los pesos y sesgos, con los parámetros de la ED
        print('starting from params: ', self.params)
        self.sampler_name = sampler
        
        print(f'Running on sampler: {sampler}')
        
        if sampler=='twalk' :
            self.sampler = pytwalk(self.ndim, k=1, U=None, w=self.likelihood, Supp=self.support, u=self.prior)
            
            self.sampler.U = self.sampler.Energy   ### modification to allow serialization when saving to joblib
            xp0 = current_weights
            xp1 = current_weights + 0.0001*np.random.randn(len(current_weights))
            self.sampler.Run(num_iterations, xp0, xp1, save_xp=True)
             
            self.Output = np.stack((self.sampler.Output[:,:-1], self.sampler.Outputp[:,:-1]), axis=1)

        elif sampler=='emcee' :
            
            self.nwalkers = int(2*self.ndim)
            print(f'using {self.nwalkers} walkers')
            theta_0 =[current_weights + 0.001*np.random.randn(len(current_weights)) for _ in range(self.nwalkers)]
            
            from emcee.moves import DEMove, WalkMove, DESnookerMove, KDEMove
            moves = [(DEMove(sigma=1e-8), 0.5), (WalkMove(self.ndim), 0.3), (DESnookerMove(gammas=1.7/2), 0.2)]
            
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, moves=moves)
            with tqdm(total=num_iterations) as pbar:
                for i, _ in enumerate(self.sampler.sample(theta_0, iterations=num_iterations)):
                    pbar.update(1)
            
            self.Output = self.sampler.get_chain()

        elif sampler=='pyhmc' :
            
            self.sampler = pyhmc(self.likelihood,self.prior,self.support,ndim=self.ndim, **kwargs)
            
            with tqdm(total=num_iterations) as pbar :
                for i, _ in enumerate(self.sampler.Run(num_iterations, current_weights)) :
                    pbar.update(1)
            
            self.sampler.Output = np.expand_dims(self.sampler.Output,axis=1)
            self.Output = self.sampler.Output
                       
        else :
            print('sampler must be one of available')
            return
    
        return
    
        