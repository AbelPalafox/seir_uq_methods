# -*- coding: utf-8 -*-

from PINN.PINN import PINN
import torch
import numpy as np
from epidemic_model import Epidemic_Model

class SEIR_PINN(PINN) :
    
    def __init__(self, n_input=1, n_output=3, n_hidden=32, n_flayers=3, **kwargs) :
        
        #torch.manual_seed(123)
        super().__init__(n_input,n_output,n_hidden,n_flayers)
        
        return         

    def run(self, t, data, x0, params_0 = [], epochs=10000, lr=1e-3) :
        
        t_data = torch.tensor(t, dtype=torch.float32).view(-1,1)
        I_data = torch.tensor(data, dtype=torch.float32).view(-1,1)
        
        S0, E0, I0, R0 = x0 #762, 0, 1, 0
        N = S0 + E0 + I0 + R0  # Población total


        if not len(params_0) == 3 :
            self.beta_log = np.random.uniform()
            self.gamma_log = np.random.uniform()
            self.sigma_log = np.random.uniform()
        else :
            self.beta_log, self.sigma_log, self.gamma_log = [ np.log(par) for par in params_0]
        
        self.beta_log, self.sigma_log, self.gamma_log = self.train(t_data, 
                                        I_data,
                                        params=[self.beta_log, self.gamma_log, self.sigma_log],
                                        S0=S0,
                                        E0=E0,
                                        I0=I0,
                                        R0=R0,
                                        N=N,
                                        lambda_data=1e0,
                                        lambda_cond=1e-2,
                                        lambda_eq=1e-2,
                                        epochs=epochs,
                                        lr=lr)
        
        return torch.exp(self.beta_log), torch.exp(self.sigma_log), torch.exp(self.gamma_log)


    def compute_eq_loss(self, y_pred, t, tensor_params) :
    
        N = self.args['N']
        
        beta_log, sigma_log, gamma_log = tensor_params
        
        beta = torch.exp(beta_log)
        sigma = torch.exp(sigma_log)
        gamma = torch.exp(gamma_log)
        
        t_physics = torch.linspace(t[0].item(), t[-1].item(), 3*len(t)).view(-1, 1).requires_grad_(True)
        S_pred, E_pred, I_pred = torch.chunk(self.forward(t_physics), 3, dim=1)
        R_pred = N - S_pred - E_pred - I_pred
        
        dS_dt = torch.autograd.grad(S_pred, t_physics, torch.ones_like(S_pred), create_graph=True)[0]
        dE_dt = torch.autograd.grad(E_pred, t_physics, torch.ones_like(E_pred), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred, t_physics, torch.ones_like(I_pred), create_graph=True)[0]
        dR_dt = torch.autograd.grad(R_pred, t_physics, torch.ones_like(R_pred), create_graph=True)[0]
    
        loss_seir = torch.mean((dS_dt + beta * S_pred * I_pred) ** 2) + \
                    torch.mean((dE_dt - beta * S_pred * I_pred + sigma * E_pred) ** 2) + \
                    torch.mean((dI_dt - sigma * E_pred + gamma * I_pred) ** 2) + \
                    torch.mean((dR_dt - gamma * I_pred) ** 2)
                    
        return loss_seir, [dS_dt, dE_dt, dI_dt, dR_dt]

    
    def compute_data_loss(self, y_pred, data, t) :
        
        N = self.args['N']
        #S_pred, E_pred, I_pred = torch.chunk(y_pred, 3, dim=1)
        S_pred = y_pred[:,0]
        
        base_model = Epidemic_Model()
        
        incidency = -base_model.incidency(np.hstack([[N],S_pred]),np.asarray(t).squeeze(),1)
        
        if (incidency < 0).any() :
            incidency = np.maximum(incidency, 0)
            print('Warning: negative incidency')
        
        if self.args['likelihood'] == 'Gaussian' :
            loss_data = torch.mean((incidency - data.squeeze()) ** 2)
        elif self.args['likelihood'] == 'Poisson' :
            epsilon = 1e-8 ## numerical regularization to avoid log 0
                
            # computing scaling constant for avoid the chain gets stucked
            C = torch.sum(data) / np.sum(incidency)
            
            p_i = C*incidency - data.squeeze()*np.log(C*incidency + epsilon)
            
            loss_data = torch.sum(p_i)
            
        return loss_data
    
    def compute_cond_loss(self, y_pred) :
    
        S0 = self.args['S0']
        E0 = self.args['E0']
        I0 = self.args['I0']
        R0 = self.args['R0']
        N = self.args['N']
        
        # Condiciones iniciales
        #S_init_pred, E_init_pred, I_init_pred = y_data[0,:]
        #S_pred, E_pred, I_pred = torch.chunk(y_pred, 3, dim=1)
        #S_init_pred = S_pred[0]
        #E_init_pred = E_pred[0]
        #I_init_pred = I_pred[0]
        
        S_init_pred, E_init_pred, I_init_pred = self.forward(torch.tensor([[0.0]]))[0]
        
        R_init_pred = N - S_init_pred - E_init_pred - I_init_pred
    
        loss_initial_conditions = (S_init_pred - S0) ** 2 + (E_init_pred - E0) ** 2 + \
                                  (I_init_pred - I0) ** 2 + (R_init_pred - R0) ** 2
    
        return loss_initial_conditions

