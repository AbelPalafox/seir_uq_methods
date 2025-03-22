#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 08:34:42 2025

@author: abel
"""
import torch
import torch.nn as nn
from types import MethodType
from numpy import pi, exp

class PINN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_FLAYERS, compute_eq_loss=None, compute_data_loss=None, compute_cond_loss=None, RFF=False):
        super().__init__()
        
        if RFF==True:
            print('Using Random Fourier Features')
            omega_scale = 1.0
            nfft = 10
            self.omega = omega_scale * torch.randn(nfft)  # 16 Fourier features
            self.b = 2 * pi * torch.rand(nfft)  # Phase shifts aleatorios  
            N_INPUT = 2*nfft
        
        
        self.RFF = RFF
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), nn.Tanh())
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh()) for _ in range(N_FLAYERS - 1)
        ])
        self.fce = nn.Sequential(nn.Linear(N_HIDDEN, N_OUTPUT), nn.ReLU())
        
        self.args = {}
        '''if not compute_eq_loss == None :
            self.compute_eq_loss = MethodType(compute_eq_loss, self)
        if not compute_data_loss == None :
            self.compute_data_loss = MethodType(compute_data_loss, self)
        if not compute_cond_loss == None :
            self.compute_cond_loss = MethodType(compute_cond_loss, self)'''

        self.apply(self.init_weights)
        

    def fourier_features(self, t):
        t = t.view(-1, 1)
        phi = torch.cat([torch.sin(self.omega * t + self.b), torch.cos(self.omega * t + self.b)], dim=1)
        return phi


    def init_weights(self, m) :
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier uniforme
            if m.bias is not None:
                nn.init.uniform_(m.bias,a=0.01, b=0.2)  # Sesgos inicializados en 0  

    def forward(self, x):
        if self.RFF :
            x = self.fourier_features(x)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x  

    def train(self, t, data, lr=1e-3, epochs=100000, params=[], status=1000, **kwargs) :
        
        self.args = kwargs
        lambda_data = self.args['lambda_data']
        lambda_cond = self.args['lambda_cond']
        lambda_eq = self.args['lambda_eq']
        optim = self.args['optim']        
        #tensor_params = [torch.tensor(i, requires_grad=True) for i in params]
        
        self.data_losses = []
        self.cond_losses = []
        self.eq_losses = []
        self.params_track = []

        if optim == 'Adam' :
            #optimizer = torch.optim.Adam(list(self.parameters()) + params, lr=lr)
            #optimizer = torch.optim.Adam(self.parameters(), lr=lr)  # Tasa de aprendizaje para los pesos
            optimizer = torch.optim.Adam([
                    {'params': self.fcs.parameters(), 'lr': lr},  # Pesos de la red
                    {'params': self.fch.parameters(), 'lr': lr},
                    {'params': self.fce.parameters(), 'lr': lr},
                    {'params': params, 'lr': lr*1}  # Parámetros del modelo SEIR
                ])
            

            for i in range(epochs) :
                
                '''if i < 20000 :
                    lambda_eq = 0
                    lambda_data = 1e6
                elif i < 30000 :
                    lambda_eq = 1e-2
                    lambda_data = 1e6
                else :
                    lambda_eq = 1e0
                    lambda_data = 1e6'''
                
                optimizer.zero_grad()
                
                y_pred = self.forward(t)
                
                eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                
                #y_pred = self.forward(t)
                data_loss = self.compute_data_loss(y_pred, data, t)
                
                cond_loss = self.compute_cond_loss(y_pred)
                
                lambda_eq_annealing = lambda_eq * (1 + i / epochs)
                #lambda_data_annealing = lambda_data * torch.exp(torch.tensor(i / epochs))
                
                loss = lambda_eq_annealing*eq_loss + lambda_data*data_loss + lambda_cond*cond_loss

                self.data_losses.append(data_loss.item())
                self.cond_losses.append(cond_loss.item())
                self.eq_losses.append(eq_loss.item())
                self.params_track.append([_.item() for _ in params])
            
                #if i % status == 0:
                #    eq_loss.backward(retain_graph=True)
                #    print("Gradientes:")
                #    for name, param in self.named_parameters():
                #        if param.grad is not None:
                #            print(name, param.grad.norm().item())
                #        else:
                #            print(name, "Gradiente NULO")
                    
                loss.backward()

                self.scale_gradients()

                optimizer.step()
                 
                if i % status == 0:
                    #print(f"Iteración {i}: Pérdida = {loss.item():.6f}, {[_.item() for _ in tensor_params]}")
                    print(f"Iteración {i}: Pérdida = {loss.item():.3g}, eq: {eq_loss.item():.3g}, data: {data_loss.item():.3g}, cond: {cond_loss.item():.3g}, {[torch.exp(_).item() for _ in params]}")
                    
                    
                #if i % status == 0 :
                #    print('noising the parameters')
                #    for param in params :
                #        param.data += 1e-1 * torch.randn_like(param) * exp(-i / epochs)

        elif optim == 'LBFGS' :
            # using the LBFGS optimizer
            #optimizer = torch.optim.LBFGS(list(self.parameters()) + params, lr=lr, max_iter=20, history_size=50)
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, line_search_fn='strong_wolfe')
            
            def closure():
                optimizer.zero_grad()
                
                y_pred = self.forward(t)
                
                #print('1 : ',t.requires_grad, t.grad_fn)
                eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                #print('2 : ',t.requires_grad, t.grad_fn)
                data_loss = self.compute_data_loss(y_pred, data, t)
                #print('3 : ',t.requires_grad, t.grad_fn)
                cond_loss = self.compute_cond_loss(y_pred)
                
                loss = lambda_eq*eq_loss + lambda_data*data_loss + lambda_cond*cond_loss
            
                self.data_losses.append(data_loss.item())
                self.cond_losses.append(cond_loss.item())
                self.eq_losses.append(eq_loss.item())
                self.params_track.append([_.item() for _ in params])

                loss.backward()
                return loss
            
            for i in range(epochs) :
                
                optimizer.step(closure)
                
                if i % status == 0 :
                    
                    y_pred = self.forward(t)
                    #print('4 : ', t.requires_grad, t.grad_fn)
                    eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                    #print('5 : ', t.requires_grad, t.grad_fn)
                    data_loss = self.compute_data_loss(y_pred, data, t)
                    #print('6 : ', t.requires_grad, t.grad_fn)
                    cond_loss = self.compute_cond_loss(y_pred)
                    
                    lambda_data_annealing = lambda_data * (1 + i / epochs)
                    
                    loss = lambda_eq*eq_loss + lambda_data*data_loss + lambda_cond*cond_loss

                    if i % status == 0:
                        eq_loss.backward(retain_graph=True)
                        print("Gradientes:")
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                print(name, param.grad.norm().item())
                            else:
                                print(name, "Gradiente NULO")

                    print(f"Iteración {i}: Pérdida = {loss.item():.3g}, eq: {eq_loss.item():.3g}, data: {data_loss.item():.3g}, cond: {cond_loss.item():.3g}, {[torch.exp(_).item() for _ in params]}")

    
        else :
            
            total_epochs = epochs 
            epochs = epochs//2
            
            optimizer = torch.optim.Adam(list(self.parameters()) + params, lr=lr)
            print('Starting Adam phase')
            for i in range(epochs) :
                
                optimizer.zero_grad()
                
                y_pred = self.forward(t)
                
                eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                
                #y_pred = self.forward(t)
                data_loss = self.compute_data_loss(y_pred, data, t)
                
                cond_loss = self.compute_cond_loss(y_pred)
                
                lambda_data_annealing = lambda_data * (1 + i / epochs)
                #lambda_data_annealing = lambda_data * torch.exp(torch.tensor(i / epochs))
                
                loss = lambda_eq*eq_loss + lambda_data_annealing*data_loss + lambda_cond*cond_loss
            
                loss.backward()
                optimizer.step()
                 
                if i % status == 0:
                    #print(f"Iteración {i}: Pérdida = {loss.item():.6f}, {[_.item() for _ in tensor_params]}")
                    print(f"Iteración {i}: Pérdida = {loss.item():.3g}, eq: {eq_loss.item():.3g}, data: {data_loss.item():.3g}, cond: {cond_loss.item():.3g}, {[torch.exp(_).item() for _ in params]}")
                    
                if i % status == 0 :
                    print('noising the parameters')
                    for param in params :
                        param.data += 1e-1 * torch.randn_like(param) * exp(-i / epochs)            
 
            # using the LBFGS optimizer
            optimizer = torch.optim.LBFGS(list(self.parameters()) + params, lr=lr, max_iter=20, history_size=50)
            
            def closure():
                optimizer.zero_grad()
                
                y_pred = self.forward(t)
                
                #print('1 : ',t.requires_grad, t.grad_fn)
                eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                #print('2 : ',t.requires_grad, t.grad_fn)
                data_loss = self.compute_data_loss(y_pred, data, t)
                #print('3 : ',t.requires_grad, t.grad_fn)
                cond_loss = self.compute_cond_loss(y_pred)
                
                lambda_data_annealing = lambda_data * (1 + i / epochs)
                
                loss = lambda_eq*eq_loss + lambda_data_annealing*data_loss + lambda_cond*cond_loss
            
                loss.backward()
                return loss
            
            epochs = total_epochs - epochs
            print('Starting LBFGS phase')
            for i in range(epochs) :
                
                optimizer.step(closure)
                
                if i % status == 0 :
                    
                    y_pred = self.forward(t)
                    #print('4 : ', t.requires_grad, t.grad_fn)
                    eq_loss, dsystem_dt = self.compute_eq_loss(y_pred, t)
                    #print('5 : ', t.requires_grad, t.grad_fn)
                    data_loss = self.compute_data_loss(y_pred, data, t)
                    #print('6 : ', t.requires_grad, t.grad_fn)
                    cond_loss = self.compute_cond_loss(y_pred)
                    
                    loss = lambda_eq*eq_loss + lambda_data_annealing*data_loss + lambda_cond*cond_loss
            
                    print(f"Iteración {i}: Pérdida = {loss.item():.3g}, eq: {eq_loss.item():.3g}, data: {data_loss.item():.3g}, cond: {cond_loss.item():.3g}, {[torch.exp(_).item() for _ in params]}")
           
 
        return params #[i.item() for i in params]


if __name__ == '__main__' :


    def compute_eq_loss(self, y_pred, t, tensor_params) :
    
        N = self.args['N']
        
        beta, sigma, gamma = tensor_params
        
        t_physics = torch.linspace(0, 14, 42).view(-1, 1).requires_grad_(True)
        
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
    
    def compute_data_loss(self, y_pred, data) :
        
        #S_pred, E_pred, I_pred = torch.chunk(y_pred, 3, dim=1)
        I_pred = y_pred[:,2]
        
        loss_data = torch.mean((I_pred - data.squeeze()) ** 2)
        
        return loss_data
    
    def compute_cond_loss(self, y_pred) :
    
        S0 = self.args['S0']
        E0 = self.args['E0']
        I0 = self.args['I0']
        R0 = self.args['R0']
        N = self.args['N']
        
        # Condiciones iniciales
        #S_init_pred, E_init_pred, I_init_pred = y_data[0,:]
        S_pred, E_pred, I_pred = torch.chunk(y_pred, 3, dim=1)
        S_init_pred = S_pred[0]
        E_init_pred = E_pred[0]
        I_init_pred = I_pred[0]
        
        R_init_pred = N - S_init_pred - E_init_pred - I_init_pred
    
        loss_initial_conditions = (S_init_pred - S0) ** 2 + (E_init_pred - E0) ** 2 + \
                                  (I_init_pred - I0) ** 2 + (R_init_pred - R0) ** 2
    
        return loss_initial_conditions


#if __name__ == '__main__' :

    
    torch.manual_seed(123)
    pinn = PINN(1,3,32,3, 
                compute_eq_loss,
                compute_data_loss,
                compute_cond_loss)
    
    beta = 0.001
    gamma = 0.1
    sigma = 0.1

    t_data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=torch.float32).view(-1, 1)
    I_data = torch.tensor([1, 3, 6, 25, 73, 222, 294, 258, 237, 191, 125, 69, 27, 11, 4], dtype=torch.float32).view(-1, 1)

    # Condiciones iniciales
    S0, E0, I0, R0 = 762, 0, 1, 0
    N = S0 + E0 + I0 + R0  # Población total
    
    beta, sigma, gamma = pinn.train(t_data, 
                                    I_data,
                                    params=[beta, gamma, sigma],
                                    S0=S0,
                                    E0=E0,
                                    I0=I0,
                                    R0=R0,
                                    N=N,
                                    lambda_data=1e-2,
                                    lambda_cond=1e-4,
                                    lambda_eq=1)



'''
# Inicialización de la PINN
torch.manual_seed(123)
pinn = FCN(1, 3, 32, 3)

# Parámetros a optimizar
beta = torch.tensor(0.001, requires_grad=True)
gamma = torch.tensor(0.1, requires_grad=True)
sigma = torch.tensor(0.1, requires_grad=True)

# Optimizador
optimiser = torch.optim.Adam(list(pinn.parameters()) + [beta, gamma, sigma], lr=1e-3)

# Entrenamiento
for i in range(150001):
    optimiser.zero_grad()

    t_physics = torch.linspace(0, 14, 42).view(-1, 1).requires_grad_(True)
    S_pred, E_pred, I_pred = torch.chunk(pinn(t_physics), 3, dim=1)
    R_pred = N - S_pred - E_pred - I_pred

    dS_dt = torch.autograd.grad(S_pred, t_physics, torch.ones_like(S_pred), create_graph=True)[0]
    dE_dt = torch.autograd.grad(E_pred, t_physics, torch.ones_like(E_pred), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_pred, t_physics, torch.ones_like(I_pred), create_graph=True)[0]
    dR_dt = torch.autograd.grad(R_pred, t_physics, torch.ones_like(R_pred), create_graph=True)[0]

    loss_seir = torch.mean((dS_dt + beta * S_pred * I_pred) ** 2) + \
                torch.mean((dE_dt - beta * S_pred * I_pred + sigma * E_pred) ** 2) + \
                torch.mean((dI_dt - sigma * E_pred + gamma * I_pred) ** 2) + \
                torch.mean((dR_dt - gamma * I_pred) ** 2)

    I_exp = pinn(t_data)[:, 2]  # Predicciones para infectados
    loss_data = torch.mean((I_exp - I_data.squeeze()) ** 2)

    # Condiciones iniciales
    S_init_pred, E_init_pred, I_init_pred = pinn(torch.tensor([[0.0]]))[0]
    R_init_pred = N - S_init_pred - E_init_pred - I_init_pred

    loss_initial_conditions = (S_init_pred - S0) ** 2 + (E_init_pred - E0) ** 2 + \
                              (I_init_pred - I0) ** 2 + (R_init_pred - R0) ** 2

    loss = loss_seir + 1e-2 * loss_data + 1e-4 * loss_initial_conditions

    loss.backward()
    optimiser.step()

    if i % 5000 == 0:
        print(f"Iteración {i}: Pérdida = {loss.item():.6f}, beta = {beta.item():.6f}, gamma = {gamma.item():.6f}, sigma = {sigma.item():.6f}")
'''