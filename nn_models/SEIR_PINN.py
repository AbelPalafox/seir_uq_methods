# -*- coding: utf-8 -*-

from PINN.PINN import PINN
import torch
from numpy import exp
from epidemic_model import Epidemic_Model
from utils import Normalizer

class SEIR_PINN(PINN) :
    
    def __init__(self, n_input=1, n_output=3, n_hidden=8, n_flayers=3, **kwargs) :
        
        #torch.manual_seed(123)
        super().__init__(n_input,n_output,n_hidden,n_flayers, **kwargs)

        return         

    def run(self, t, data, x0, params_0 = [], epochs=10000, lr=1e-3, N=1000) :
        
        self.normalizer = Normalizer(0,N)
        
        t_data = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
        inc_data = torch.tensor(data, dtype=torch.float32).view(-1,1)
        
        #inc_data = self.normalizer.normalize(inc_data)
        
        x0 = self.normalizer.normalize(x0)
        
        S0, E0, I0, R0 = x0 
        
        if not len(params_0) == 3 :
            self.log_beta = torch.randn(1, requires_grad=True)  # Log(beta)
            self.log_sigma = torch.randn(1, requires_grad=True)  # Log(sigma)
            self.log_gamma = torch.randn(1, requires_grad=True)  # Log(gamma)

        else :
            #self.log_beta, self.log_sigma, self.log_gamma = [ torch.tensor(np.log(par), requires_grad=False) for par in params_0]
            #self.log_beta, self.log_sigma, self.log_gamma = [torch.log(torch.tensor(par, dtype=torch.float32, requires_grad=True)) for par in params_0]
            self.log_beta, self.log_sigma, self.log_gamma = [torch.nn.Parameter(torch.log(torch.tensor(par, dtype=torch.float32)), requires_grad=True) for par in params_0]

            #self.log_beta = torch.log(self.log_beta)
            #self.log_sigma = torch.log(self.log_sigma)
            #self.log_gamma = torch.log(self.log_gamma)
            
            
            #self.log_beta = torch.randn(1, requires_grad=True)  # Log(beta)
            #self.log_sigma = torch.randn(1, requires_grad=True)  # Log(sigma)
            #self.log_gamma = torch.randn(1, requires_grad=True)  # Log(gamma)

        
        self.log_beta, self.log_sigma, self.log_gamma = self.train(t_data, 
                                        inc_data,
                                        params=[self.log_beta, self.log_sigma, self.log_gamma],
                                        S0=S0,
                                        E0=E0,
                                        I0=I0,
                                        R0=R0,
                                        N=N,
                                        lambda_data=1e2,
                                        lambda_cond=1e3,
                                        lambda_eq=1e7,
                                        epochs=epochs,
                                        lr=lr,
                                        likelihood='Poisson',
                                        optim='Adam')
        
        return exp(self.log_beta.item()), exp(self.log_sigma.item()), exp(self.log_gamma.item())

    def compute_incidency(self, S, N) :
        
        
        #N_tensor = torch.tensor([N],dtype=torch.float32, device=S.device, requires_grad=True)
        
        N_tensor = torch.tensor([N],dtype=torch.float32).requires_grad_()
        incidency = -torch.diff(S,dim=0, prepend=N_tensor)

        #incidency = torch.clamp(incidency,min=0)
        #incidency = torch.nn.functional.softplus(incidency)
        incidency = 0.5 * (incidency + (incidency**2 + 1.0/N_tensor).sqrt())
        
        return incidency    


    def compute_eq_loss(self, y_pred, t) :
        
        beta = torch.exp(self.log_beta)
        sigma = torch.exp(self.log_sigma)
        gamma = torch.exp(self.log_gamma)
        
        S_pred, E_pred, I_pred = torch.split(y_pred, 1, dim=1)
        #R_pred = N - S_pred - E_pred - I_pred
                
        dS_dt = torch.autograd.grad(S_pred, t, torch.ones_like(S_pred), create_graph=True)[0]
        dE_dt = torch.autograd.grad(E_pred, t, torch.ones_like(E_pred), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred, t, torch.ones_like(I_pred), create_graph=True)[0]
        #dR_dt = torch.autograd.grad(R_pred, t, torch.ones_like(R_pred), create_graph=True)[0]
    
        loss_seir = torch.mean((dS_dt + beta * S_pred * I_pred) ** 2) + \
                    torch.mean((dE_dt - beta * S_pred * I_pred + sigma * E_pred) ** 2) + \
                    torch.mean((dI_dt - sigma * E_pred + gamma * I_pred) ** 2) #+ \
                    #torch.mean((dR_dt - gamma * I_pred) ** 2)
    
        #print('Gradient norm: ', torch.norm(dS_dt).item(),  torch.norm(dE_dt).item(),  torch.norm(dI_dt).item())
    
        return loss_seir, [dS_dt, dE_dt, dI_dt]

    
    def compute_data_loss(self, y_pred, data, t) :
        
        N = self.args['N']
        
        #S_pred = y_pred[:,0].requires_grad_()

        S_pred = y_pred[:, 0]
        I_pred = y_pred[:, 2]

        S_pred_denormalized = self.normalizer.denormalize(S_pred)
        I_pred_denormalized = self.normalizer.denormalize(I_pred)

        # Corrección IC trick para que S_pred solo decrezca
        #beta = torch.exp(self.log_beta)  # Aquí usas tu parámetro
        #S_pred_denormalized = N - torch.cumsum( beta * S_pred_denormalized * I_pred_denormalized, dim=0)
        
        #data_denormalized = self.normalizer.denormalize(data)#.requires_grad_()
        
        incidency = self.compute_incidency(S_pred_denormalized, N)
                
        if self.args['likelihood'] == 'Gaussian' :
            
            data_loss = torch.mean((incidency - data.squeeze()) ** 2)
            #data_loss = torch.mean((incidency - data_denormalized) ** 2)
            
        elif self.args['likelihood'] == 'Poisson' :
            epsilon = 1 ## numerical regularization to avoid log 0
            
            data_loss = torch.mean(incidency - data.squeeze()*torch.log(incidency + epsilon) )

            #if data_loss.item() < 0 :
            #    print('negative data loss')
            #    print(incidency)
            #    print(torch.log(incidency + epsilon))

            #data_loss = torch.mean(incidency - data_denormalized*torch.log(incidency + epsilon))
            
            #if np.sum(incidency) == 0:
            #    return np.inf
            
            # computing scaling constant for avoid the chain gets stucked
            #C = torch.sum(data) / np.sum(incidency)
            
            #p_i = C*incidency - data.squeeze()*np.log(C*incidency + epsilon)
            
            #if np.isnan(np.log(C*incidency + epsilon)).any() :
            #    print('***** negative incidency')
            #    print(incidency)
            
            #loss_data = torch.sum(p_i)

        return data_loss
    
    def compute_cond_loss(self, y_pred) :
    
        S0 = self.args['S0']
        E0 = self.args['E0']
        I0 = self.args['I0']
        R0 = self.args['R0']
        N = self.args['N']
        
        # Condiciones iniciales
        S_pred, E_pred, I_pred = torch.split(y_pred, 1, dim=1)
        S_init_pred = S_pred[0]
        E_init_pred = E_pred[0]
        I_init_pred = I_pred[0]
        
        R_init_pred = 1 - S_init_pred - E_init_pred - I_init_pred
    
        loss_initial_conditions = (S_init_pred - S0) ** 2 + (E_init_pred - E0) ** 2 + \
                                  (I_init_pred - I0) ** 2 + (R_init_pred - R0) ** 2
    
        return loss_initial_conditions


    def scale_gradients(self) :

        scaling_factor = 1e7

        with torch.no_grad() :
            for param in [self.log_beta, self.log_sigma, self.log_gamma] :
                if param.grad is not None :
                    param.grad *= scaling_factor
