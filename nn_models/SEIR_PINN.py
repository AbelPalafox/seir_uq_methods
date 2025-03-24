# -*- coding: utf-8 -*-

from PINN.PINN import PINN
import torch
from numpy import exp
from epidemic_model import Epidemic_Model
from utils import Normalizer

class SEIR_PINN(PINN) :
    
    def __init__(self, n_input=1, n_output=3, n_hidden=16, n_flayers=3, **kwargs) :
        
        #torch.manual_seed(123)
        super().__init__(n_input,n_output,n_hidden,n_flayers, **kwargs)
        self.labels = ['beta', 'sigma', 'gamma']
        return         

    def run(self, t, data, x0, params_0 = [], epochs=10000, lr=1e-3, **kwargs) :
        
        N = kwargs['N']
        min_val = torch.tensor(0, dtype=torch.float32)
        max_val = torch.tensor(N, dtype=torch.float32)
        self.normalizer = Normalizer(min_val,max_val)
        
        t_data = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1,1)
        inc_data = torch.tensor(data, dtype=torch.float32).view(-1,1)
        
        #inc_data = self.normalizer.normalize(inc_data)
        
        #x0 = self.normalizer.normalize(x0)
        
        S0, E0, I0, R0 = x0 
        
        if not len(params_0) == 3 :
            self.log_beta = torch.nn.Parameter(torch.rand(1), requires_grad=True)  # Log(beta)
            self.log_sigma = torch.nn.Parameter(torch.rand(1), requires_grad=True)  # Log(sigma)
            self.log_gamma = torch.nn.Parameter(torch.rand(1), requires_grad=True)  # Log(gamma)

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
                                        lambda_data=1e0,
                                        lambda_cond=1e-5,
                                        lambda_eq=1e-5,
                                        epochs=epochs,
                                        lr=lr,
                                        optim='Adam', **kwargs)
        
        return exp(self.log_beta.item()), exp(self.log_sigma.item()), exp(self.log_gamma.item())

    def compute_incidency(self, P, N, method='susceptible') :
        
        
        if method == 'susceptible' :           
            #N_tensor = torch.tensor([N],dtype=torch.float32).requires_grad_()
            #incidency = -1*torch.diff(P,dim=0, prepend=N_tensor)
            #print(incidency[:50],P[:50])

            N_tensor = torch.tensor([N], dtype=torch.float32, device=P.device)  # Si es constante
            N_tensor = N_tensor.expand_as(P[:1])  # Asegura compatibilidad de dimensiones
            incidency = -1 * (P - torch.cat([N_tensor, P[:-1]]))  # Diferencia manual

            #incidency = torch.clamp(incidency,min=0)
            incidency = torch.nn.functional.softplus(incidency)
            #incidency = 0.5 * (incidency + (incidency**2 + 1.0/N_tensor).sqrt())

            return incidency

        elif method == 'exposed' :
            
            # incidency will be computed as the differential equation for the 
            # infected population dI/dt = sigma*E - gamma*I
            #N_tensor = torch.tensor([N],dtype=torch.float32)
            E, I = P
            sigma = torch.exp(self.log_sigma)
            gamma = torch.exp(self.log_gamma)
            incidency = sigma*E - gamma*I
            incidency = torch.nn.functional.softplus(incidency)
            #incidency = 0.5 * (incidency + (incidency**2 + 1.0/N_tensor).sqrt())
            #print(sigma, gamma, E, I)
            #incidency = torch.clamp(incidency,min=0)

            return incidency


        else :

            print('Incidency method has not been implemented yet')
            return None

        return None


    def compute_eq_loss(self, y_pred, t) :
        
        beta = torch.exp(self.log_beta)
        sigma = torch.exp(self.log_sigma)
        gamma = torch.exp(self.log_gamma)
        N = torch.tensor(self.args['N'], dtype=torch.float32)

        #S_pred, E_pred, I_pred = torch.split(y_pred, 1, dim=1)
        S_pred = y_pred[:, 0]
        E_pred = y_pred[:, 1]
        I_pred = y_pred[:, 2]

        S_pred_denormalized = self.normalizer.denormalize(S_pred)
        E_pred_denormalized = self.normalizer.denormalize(E_pred)
        I_pred_denormalized = self.normalizer.denormalize(I_pred)

        R_pred = N - S_pred - E_pred - I_pred
                
        dS_dt = torch.autograd.grad(S_pred, t, torch.ones_like(S_pred), create_graph=True)[0]
        dE_dt = torch.autograd.grad(E_pred, t, torch.ones_like(E_pred), create_graph=True)[0]
        dI_dt = torch.autograd.grad(I_pred, t, torch.ones_like(I_pred), create_graph=True)[0]
        dR_dt = torch.autograd.grad(R_pred, t, torch.ones_like(R_pred), create_graph=True)[0]

        dS_dt_denormalizated = self.normalizer.denormalize(dS_dt)
        dE_dt_denormalizated = self.normalizer.denormalize(dE_dt)
        dI_dt_denormalizated = self.normalizer.denormalize(dI_dt)
        dR_dt_denormalizated = self.normalizer.denormalize(dR_dt)

        loss_seir = torch.mean((dS_dt_denormalizated + beta * S_pred_denormalized * I_pred_denormalized/N) ** 2) + \
                    torch.mean((dE_dt_denormalizated - beta * S_pred_denormalized * I_pred_denormalized/N + sigma * E_pred_denormalized) ** 2) + \
                    torch.mean((dI_dt_denormalizated - sigma * E_pred_denormalized + gamma * I_pred_denormalized) ** 2) + \
                    torch.mean((dR_dt_denormalizated - gamma * I_pred_denormalized) ** 2)
    
        return loss_seir/(N**2), [dS_dt, dE_dt, dI_dt]

    def support(self, incidency) :
        if (incidency < 0).any() :
            return False
        
        beta = torch.exp(self.log_beta)
        sigma = torch.exp(self.log_sigma)
        gamma = torch.exp(self.log_gamma)
        if beta < self.args['beta_min'] or beta > self.args['beta_max'] :
            return False
        if sigma < self.args['sigma_min'] or sigma > self.args['sigma_max'] :
            return False
        if gamma < self.args['gamma_min'] or gamma > self.args['gamma_max'] :
            return False
        
        return True
    
    def compute_data_loss(self, y_pred, data, t) :
        
        N = self.args['N']
        
        #S_pred = y_pred[:,0].requires_grad_()

        S_pred = y_pred[:, 0]
        E_pred = y_pred[:, 1]
        I_pred = y_pred[:, 2]

        S_pred_denormalized = self.normalizer.denormalize(S_pred)
        I_pred_denormalized = self.normalizer.denormalize(I_pred)
        E_pred_denormalized = self.normalizer.denormalize(E_pred)

        # Corrección IC trick para que S_pred solo decrezca
        #beta = torch.exp(self.log_beta)  # Aquí usas tu parámetro
        #S_pred_denormalized = N - torch.cumsum( beta * S_pred_denormalized * I_pred_denormalized, dim=0)
        
        #data_denormalized = self.normalizer.denormalize(data)#.requires_grad_()
        if self.args['inc_method'] == 'exposed' : 
            incidency = self.compute_incidency([E_pred_denormalized, I_pred_denormalized], N, method='exposed')
        else :
            incidency = self.compute_incidency(S_pred_denormalized, N)

        if not self.support(incidency) :
            return torch.tensor(1e6)

        if self.args['likelihood'] == 'Gaussian' :
            
            data_loss = torch.mean((incidency - data.squeeze()) ** 2)
            #data_loss = torch.mean((incidency - data_denormalized) ** 2)
            
        elif self.args['likelihood'] == 'Poisson' :
            epsilon = 1e-8 ## numerical regularization to avoid log 0
            #print(incidency[:5], incidency.shape, data.shape)
            loss_fn = torch.nn.PoissonNLLLoss(log_input=False, reduction='mean', full=True, eps=epsilon)
            data_loss = loss_fn(incidency, data.squeeze())
            #data_loss = torch.mean(torch.abs(incidency - data.squeeze()*torch.log(incidency + epsilon)) )

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
        else :
            print('Likelihood model not defined in configuration file')
            return None


        # if not isinstance(data_loss/N, torch.Tensor) :
        #     print('Warning. Something is goind wrong. data_loss is not a tensor')
        #     print(data_loss/N)
        #     return torch.tensor(data_loss/N)
        
        return data_loss/N
    
    def compute_cond_loss(self, y_pred, data) :
    
        N = torch.tensor(self.args['N'], dtype=torch.float32)
        I0 = torch.tensor(self.args['I0'],dtype=torch.float32)

        if self.args['init_cond'] == 'fixed' :
            S0 = torch.tensor(self.args['S0'], dtype=torch.float32, requires_grad=False)
            E0 = torch.tensor(self.args['E0'], dtype=torch.float32, requires_grad=False)
            I0 = torch.tensor(self.args['I0'], dtype=torch.float32, requires_grad=False)
            R0 = torch.tensor(self.args['R0'], dtype=torch.float32, requires_grad=False)
        elif self.args['init_cond'] == 'estimated':

            with torch.no_grad() :
                sigma = torch.exp(self.log_sigma)
                gamma = torch.exp(self.log_gamma)

                hat_I0 = I0  
                k = ((1.0+gamma)*data[1] - data[0])/sigma
                hat_E1 = (sigma*k + gamma*hat_I0 + data[0])/sigma
                hat_E0 = hat_E1 - k
                hat_R0 = gamma*data[1]

                hat_S0 = N - hat_E0 - hat_I0 - hat_R0

                S0 = hat_S0
                E0 = hat_E0
                I0 = hat_I0
                R0 = hat_R0
        else :
            print('Initial conditions method has not been implemented')
            print('init_cond should be fixed or estimated in the configuration file')
            return None
               
        # Condiciones iniciales
        #S_pred, E_pred, I_pred = torch.split(y_pred, 1, dim=1)
        S_pred = y_pred[:, 0]
        E_pred = y_pred[:, 1]
        I_pred = y_pred[:, 2]

        S_pred_denormalized = self.normalizer.denormalize(S_pred)
        I_pred_denormalized = self.normalizer.denormalize(I_pred)
        E_pred_denormalized = self.normalizer.denormalize(E_pred)

        S_init_pred = S_pred_denormalized[0]
        E_init_pred = E_pred_denormalized[0]
        I_init_pred = I_pred_denormalized[0]
        
        R_init_pred = N - S_init_pred - E_init_pred - I_init_pred
    
        #print(S0, E0, I0, R0 )

        loss_initial_conditions = (S_init_pred - S0) ** 2 + (E_init_pred - E0) ** 2 + \
                                  (I_init_pred - I0) ** 2 + (R_init_pred - R0) ** 2
       
        return loss_initial_conditions/(N**2)


    def scale_gradients(self) :

        scaling_factor = 1e2

        with torch.no_grad() :
            for param in [self.log_beta, self.log_sigma, self.log_gamma] :
                if param.grad is not None :
                    param.grad *= scaling_factor


    def clamp_params(self) :

        #print(self.log_beta, self.log_sigma, self.log_gamma)
        with torch.no_grad() :
            for log_param, label in zip([self.log_beta, self.log_sigma, self.log_gamma], self.labels) :
                param = torch.exp(log_param)
                param = torch.clamp(param,self.args[f'{label}_min'],self.args[f'{label}_max'])
                log_param.copy_(torch.log(param))

        #print(self.log_beta, self.log_sigma, self.log_gamma)



