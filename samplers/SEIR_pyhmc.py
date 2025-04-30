#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:45:10 2025

@author: abel
"""

from tqdm import tqdm
from .SEIR_mcmc_base import SEIR_mcmc_base
from .pyhmc import pyhmc
from epidemic_model.SEIR_Model import SEIR_Model
import numpy as np
from scipy.special import gammaln 
from scipy.special import digamma
from scipy.stats import nbinom
from scipy.integrate import solve_ivp
import warnings

def seir(y, t, N, beta, sigma, gamma) :
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def Euler(fun, x0, t, N, beta, sigma, gamma):
    dt = t[1:]-t[:-1]  #160
    n = dt.shape[0]  
    dt /= n
    sol = np.zeros([n+1,4]) # 161
    sol[0,:] = x0
    for i in range(n):  #160
        aux = sol[i,:] + dt[i]*np.array(fun(sol[i,:],t[i],N,beta,sigma,gamma))
        aux[aux<0] = 0
        sol[i+1,:] = aux
    return sol #161

class SEIR_pyhmc(pyhmc, SEIR_mcmc_base) :
    
    def __init__(self, *argv, **kwargs) :
        
        SEIR_mcmc_base.__init__(self, *argv, **kwargs)
        
        if self.prior_model == 'Beta' :
            self.PriorEnergy = self.PriorBeta
        else :
            self.PriorEnergy = self.PriorUniform

        if self.likelihood_model == 'Poisson' :    
            super().__init__(loglikelihood=self.LikelihoodEnergyPoisson, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        elif self.likelihood_model == 'NegBinomial':
            super().__init__(loglikelihood=self.LikelihoodEnergyNegBinom, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        else :
            super().__init__(loglikelihood=self.LikelihoodEnergyGaussian, logprior=self.PriorEnergy, support=self.Supp, **kwargs)
        
        self.instance = 'pyhmc'
        self.energy_hmc = None
    
    def grad_prior(self, theta) :
        log_prior = 0.0
        grad = np.zeros(len(theta))

        for i, param, p in zip(range(len(theta)),self.labels, theta) :

            shape = self.params[f'shape_{param}_prior']
            scale = self.params[f'scale_{param}_prior']

            grad[i] = (shape-1.0)/p - 1.0/scale

        return grad

    def gradient_v2(self, theta) :

        N = self.params['N']
        t = self.time
        data = self.data
        r_disp = self.params['p_negbinom']
        beta, sigma, gamma = theta
        x0 = self.get_initial_conditions(theta)
        S0, E0, I0, R0 = x0

        y0 = np.array([S0, E0, I0, R0])
        sens0 = np.zeros((4, 3))
        y_aug0 = np.concatenate([y0, sens0.flatten()])


        def seir_with_sens(t, y_aug):
            y = y_aug[:4]
            S, E, I, R = y
            sens = y_aug[4:].reshape((4, 3))
            dS = -beta * S * I / N
            dE = beta * S * I / N - sigma * E
            dI = sigma * E - gamma * I
            dR = gamma * I
            dy = np.array([dS, dE, dI, dR])
            J = np.array([
                [-beta * I / N, 0, -beta * S / N, 0],
                [ beta * I / N, -sigma, beta * S / N, 0],
                [0, sigma, -gamma, 0],
                [0, 0, gamma, 0]
            ])
            df_dtheta = np.array([
                [-S * I / N, 0, 0],
                [ S * I / N, -E, 0],
                [0, E, -I],
                [0, 0, I]
            ])
            dsens_dt = J @ sens + df_dtheta
            return np.concatenate([dy, dsens_dt.flatten()])

        method = 'LSODA' if (beta/sigma > 100 or beta/gamma > 100) else 'RK45'

        sol = solve_ivp(seir_with_sens, [t[0], t[-1]], y_aug0,
                    t_eval=t, method=method, rtol=1e-2, atol=1e-3, max_step=1.0)
        
        E = sol.y[1]
        dE_dtheta = sol.y[4:].reshape(4, 3, -1)[1]

        # Incidencia diaria
        Y_model = sigma * np.diff(E, prepend=E[0])

        # Derivadas de Y_model respecto a theta
        dY_dtheta_all = []
        for j in range(3):
            dY = sigma * np.diff(dE_dtheta[j], prepend=dE_dtheta[j, 0])
            if j == 1:
                dY += np.diff(E, prepend=E[0])
            dY_dtheta_all.append(dY)
        dY_dtheta_all = np.array(dY_dtheta_all)

        # Log-verosimilitud con Binomial Negativa
        mu = Y_model
        p = r_disp / (r_disp + mu)
        loglik = np.sum(nbinom.logpmf(data, r_disp, p))

        # Gradiente de log-verosimilitud
        dlogL_dmu = (data - mu) / (mu + r_disp)
        grad_loglik = np.sum(dlogL_dmu * dY_dtheta_all, axis=1)

        grad_prior = self.grad_prior(theta)

        return -(grad_loglik + 0.*grad_prior)


    def gradient(self, theta) :

        beta, sigma, gamma = theta
        N = self.params['N']
        times = self.time
        data = self.data[1:]
        r_disp = self.params['p_negbinom']
        n = len(times)
        dt = self.params['dt']

        def generate_t_eval(t_max, sigma, gamma, resolution=10):
            dt = min(1 / (resolution * sigma), 1 / (resolution * gamma))
            return dt#np.arange(0, t_max + dt, dt)

        #dt = generate_t_eval(times[-1], sigma, gamma, resolution=self.params['dt'])
        #print('using dt: ',dt)

        n = int((times[-1] - times[0])/dt)
        if n <= 0 :
            print(n, times[-1] - times[0], dt, theta)
        # Estado original
        S = np.zeros(n)
        E = np.zeros(n)
        I = np.zeros(n)
        R = np.zeros(n)

        # Sensibilidades
        dE_dbeta = np.zeros(n)
        dE_dsigma = np.zeros(n)
        dE_dgamma = np.zeros(n)

        # Inicialización
        x0 = self.get_initial_conditions(theta)
        S0, E0, I0, R0 = x0
        S[0], E[0], I[0], R[0] = S0, E0, I0, R0
        s_beta = e_beta = i_beta = r_beta = 0.0
        s_sigma = e_sigma = i_sigma = r_sigma = 0.0
        s_gamma = e_gamma = i_gamma = r_gamma = 0.0

        for t in range(1, n):
            # Estado actual
            s, e, i, r = S[t-1], E[t-1], I[t-1], R[t-1]

            # Derivadas
            dS = -beta * s * i / N
            dE = beta * s * i / N - sigma * e
            dI = sigma * e - gamma * i
            dR = gamma * i

            # Estado siguiente
            S[t] = s + dt * dS
            E[t] = e + dt * dE
            I[t] = i + dt * dI
            R[t] = r + dt * dR

            # --- Derivadas respecto a beta
            ds_beta = -i / N * s - beta * i / N * s_beta - beta * s / N * i_beta
            de_beta = i / N * s + beta * i / N * s_beta + beta * s / N * i_beta - sigma * e_beta
            di_beta = sigma * e_beta - gamma * i_beta
            dr_beta = gamma * i_beta

            s_beta += dt * ds_beta
            e_beta += dt * de_beta
            i_beta += dt * di_beta
            r_beta += dt * dr_beta
            dE_dbeta[t] = e_beta

            # --- Derivadas respecto a sigma
            ds_sigma = 0.0
            de_sigma = -e - sigma * e_sigma
            di_sigma = e + sigma * e_sigma - gamma * i_sigma
            dr_sigma = gamma * i_sigma

            s_sigma += dt * ds_sigma
            e_sigma += dt * de_sigma
            i_sigma += dt * di_sigma
            r_sigma += dt * dr_sigma
            dE_dsigma[t] = e_sigma

            # --- Derivadas respecto a gamma
            ds_gamma = 0.0
            de_gamma = -sigma * e_gamma
            di_gamma = sigma * e_gamma - i - gamma * i_gamma
            dr_gamma = i + gamma * i_gamma

            s_gamma += dt * ds_gamma
            e_gamma += dt * de_gamma
            i_gamma += dt * di_gamma
            r_gamma += dt * dr_gamma
            dE_dgamma[t] = e_gamma

        if not (np.isfinite(S).any() or np.isfinite(E).any() or np.isfinite(I).any() or np.isfinite(R).any()) :
            print(theta)

        dE_dtheta = np.array([dE_dbeta, dE_dsigma, dE_dgamma])

        Y_model = sigma * np.diff(E, prepend=E[0])

        # Derivadas de Y_model respecto a theta
        dY_dtheta_all = []
        for j in range(3):
            dY = sigma * np.diff(dE_dtheta[j], prepend=dE_dtheta[j, 0])
            if j == 1:
                dY += np.diff(E, prepend=E[0])
            dY_dtheta_all.append(dY)
        dY_dtheta_all = np.array(dY_dtheta_all)

        # Log-verosimilitud con Binomial Negativa
        mu = Y_model

        def estimate_dispersion(data, mu_est):
            residuals = data - mu_est
            var_empirical = np.var(residuals, ddof=1)
            mu_mean = np.mean(mu_est)
            
            # Si la varianza empírica es menor a la media, forzamos r_disp grande (poco ruido)
            if var_empirical <= mu_mean:
                return 1e6  # aproximación a Poisson
            
            # Ecuación: Var = mu + mu^2 / r_disp  =>  r_disp = mu^2 / (Var - mu)
            r_disp = mu_mean**2 / (var_empirical - mu_mean)
            
            return r_disp

        r_disp = estimate_dispersion(self.data[1:], mu)

        p = r_disp / (r_disp + mu)
        
        loglik = np.sum(nbinom.logpmf(data, r_disp, p))

        # Gradiente de log-verosimilitud
        dlogL_dmu = -(data - mu) / (mu + r_disp)
        grad_loglik = np.sum(dlogL_dmu * dY_dtheta_all, axis=1)

        #grad_logl = np.array([np.mean(dE_dbeta), np.mean(dE_dsigma), np.mean(dE_dgamma)])
        grad_prior = self.grad_prior(theta)
        #print(grad_loglik, grad_prior)
        return 2*(grad_loglik + grad_prior)/N


    def gradient_sergio(self, theta) :

        beta,sigma,gamma = theta

        ######################################    ######################################
        #Solució del ODE

        t = self.time
        #dt = self.params['dt']
        N = self.params['N']
        p = self.params['p_negbinom']
        factor = self.params['factor']
        npoints = int(len(t)*factor) #int(t.size/dt)
        dt = (t[-1] - t[0])/npoints
        #S = np.zeros([npoints,])
        #E = np.zeros([npoints,])
        #I = np.zeros([npoints,])
        #aux = np.zeros([npoints,])
        Yn = np.zeros([npoints,])
        #Valores inciales de los estados
        #x0 = self.get_initial_conditions(theta)

        #S[0] = x0[0]
        #E[0] = x0[1]
        #I[0] = x0[2]
        #aux[0] = self.data[0] #Incidencias parciales, calcula la integral en los tiempos
    #0, 0.5, 1.0, ...
        x0 = self.get_initial_conditions(theta)

        t_high = np.linspace(t[0], t[-1], npoints, endpoint=True)
        sol = Euler(seir, x0, t_high, N, beta, sigma, gamma)

        S, E, I, R = sol.T

        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a beta
        dSdb = np.zeros([npoints,])
        dEdb = np.zeros([npoints,])
        dIdb = np.zeros([npoints,])

        dSpdb = np.zeros([npoints,])
        dEpdb = np.zeros([npoints,])
        dIpdb = np.zeros([npoints,])
        
        #Valores iniciales 
        dSpdb[0] = -S[0]*I[0]/N
        dEpdb[0] = -dSpdb[0]

        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a sigma
        dSds = np.zeros([npoints,])
        dEds = np.zeros([npoints,])
        dIds = np.zeros([npoints,])

        dSpds = np.zeros([npoints,])
        dEpds = np.zeros([npoints,])
        dIpds = np.zeros([npoints,])

        #Valores iniciales 
        dSds[0] = self.data[0]/(sigma**2)
        dEds[0] = -dSds[0]
        dSpds[0] = 0
        dEpds[0] = -E[0]
        dIpds[0] = -dEpds[0]
        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a gamma
        dSdg = np.zeros([npoints,])
        dEdg = np.zeros([npoints,])
        dIdg = np.zeros([npoints,])

        dSpdg = np.zeros([npoints,])
        dEpdg = np.zeros([npoints,])
        dIpdg = np.zeros([npoints,])

        #Valores iniciales 
        dSpdg[0] = 0
        dEpdg[0] = 0
        dIpdg[0] = -I[0]
        ######################################    ######################################
        #Parciales de los estados y sus derivadas en el tiempo con respecto a I0
        dSdI0 = np.zeros([npoints,])
        dEdI0 = np.zeros([npoints,])
        dIdI0 = np.zeros([npoints,])

        dSpdI0 = np.zeros([npoints,])
        dEpdI0 = np.zeros([npoints,])
        dIpdI0 = np.zeros([npoints,])

        #Valores iniciales
        dSdI0[0] = -1
        dIdI0[0] = 1
        dSpdI0[0] = -beta*S[0]/N
        dEpdI0[0] = -dSpdI0[0]
        dIpdI0[0] = -gamma
        ######################################    ######################################
        #En este ciclo se calculan los valores de las parciales para cada valor en el tiempo
        for i in range(1,npoints):
                    
            dSdb[i] = dSdb[i-1] + dt*dSpdb[i-1]
            dEdb[i] = dEdb[i-1] + dt*dEpdb[i-1]
            dIdb[i] = dIdb[i-1] + dt*dIpdb[i-1]
            dSpdb[i] = -(S[i]*I[i] + S[i]*beta*dIdb[i] + I[i]*beta*dSdb[i])/N
            dEpdb[i] = -dSpdb[i] - sigma*dEdb[i]
            dIpdb[i] = -(dSpdb[i]+dEpdb[i]) -gamma*dIdb[i]


            dSds[i] = dSds[i-1] + dt*dSpds[i-1]
            dEds[i] = dEds[i-1] + dt*dEpds[i-1]
            dIds[i] = dIds[i-1] + dt*dIpds[i-1]
            dSpds[i] = -beta*(I[i]*dSds[i] + S[i]*dIds[i])/N
            dEpds[i] = -dSpds[i] - E[i] -sigma*dEds[i]
            dIpds[i] = -(dSpds[i]+dEpds[i]) -gamma*dIds[i]


            dSdg[i] = dSdg[i-1] + dt*dSpdg[i-1]
            dEdg[i] = dEdg[i-1] + dt*dEpdg[i-1]
            dIdg[i] = dIdg[i-1] + dt*dIpdg[i-1]
            dSpdg[i] = -beta*(I[i]*dSdg[i] + S[i]*dIdg[i])/N
            dEpdg[i] = -dSpdg[i] - sigma*dEdg[i]
            dIpdg[i] = -(dSpdg[i]+dEpdg[i]) - gamma*dIdg[i] -I[i]
            
            
            dSdI0[i] = dSdI0[i-1] + dt*dSpdI0[i-1]
            dEdI0[i] = dEdI0[i-1] + dt*dEpdI0[i-1]
            dIdI0[i] = dIdI0[i-1] + dt*dIpdI0[i-1]
            dSpdI0[i] = -beta*(I[i]*dSdI0[i] + S[i]*dIdI0[i])/N
            dEpdI0[i] = -dSpdI0[i] - sigma*dEdI0[i]
            dIpdI0[i] = -(dSpdI0[i]+dEpdI0[i]) - gamma*dIdI0[i]
        
        
        #Gradiente del estado E
        dE = np.zeros([npoints,4])
        dE[:,0] = dEdb
        dE[:,1] = -dEds
        dE[:,2] = dEdg
        dE[:,3] = dEdI0
        
        #Calculo del gradiente de la incidencia
        #dY = np.zeros([npoints,4])
        daux = np.zeros([npoints,4])
        daux[1:,:] = 0.5*dt*sigma*( dE[:-1,:]+dE[1:,:] )
        daux[1:,1] += 0.5*dt*(E[:-1]+E[1:]) #Termino que se agrega por el producto con sigma
        #hat_daux = np.cumsum(np.concatenate([np.array([0,0,0,0]).reshape(1,4),daux], axis=0), axis=0)
        
        #dY[1:,:] = np.diff(hat_daux[1:,:], axis=0)
        dY = daux[1::factor] + daux[2::factor]
        #dY = dY[::factor]

        Y = np.zeros_like(t_high)
        Y[0] = self.data[0]
        aux = 0.5*dt*sigma*(E[1:]+E[:-1])
        #aux_ = np.cumsum(np.hstack([0,aux]))

        #Y[1:] = np.diff(aux_)
        #Y  = Y[::factor]
        Y = aux[0::factor] + aux[1::factor]

        Y = np.round(Y) + 1e-8

        logVer = gammaln(Y+self.data) - gammaln(Y) + self.data*np.log(p) - gammaln(self.data+1) + Y - np.log(1-p)

        GradlogVer = -(digamma(Y+self.data)-digamma(Y) + 1)@dY

        #print(GradlogVer[:-1], dY)

        return -GradlogVer[:-1]*1e-9 #np.sum(logVer)


    def hessian(self, theta) :
        eps = self.params['h']
        n = len(theta)
        H = np.zeros((n, n))
        #return np.eye(n)
    
        for j in range(n):
            theta_eps_plus = theta.copy()
            theta_eps_minus = theta.copy()
            theta_eps_plus[j] += eps
            theta_eps_minus[j] -= eps

            grad_plus = self.gradient(theta_eps_plus)
            grad_minus = self.gradient(theta_eps_minus)

            # Segunda derivada aproximada (columna j)
            H[:, j] = (grad_plus - grad_minus) / (2 * eps)

        for i in range(n):
            for j in range(i+1,n) :
                H[i,j] = H[j,i] 

        #l = np.min(np.diag(H))
        return H #@H.T+l*np.eye(n)


    def U_(self, theta) :
        
        beta, sigma, gamma = theta
        t = self.time

        dt = self.params['dt']
        N = self.params['N']
        p = self.params['p_negbinom']

        x0 = self.get_initial_conditions(theta)

        t_high = np.linspace(t[0], t[-1], len(t)*self.params['factor'], endpoint=True)
        sol = Euler(seir, x0, t_high, N, beta, sigma, gamma)

        S, E, I, R = sol[::self.params['factor']].T

        #sol = Euler(seir, x0, t, N, beta, sigma, gamma)

        #S, E, I, R = sol.T  # 161

        Y = np.zeros_like(t) # 161
        Y[0] = self.data[0]
        hat_E = np.hstack([self.data[0],E]) # 162
        aux = 0.5*dt*sigma*(E[1:]+E[:-1]) # 160
        aux_ = np.cumsum(np.hstack([0,aux]))

        Y[1:] = np.diff(aux_)
        #Y[1:] = aux[:-1] + aux[1::] # 159
        
        Y = np.round(Y)
        Y = np.clip(Y,a_min=1e-8,a_max=None)

        logVer = gammaln(Y+self.data) - gammaln(Y) + self.data*np.log(p) - gammaln(self.data+1) + Y - np.log(1-p)
        #print(-1e-4*np.sum(logVer) , self.logprior(theta))
        #return 1e4*(-np.sum(logVer) + 5e2*self.logprior(theta))/(N)
        return -np.sum(logVer) + 0*self.logprior(theta)
    
    def run(self, T, theta_0) :
        
        with tqdm(total=T) as pbar :
            for i, _ in enumerate(self.Run(T, theta_0)) :
                pbar.update(1)
        
        self.nsamples = T
        # put the output in a dataframe
        self.Output = np.column_stack([self.Output[1:],self.U_samples])

        self.Outputp = self.Output 
        
        self.create_dictionary()
        
        return True
    
    def project_params(self, theta) :

        for i, label in enumerate(self.labels) :
            theta[i] = np.clip(theta[i],self.params[f'{label}_min']+1e-6, self.params[f'{label}_max']) 

        return theta