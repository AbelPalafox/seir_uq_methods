#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:46:31 2025

@author: snorkk
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time 
from numpy import linalg 
from statsmodels.graphics.tsaplots import plot_acf

###############################################################################
#Generacion de datos
#4
np.random.seed(0)

pars = np.array([0.8,1/5.2,0.1])

'''
# Parámetros del modelo
N = 1e4   # Población total
beta = 0.8      # Tasa de transmisión
sigma = 1/5.2   # Tasa de incubación (1/días de incubación)
gamma = 1/10     # Tasa de recuperación (1/días infeccioso)
'''

N = 1e4   # Población total
beta, sigma, gamma = pars


# Condiciones iniciales
I0 = 1          # Infectados iniciales
E0 = 0          # Expuestos iniciales
R0 = 0          # Recuperados iniciales
S0 = N - I0 - E0 - R0  # Susceptibles iniciales

# Modelo SEIR
def seir(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

#odeint(seir, y0, t, args=(N, beta, sigma, gamma))
def Euler(fun, x0, t,N,beta,sigma,gamma):
    dt = t[1:]-t[:-1]
    #print(dt)
    n = dt.shape[0]
    sol = np.zeros([n+1,4])
    sol[0,:] = x0
    for i in range(n):
        aux = sol[i,:] + dt[i]*np.array(fun(sol[i,:],t[i],N,beta,sigma,gamma))
        aux[aux<0] = 0
        sol[i+1,:] = aux
    return sol

# Rango de tiempo (días)
t = np.linspace(0, 100, 201)

# Resolver las ecuaciones diferenciales
y0 = (S0, E0, I0, R0)
#sol = odeint(seir, y0, t, args=(N, beta, sigma, gamma))
sol = Euler(seir, y0, t,N,beta,sigma,gamma)
S, E, I, R = sol.T
'''
# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptibles")
plt.plot(t, E, label="Expuestos")
plt.plot(t, I, label="Infectados")
plt.plot(t, R, label="Recuperados")
plt.xlabel("Días")
plt.ylabel("Población")
plt.legend()
plt.title("Modelo SEIR")
plt.grid()
plt.show()
'''
Incidencia = np.zeros([60,])

for i in range(60):
    Incidencia[i] = sigma*E[2*(i+1)]


plt.plot(Incidencia,label = 'Datos generados')

Incidencia = Incidencia + np.random.normal(0,9,60)
Incidencia[Incidencia<0] = 0
plt.plot(Incidencia, label = 'Datos con ruido agregado')
plt.title("Incidencia")
plt.grid()
plt.legend()
Datos = np.copy(Incidencia)



###############################################################################

def Energia(pars, Datos):
    beta,sigma,gamma = pars
    
    t = np.linspace(0,60,121)
    sol = Euler(seir, y0, t,N,beta,sigma,gamma)
    inc = np.zeros([60,])

    for i in range(60):
        inc[i] = sigma*sol[2*(i+1),1]
    
    
    
    return np.sum((Datos-inc)**2)



pars = np.array([beta,sigma,gamma])
print(Energia(pars, Datos))




def ExactGrad(x0, Datos):
    '''
    Parameters
    ----------
    x0 : Valores de los parámetros beta, sigma y gamma.

    Returns
    -------
    sol: la solución del ODE.
    2*(Datos-Y)@dE: El gradiente de la función objetivo. Para este caso se usa 
        una verosimilitud de tipo gaussiana.
    '''
    beta,sigma,gamma = x0

    ######################################    ######################################
    #Solució del ODE
    npoints = 121

    S = np.zeros([npoints,])
    E = np.zeros([npoints,])
    I = np.zeros([npoints,])
    Yn = np.zeros([npoints,])
    #Valores inciiales de los estados
    S[0] = y0[0]
    E[0] = y0[1]
    I[0] = y0[2]
    
    t = np.linspace(0,60,npoints)
    dt = 0.5
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
    #En este ciclo se calculan los valores de las parciales para cada valor en el tiempo
    for i in range(1,npoints):
        S[i] = S[i-1] - dt*beta*S[i-1]*I[i-1]/N
        E[i] = E[i-1] + dt*(beta*S[i-1]*I[i-1]/N - sigma*E[i-1])
        I[i] = I[i-1] + dt*(sigma*E[i-1] - gamma*I[i-1])
        if S[i] < 0:
            S[i] = 0
        if E[i] < 0:
            E[i] = 0
        if I[i] < 0:
            I[i] = 0
        
        Yn[i-1] = sigma*E[i]
        
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
    
    dE = np.zeros([60,3])
    dE[:,0] = sigma*dEdb[2::2]
    dE[:,1] = sigma*dEds[2::2] + E[2::2]
    dE[:,2] = sigma*dEdg[2::2]

    
    Energy = (Datos-Yn[1::2])@(Datos-Yn[1::2])
    Grad = -2*((Datos-Yn[1::2])@dE)
    
    return Energy, Grad



#Hamiltonian MCMC
def hamiltonian_monte_carlo(n_samples, Fun, initial_position, Datos, NumberSteps = 1, step_size=0.5):
###############################################################################
#Leapfrog
    def leapfrog(q, p, U, Energia_ini, Gradiente, WeightInv, Datos, NumberSteps, step_size):
        dim = np.shape(q)[0]
        qini, pini = np.copy(q), np.copy(p)
        q, p = np.copy(q), np.copy(p)
        
        p -= step_size * Gradiente/ 2  # half step
        for _ in range(NumberSteps): 
            q += step_size * WeightInv@p  # whole step
            if np.any(q<limits[0]) or np.any(q>limits[1]):
                while np.any(q<limits[0]) or np.any(q>limits[1]):
                    q = qini + 0.01*np.random.randn(dim)
                Potential, Gradiente = Fun(q, Datos)
                
                if np.log(np.random.rand()) < Energia_ini - Potential:
                    return q, Potential, Gradiente, 1
                else:
                    return q, Potential, Gradiente, 0
            Potential, Gradiente = Fun(q, Datos)
            p -= step_size * Gradiente  # whole step
        q += step_size * WeightInv@p  # whole step    
        if np.any(q<limits[0]) or np.any(q>limits[1]):
            while np.any(q<limits[0]) or np.any(q>limits[1]):
                q = qini + 0.01*np.random.randn(dim)
            Potential, Gradiente = Fun(q, Datos)
            
            if np.log(np.random.rand()) < Energia_ini - Potential:
                return q, Potential, Gradiente, 1
            else:
                return q, Potential, Gradiente, 0

        Potential, Gradiente = Fun(q, Datos)
        p -= step_size * Gradiente / 2  # half step
        
        Start_log_p = Energia_ini + 0.5*pini @WeightInv@pini
        New_log_p = Potential + 0.5*p @ WeightInv@p
        
        if np.log(np.random.rand()) < Start_log_p - New_log_p:
            return q, Potential, Gradiente, 1
        else:
            return q, Potential, Gradiente, 0
###############################################################################
#Hessiana de una funcion arbitraria
    def Hessian(fun,x0, Datos):
        tam = np.shape(x0)[0]
        Hess = np.zeros([tam,tam])
        h = 5e-5
        Hj = np.zeros([tam])
        Hi = np.zeros([tam])
        
        for i in range(tam):    
            Hi[i] = h
            for j in range(i):
                Hj[j] = h
                Hess[i,j] = (fun(x0+Hi+Hj, Datos)+fun(x0-Hi-Hj, Datos)-fun(x0-Hi+Hj, Datos)-fun(x0+Hi-Hj, Datos))/(4*h**2)
                Hess[j,i] = Hess[i,j]
                Hj[j] = 0.0           
            Hess[i,i] = (fun(x0+Hi, Datos)-2*fun(x0, Datos)+fun(x0-Hi, Datos))/(h**2)
            Hi[i] = 0.0
        
        return Hess
###############################################################################

#Inicio del HMCMC
    dim = np.shape(initial_position)[0]
    Iter = []
    # collect all our samples in a list
    samples = [initial_position]
    Usamples = np.zeros(n_samples,)
    Usamples[0], gradiente = Fun(samples[-1], Datos)

    Opt = initial_position
    Value = Usamples[0]

    Weight = Hessian(Energia,samples[-1], Datos)
    
    AutoVal, AutoVect = linalg.eig(Weight)
    

    if np.any(AutoVal < 1e-6):
        if np.any(np.abs(AutoVal)<1e-6):
            Weight = np.diag(np.ones([dim,]))
            WeightInv = np.diag(np.ones([dim,]))
            AutoVal = np.ones([dim,])
        if np.any(AutoVal<0):
            AutoVal = abs(AutoVal)
            Weight = AutoVect @ np.diag(AutoVal) @ AutoVect.transpose()
            WeightInv = AutoVect @ np.diag(1/AutoVal) @ AutoVect.transpose()
    else:
        WeightInv = AutoVect @ np.diag(1/AutoVal) @ AutoVect.transpose()
    
    p0 = np.random.multivariate_normal( np.zeros(dim),Weight, 1)
    p0 = p0.reshape(dim,)
    
    q_new, energia, gradiente, salida = leapfrog(
        samples[-1],
        p0,
        Fun,
        Usamples[0],
        gradiente,
        WeightInv,
        Datos,
        NumberSteps=NumberSteps,
        step_size=step_size,
    )
    
    samples.append(np.copy(q_new))    
    U_new = energia

    if U_new < Value:
        Opt = q_new
        Value = U_new
 
    
    for i in range(1,n_samples):
        if i%100 == 0:
            print(i)
        Weight = Hessian(Energia,samples[-1], Datos)
        
        AutoVal, AutoVect = linalg.eig(Weight)
        
        if np.any(AutoVal < 1e-6):
            if np.any(np.abs(AutoVal)<1e-6):
                Weight = np.diag(np.ones([dim,]))
                WeightInv = np.diag(np.ones([dim,]))
                AutoVal = np.ones([dim,])
            if np.any(AutoVal<0):
                AutoVal = abs(AutoVal)
                Weight = AutoVect @ np.diag(AutoVal) @ AutoVect.transpose()
                WeightInv = AutoVect @ np.diag(1/AutoVal) @ AutoVect.transpose()
        

        p0 = np.random.multivariate_normal( np.zeros(dim),Weight, 1)
        p0 = p0.reshape(dim,)
        
        q_new, energia, gradiente, salida = leapfrog(
            samples[-1],
            p0,
            Fun,
            Usamples[i-1],
            gradiente,
            WeightInv,
            Datos,
            NumberSteps=NumberSteps,
            step_size=step_size,
        )

        #salida = 1
        U_new = energia
        
        if U_new < Value:
            Value = U_new
            Opt = q_new
        
        if salida == 1:
            samples.append(np.copy(q_new))
            Usamples[i] = U_new

        else:
            samples.append(np.copy(samples[-1]))
            Usamples[i] = Usamples[i-1]
                
    return np.array(samples[1:]), Usamples, Opt, Value



n_samples = 10000
Fun = ExactGrad
initial_position = pars + 0.0
NumberSteps = 16
step_size = 1e-2
limits = np.array([0,3])#Valores mínimo y máximo de las variables. Cada variable tiene el mismo rango.

t1 = time.time()
sample, Usample, Opt, Value = hamiltonian_monte_carlo(n_samples, Fun, initial_position, Datos, NumberSteps, step_size)
t2 = time.time()


Burnin = 0

print('Tiempo: ',t2-t1)

plt.figure('beta')
plt.title('beta')
plt.plot(sample[Burnin:,0])


plt.figure('sigma')
plt.title('sigma')
plt.plot(sample[Burnin:,1])


plt.figure('gamma')
plt.title('gamma')
plt.plot(sample[Burnin:,2])

plt.figure('Energia')
plt.title('Energia')
plt.plot(Usample[Burnin:])

print('punto final     : ',Opt)
print('posicion inicial: ', initial_position)
print('valores reales  : ', pars)


beta_est,sigma_est,gamma_est = Opt
t = np.linspace(0,60,121)

sol_est = Euler(seir, y0, t,N,beta_est,sigma_est,gamma_est)
S, E, I, R = sol.T

Incidencia_est = np.zeros([60,])

for i in range(60):
#    aux = seir(sol_est[2*(i+1)],0,N,beta_est,sigma_est,gamma_est)
    Incidencia_est[i] = sigma_est*E[2*(i+1)]

plt.figure('Estimacion')
plt.plot(Incidencia,label = 'Datos')
plt.plot(Incidencia_est, label = 'Estimado')
plt.legend()



Burnin = 1000
plot_acf(sample[Burnin:,0], lags=20)
plot_acf(sample[Burnin:,1], lags=20)
plot_acf(sample[Burnin:,2], lags=20)