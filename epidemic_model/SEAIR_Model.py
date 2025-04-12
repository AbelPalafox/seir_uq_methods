
from Epidemic_Model import Epidemic_Model
from scipy.integrate import solve_ivp
import pandas as pd
from numpy import maximum

class SEAIR_Model(Epidemic_Model) :

    def __init__(self, Lambda, beta_I, beta_A, mu, sigma, p, gamma_I, gamma_A, N) :
        """
        Initializes the parmeters of the SEIR model
        :param beta: Transmision rate
        :param sigma: Incubation rate (1/days spend in the exposed state)
        :param gamma: Recovery rate (1/days in infectious state)
        :param N: Population size
        """
        self.Lambda, self.beta_I, self.beta_A, self.mu, self.sigma, self.p, self.gamma_I, self.gamma_A = Lambda, beta_I, beta_A, mu, sigma, p, gamma_I, gamma_A
        self.N = N
        self.labels = ['Susceptible', 'Exposed', 'Asymptomatic', 'Infectuous', 'Recovery']
        
    def model(self, t, x) :
        """
        Defines the differential equations system for the SEIR model.
        :param t: Time.
        :param x: State of the system (S, E, I, R).
        :return: Derivatives of populations S, E, I, R with respect to time.
        """  
        S, E, A, I, R = maximum(x, 0.0)
        dSdt = self.Lambda-self.beta_I*S*I/self.N - self.beta_A*S*A/self.N - self.mu*S
        dEdt = self.beta_I*S*I/self.N + self.beta_A*S*A/self.N - (self.sigma + self.mu)*E
        dAdt = (1-self.p)*self.sigma*E - (self.gamma_A + self.mu)*A 
        dIdt = self.p*self.sigma*E - (self.gamma_I + self.mu)*I
        dRdt = self.gamma_A*A + self.gamma_I*I - self.mu*R
        
        return [dSdt, dEdt, dAdt, dIdt, dRdt]
        
    def run(self, x0, t_eval) :
        """
        Solves the differential equations system using the solve_ivp solver with RK45 method.
        :param x0: Initial values (S_0, E_0, I_0, R_0).
        :param t_eval: Time steps where solution is given.
        :return: Solutions (S, E, I, R) in the times t_eval.
        """
        t_span = [t_eval[0], t_eval[-1]]
        
        method = 'LSODA' if (self.beta_I/self.sigma > 100 or self.beta_I/self.gamma_I > 100) else 'RK45'
        
        sol = solve_ivp(
            self.model, t_span, x0, t_eval=t_eval, method=method,
            rtol=1e-2, atol=1e-3, max_step=1.0
        )

        return sol['y']

    def plot(self, t, x) :
    
        super().plot(t,x,labels=self.labels)
        
    def save(self, fname, t, x) :
        
        print('Saving output data as a dataframe')
        data = {}
        for label, x_i in zip(self.labels, x) :
            data[label] = x_i
        data['time'] = t
        
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(fname,index=False)
        
        
if __name__ == '__main__' :

    import numpy as np
    import matplotlib.pyplot as plt

    Lambda = 1.7826e-5
    beta_I = 4.52
    beta_A = 1.9
    mu = 1.7826e-5
    sigma = 1/6.4
    p = 0.868343
    gamma_I = 0.33029
    gamma_A = 0.13978
    N = 128000000
    #t = np.arange(0,100,1)
    t = np.linspace(0,100,300)

    seir = SEAIR_Model(Lambda, beta_I, beta_A, mu, sigma, p, gamma_I, gamma_A, N)

    I0 = 4
    R0 = 0
    E0 = 4
    A0 = 1
    S0 = N - I0 - E0 - A0 - R0
    x0 = [S0, E0, A0, I0, R0]
    
    x = seir.run(x0, t)

    seir.plot(t,x)

    plt.figure()
    plt.plot(t,x[3],label='Infectious')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t,x[1], label='Exposed')
    plt.plot(t,x[2], label='Asymptomatic')
    plt.grid()
    plt.show()

    #seir.save('test_data.csv',t,x)
