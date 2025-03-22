
from .Epidemic_Model import Epidemic_Model
from scipy.integrate import solve_ivp
import pandas as pd
from numpy import maximum

class SEIR_Model(Epidemic_Model) :

    def __init__(self, beta, sigma, gamma, N) :
        """
        Initializes the parmeters of the SEIR model
        :param beta: Transmision rate
        :param sigma: Incubation rate (1/days spend in the exposed state)
        :param gamma: Recovery rate (1/days in infectious state)
        :param N: Population size
        """
    
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        self.labels = ['Susceptible', 'Exposed', 'Infectuous', 'Recovery']
        
    def model(self, t, x) :
        """
        Defines the differential equations system for the SEIR model.
        :param t: Time.
        :param x: State of the system (S, E, I, R).
        :return: Derivatives of populations S, E, I, R with respect to time.
        """  
        
        S, E, I, R = maximum(x, 0.0)
        dSdt = -self.beta*S*I/self.N
        dEdt = self.beta*S*I/self.N - self.sigma*E
        dIdt = self.sigma*E - self.gamma*I
        dRdt = self.gamma*I
        
        return [dSdt, dEdt, dIdt, dRdt]
        
    def run(self, x0, t_eval) :
        """
        Solves the differential equations system using the solve_ivp solver with RK45 method.
        :param x0: Initial values (S_0, E_0, I_0, R_0).
        :param t_eval: Time steps where solution is given.
        :return: Solutions (S, E, I, R) in the times t_eval.
        """
        t_span = [t_eval[0], t_eval[-1]]
        
        method = 'LSODA' if (self.beta/self.sigma > 100 or self.beta/self.gamma > 100) else 'RK45'
        
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
        data['S'] = x[0]
        data['E'] = x[1]
        data['I'] = x[2]
        data['R'] = x[3]
        data['time'] = t
        
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(fname,index=False)
        
        
if __name__ == '__main__' :

    import numpy as np

    beta = 0.36
    sigma = 0.98
    gamma = 0.20
    N = 100000
    t = np.arange(0,175,1)

    seir = SEIR_Model(beta, sigma, gamma, N)

    I0 = 5
    S0 = N-I0
    R0 = 0
    E0 = 0
    x0 = [S0, E0, I0, R0]
    
    x = seir.run(x0, t)

    seir.plot(t,x)

    seir.save('test_data.csv',t,x)
