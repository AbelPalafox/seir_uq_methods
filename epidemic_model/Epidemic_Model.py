
from numpy import roll, maximum, cumsum, diff
import matplotlib.pyplot as plt

class Epidemic_Model :

    def __init__(self) :
    
        self.labels = []
        
    def incidency(self,P,t,timestep,method='susceptible', **kwargs) :
        """
        Computes the incidence for the infectous population, given at times t, after timestep time.
        Times when infectous are reported could be not evenly spaced (regular).
        :param P: Population at times t.
        :param t: times when infectous are given.
        :param timestep: time interval to report the incidence.
        :param method: method to compute the incidence. Options are 'susceptible' and 'infectous'.
        :return: Incidency 
        """
        
        if method == 'susceptible' :
        
            delta_t = t[1] - t[0]
            if timestep < delta_t :
                print(f"Warning: timestep is lower than the reporting times. timestep will be considered as {delta_t}")
                timestep = delta_t
                
            if timestep % delta_t != 0 :
                print("Warning: timestep is not multiple of the reporting time.")
                
            step = int(timestep // delta_t)
            P_timestep = roll(P,-step)
            inc = (P_timestep - P)[::step]
        
            return inc[:-1]
    
        elif method == 'exposed' :
        
            delta_t = t[1] - t[0]
            if timestep < delta_t :
                print(f"Warning: timestep is lower than the reporting times. timestep will be considered as {delta_t}")
                timestep = delta_t
                
            if timestep % delta_t != 0 :
                print("Warning: timestep is not multiple of the reporting time.")
                
            step = int(timestep // delta_t)
            E, I = P
            sigma = kwargs['sigma']
            gamma = kwargs['gamma']

            inc = sigma*E - gamma*I

            return inc
        
        elif method == 'roman' :

            sigma = kwargs['sigma']
            Y = cumsum(sigma*P)
            inc = diff(Y)
            #print(P[:5], inc[:5], sigma*P[0])
            return inc
        
        else :
            print("Warning: method not implemented.")
            return None
        
        
    def plot(self, t, x, **kwargs) :
        """
        Creates a plot of the populations at times t
        :param t: Times:
        :param x: array of populations (population, times).
        :param kwargs: parameters for matplotlib plotter 
        """
        labels = kwargs['labels']
        
        plt.figure()
        for x_i, label_i in zip(x,labels) :
            plt.plot(t,x_i,label=label_i)
        plt.grid()
        plt.legend()
        plt.show()
            
        return
            
            

        
        
        
