
from numpy import roll
import matplotlib.pyplot as plt

class Epidemic_Model :

    def __init__(self) :
    
        self.labels = []
        
    def incidency(self,I,t,timestep,regular=True) :
        """
        Computes the incidence for the infectous population, given at times t, after timestep time.
        Times when infectous are reported could be not evenly spaced (regular).
        :param I: Infectous at times t.
        :param t: times when infectous are given.
        :param timestep: time interval to report the incidence.
        :param regular: flag to indicate whether reporting times are evenly spaced.
        :return: Incidency 
        """
        
        if regular :
        
            delta_t = t[1] - t[0]
            if timestep < delta_t :
                print(f"Warning: timestep is lower than the reporting times. timestep will be considered as {delta_t}")
                timestep = delta_t
                
            if timestep % delta_t != 0 :
                print("Warning: timestep is not multiple of the reporting time.")
                
            step = int(timestep // delta_t)
            I_timestep = roll(I,-step)
            inc = (I_timestep - I)[::step]
        
        return inc[:-1]
        '''
        else :
        
            ### in progress.....
            t0, t1 = t[0], t[-1]
            curr_time = t[0]
            curr_index = 0
            inc = []
            I0 = I[0]
            
            while curr_time < t1 :
            
                i = curr_index
                t_i = 0
                while t_i < timestep :
                    t_i
        '''    
        
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
            
            

        
        
        
