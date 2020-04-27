"""
Various dynamic models and simulators

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint


def fcn_Lorenz(x, t, par):
    x1, x2, x3 = x
    sigma, rho, beta = par
    
    x1_dot = sigma * (x2 - x1)
    x2_dot = x1 * (rho - x3) - x2
    x3_dot = x1 * x2 - beta * x3
    return np.array([x1_dot, x2_dot, x3_dot])


class Lorenz_system:
    
    parameters = []
    
    def init_model(self, parameters):
        self.parameters = parameters
        
    def __init__(self, parameters = [10, 28, 8/3]):
        self.init_model(parameters)
        
    def simulation(self, t = np.arange(0, 40, 0.01), x0 = [1, 1, 1], plot = True):
        
        Ts = t[1]-t[0] 
        idx_final = int(t[-1]/Ts)+1    
        x = np.zeros([idx_final, len(x0)])
        x[0,:] = x0   
        
        for i in range(1,idx_final):
            y = odeint(fcn_Lorenz, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                     args=(self.parameters,)
                     )
            x[i,:] = y[-1,:]
            
        
        if plot:
            fig = plt.figure(1)
            ax = fig.gca(projection='3d')
            ax.plot(x[:,0], x[:,1], x[:,2])
            plt.show()
            plt.figure(2)
            plt.subplot(311)
            plt.plot(t,x[:,0])
            plt.subplot(312)
            plt.plot(t,x[:,1])
            plt.subplot(313)
            plt.plot(t,x[:,2])
            
        return x

class Logistic_map:
    
    r = 1
    
    def init_model(self, r):
        self.parameters = r
        
    def __init__(self, r = 1):
        self.init_model(r)
        
    def simulation(self, x0 = 0.5, N = 300, plot = True):
        
        x = np.zeros(N)
        x[0] = x0
        
        for k in range(1, N):
            x[k] = self.r * x[k-1] * (1 - x[k])
        
        if plot:
            plt.plot(x)
            
        return x
    
    def bifuraction(self):
        N = 300
        
        for r in np.arange(0, 4, 0.01):
            x = np.zeros(N)
            x[0] = 0.1
        
            for k in range(1, N):
                x[k] = r * x[k-1] * (1 - x[k-1])
        
            plt.plot(r * np.ones_like(x[100:]), x[100:],'k.', markersize=1)
