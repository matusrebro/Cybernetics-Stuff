"""

packaged control systems module

"""


import numpy as np
from scipy.integrate import odeint
from scipy.signal import tf2ss


def fun_lin_sys_odes(x,t,u,A,B):
    if np.size(u)==1:
        return np.squeeze(np.dot(A,x)+np.squeeze(B*u))
    else:      
        return np.squeeze(np.dot(A,x)+np.dot(B,u))

def sim_lin_sys_ss(u, x0, A, B, C, D, Ts):
    x=odeint(fun_lin_sys_odes,x0,np.linspace(0,Ts),
             args=(u,A,B), rtol=1e-5
             )[-1,:]
    if np.size(u)==1:
        y=np.dot(np.transpose(C),x) + np.squeeze(D*u)
    else:
        y=np.dot(np.transpose(C),x) + np.dot(D,u)
    return x, y


def sim_lin_sys_tf(u, x0, num, den, Ts):
    A, B, C, D = tf2ss(num, den)
    x, y = sim_lin_sys_ss(u, x0, A, B, np.transpose(C), D, Ts)
    return x, y


class lin_model:
    
    A = 0
    B = 0
    C = 0
    D = 0
    
    def __init__(self, parameters):
        A, B, C, D = parameters
        self.A = A
        self.B = A
        self.C = A
        self.D = A
    
    
    x0 = 0
    x = 0
    
    # def simulation(self, u, Ts):
        
    