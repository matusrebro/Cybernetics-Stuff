"""

packaged control systems module

"""


import numpy as np
from scipy.integrate import odeint
from scipy.signal import tf2ss, ss2tf


def fun_lin_sys_odes(x,t,u,A,B):
    return np.squeeze(np.dot(A,x)+np.dot(B,u))

def sim_lin_sys_ss(u, x0, A, B, C, D, Ts):
    x = odeint(fun_lin_sys_odes, x0, np.linspace(0,Ts),
             args=(u,A,B), rtol=1e-5
             )[-1,:]
    
    y = np.dot(C ,x) + np.dot(D, u)
    return x, y


class lin_model:
    
    A = 0
    B = 0
    C = 0
    D = 0
    
    num = []
    den = []
    
    x0 = 0
    
    def __init__(self, parameters):
        
        if len(parameters) == 4:
            self.A, self.B, self.C, self.D = parameters
            self.num, self.den = ss2tf(self.A, self.B, self.C, self.D)
            
        elif len(parameters) ==2:
            self.num, self.den = parameters
            self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)

        else:
            raise ValueError("Invalid parameter format")

    def set_parameters(self, parameters):
    
        if len(parameters) == 4:
            self.A, self.B, self.C, self.D = parameters
            self.num, self.den = ss2tf(self.A, self.B, self.C, self.D)
            
        elif len(parameters) ==2:
            self.num, self.den = parameters
            self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)
    
        else:
            raise ValueError("Invalid parameter format")
            
            
    def simulation(self, x0, t, u):
        self.x0 = x0
        
        Ts = t[1] - t[0]
        
        x=np.zeros([len(t), len(x0)])

        if len(self.C.shape)==1:
            output_count = 1
        else:
            output_count = self.C.shape[0]

        y = np.zeros([len(t), output_count])

        if len(u.shape) == 1:
            u.shape = (u.shape[0], 1)
        
        for k in range(1, len(t)):
            self.x0, y[k] = sim_lin_sys_ss(u[k-1,:], self.x0, self.A, self.B, self.C, self.D, Ts)
            x[k, :] = self.x0
            
        return x, y
    
    
    def freq_response(self, omega):
        jomega = 1j * omega
        nyquist = np.polyval(self.num, jomega) / np.polyval(self.den, jomega)
        mag = 20 * np.log10(nyquist)
        phase = np.rad2deg(np.angle(nyquist))
        
        return mag, phase, nyquist
    
    