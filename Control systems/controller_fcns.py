"""
Implementation of control laws
"""

import numpy as np
from scipy.integrate import odeint
from controller_odes import fun_pid

def pid_cont(e, x0, p, Ts):
    Kp, Ki, Kd, Td = p
    a1 = 1/Td
    b0 = Ki/Td
    b1 = Kp/Td + Ki
    b2 = Kd/Td + Kp
    
    x=odeint(fun_pid,x0,np.linspace(0,Ts),
                 args=(e,p), rtol=1e-5
                 )
 
    u = np.dot([b0, b1 - a1*b2],x[-1,:]) + b2*e
    
    return u, x[-1,:]
    


