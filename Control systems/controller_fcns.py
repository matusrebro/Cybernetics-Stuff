"""
Implementation of control laws
"""

import numpy as np
from scipy.integrate import odeint
from controller_odes import fun_pid

# continuous PID controller algorithm
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
    
# discrete time implementation of PID via Euler method
# beware: Td > Ts 
def pid_disc(e, x0, p, Ts):
    Kp, Ki, Kd, Td = p
    a1 = 1/Td
    b0 = Ki/Td
    b1 = Kp/Td + Ki
    b2 = Kd/Td + Kp
    
    x01, x02 = x0
    x1 = Ts * x02 + x01
    x2 = Ts * (-a1*x02 + e) + x02
 
    x = np.array([x1, x2])
    
    u = np.dot([b0, b1 - a1*b2],x) + b2*e
    
    return u, x

