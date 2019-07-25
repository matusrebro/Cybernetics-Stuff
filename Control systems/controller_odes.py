"""
Here are all control system ODEs

"""

import numpy as np

def fun_pid(x,t,e,p):
    Kp, Ki, Kd, Td = p
    x1, x2 = x
    a1 = 1/Td
    
    x1_dot = x2
    x2_dot = -a1 * x2 + e
    
    return np.array([x1_dot, x2_dot])