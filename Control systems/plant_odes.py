"""
Here are all ODEs of plants for control algorithms testing
"""

import numpy as np

def fun_lin_sys_ss(x,t,u,A,B):
    if np.size(u)==1:
        return np.squeeze(np.dot(A,x)+np.squeeze(B*u))
    else:      
        return np.squeeze(np.dot(A,x)+np.dot(B,u))


