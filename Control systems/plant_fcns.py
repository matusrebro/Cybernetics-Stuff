"""
Here are implementations of plants for testing of control algorithms

u: m
y: p

A: nxn
B: nxm
C: nxp
D: pxm


"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import tf2ss
from plant_odes import fun_lin_sys_ss

def sim_lin_sys_ss(u, x0, A, B, C, D, Ts):
    x=odeint(fun_lin_sys_ss,x0,np.linspace(0,Ts),
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