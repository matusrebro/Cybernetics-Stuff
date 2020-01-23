"""
Here is the demonstration of the identification and simulation of discrete 
stochastic LTI systems
"""
import numpy as np
import matplotlib.pyplot as plt

from pred_fcns import arx_pred, arx_par_est, arx_sim

t=np.arange(0,10,0.05)

u=2*np.sin(2*np.pi*2*t)
y=0.5*np.sin(2*np.pi*2*t-1)

na=2
nb=2

theta=arx_par_est(na, nb, y, u)

theta_AR=theta[0:2]
theta_X=theta[2:]


ysim=arx_sim(u, y[0], theta_AR, theta_X)

plt.plot(t,u)
plt.plot(t,y)
plt.plot(t,ysim)
