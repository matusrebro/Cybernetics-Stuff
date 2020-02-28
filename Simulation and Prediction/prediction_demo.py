"""
Here is the demonstration of the identification and simulation of discrete 
stochastic LTI systems
"""
import numpy as np
import matplotlib.pyplot as plt

from pred_fcns import arx_par_est, arx_sim

t=np.arange(0,10,0.05)

u=2*np.sin(2*np.pi*2*t)
y=0.5*np.sin(2*np.pi*2*t-1)


ny = [2]
nu = [2]

theta = arx_par_est(ny, ny, y, u)


ysim = arx_sim(ny, nu, u, [y[0]], theta)

plt.plot(t, u, label = 'u')
plt.plot(t, y, label = 'y')
plt.plot(t, ysim, label = 'ysim')
plt.legend()
