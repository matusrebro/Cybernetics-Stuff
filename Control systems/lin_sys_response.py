"""
Showcase of linear systems responses
"""


import numpy as np
import matplotlib.pyplot as plt

from plant_fcns import sim_lin_sys_tf, sim_lin_sys_ss

# --- first order ss system step response

Ts=0.05
t=np.arange(0,3,Ts)

A=-5 * np.eye(1)
B=1 * np.ones(1)
C=1 * np.ones(1)
D=0 * np.ones(1)

x=np.zeros_like(t)
x0=np.zeros(1)

for k in range(1, len(t)):
    x0, y = sim_lin_sys_ss(1, x0, A, B, C, D, Ts)
    x[k] = x0

plt.figure()
plt.subplot(211)    
plt.title('first order ss system')    
plt.plot(t,x,label=r'$x(t)$')
plt.xlabel('time')
plt.ylabel(r'$x(t)$')
plt.legend()

# --- second order ss system step response

Ts=0.05
t=np.arange(0,5,Ts)

A=np.array([[0, 1], [-3, -2]])
B=np.array([0, 1])
C=np.ones(2)
D=0 * np.ones(1)
x=np.zeros([len(t),2])
x0=np.zeros(2)
for k in range(1, len(t)):
    x0, y = sim_lin_sys_ss(1, x0, A, B, C, D, Ts)
    x[k,:] = x0

plt.subplot(212)
plt.title('second order ss system')  
for k in range(x.shape[1]):        
    plt.plot(t,x[:,k],label=r'$x_'+str(k+1)+'(t)$')

plt.xlabel('time')
plt.ylabel(r'$x(t)$')
plt.legend()
plt.tight_layout()

# --- first order tf system step response

Ts=0.05
t=np.arange(0,3,Ts)

num = [3]
den = [1, 3]

y=np.zeros_like(t)
x0=np.zeros(1)
for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(1, x0, num, den, Ts)
    y[k] = yk

plt.figure()
plt.subplot(211)
plt.title('first order tf system')  
plt.plot(t,y, label=r'$y(t)$')
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend()


# --- second order tf system step response
Ts=0.05
t=np.arange(0,5,Ts)

num = [3]
den = [1, 2, 3]

y=np.zeros_like(t)
x0=np.zeros(2)
for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(1, x0, num, den, Ts)
    y[k] = yk

plt.subplot(212)
plt.title('second order tf system')  
plt.plot(t,y, label=r'$y(t)$')
plt.xlabel('time')
plt.ylabel(r'$y(t)$')
plt.legend()
plt.tight_layout()