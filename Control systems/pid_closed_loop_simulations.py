"""
Closed loop simulations - PID controller
"""

import numpy as np
import matplotlib.pyplot as plt

from plant_fcns import sim_lin_sys_tf
from controller_fcns import pid_cont

Ts=0.05
t=np.arange(0,5,Ts)
r=np.ones_like(t)
u=np.zeros_like(t)


# --- first order linear plant

num = [3]
den = [1, 3]

# --- P controller ---
Td=Ts*1
Kp=5
Ki=0
Kd=0
p= [Kp, Ki, Kd, Td]

y=np.zeros_like(t)
x0=np.zeros(1)
xp=np.zeros(2)

# first action of controller after first output measurment
# which is in this case initial condition of the system
# is given by:

u[0], xp = pid_cont(r[0]-y[0],xp,p,Ts)

# and so the closed loop system response begins:

for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(u[k-1], x0, num, den, Ts)
    e=r[k-1]-y[k-1]
    u[k], xp = pid_cont(e,xp,p,Ts)
    y[k] = yk

plt.figure()
plt.title('First order system - P controller')
plt.subplot(211)
plt.xlabel('time')
plt.ylabel('y, r')
plt.plot(t,y,label=r'$y(t)$')
plt.plot(t,r,label=r'$r(t)$')
plt.legend()
plt.subplot(212)
plt.xlabel('time')
plt.ylabel('u')
plt.plot(t,u,label=r'$u(t)$')
plt.legend()

# --- PI controller ---
Td=Ts*1
Kp=2
Ki=6
Kd=0
p= [Kp, Ki, Kd, Td]

y=np.zeros_like(t)
x0=np.zeros(1)
xp=np.zeros(2)

u[0], xp = pid_cont(r[0]-y[0],xp,p,Ts)

for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(u[k-1], x0, num, den, Ts)
    e=r[k-1]-y[k-1]
    u[k], xp = pid_cont(e,xp,p,Ts)
    y[k] = yk

plt.figure()
plt.title('First order system - PI controller')
plt.subplot(211)
plt.xlabel('time')
plt.ylabel('y, r')
plt.plot(t,y,label=r'$y(t)$')
plt.plot(t,r,label=r'$r(t)$')
plt.legend()
plt.subplot(212)
plt.xlabel('time')
plt.ylabel('u')
plt.plot(t,u,label=r'$u(t)$')
plt.legend()



# --- second order linear plant

num = [2]
den = [1, 3, 2]


# --- PI controller ---
Td=Ts*1
Kp=4
Ki=2
Kd=0
p= [Kp, Ki, Kd, Td]

y=np.zeros_like(t)
x0=np.zeros(2)
xp=np.zeros(2)

u[0], xp = pid_cont(r[0]-y[0],xp,p,Ts)

for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(u[k-1], x0, num, den, Ts)
    e=r[k-1]-y[k-1]
    u[k], xp = pid_cont(e,xp,p,Ts)
    y[k] = yk

plt.figure()
plt.title('Second order system - PI controller')
plt.subplot(211)
plt.xlabel('time')
plt.ylabel('y, r')
plt.plot(t,y,label=r'$y(t)$')
plt.plot(t,r,label=r'$r(t)$')
plt.legend()
plt.subplot(212)
plt.xlabel('time')
plt.ylabel('u')
plt.plot(t,u,label=r'$u(t)$')
plt.legend()


# --- PID controller ---
Td=Ts*1
Kp=3
Ki=2.5
Kd=0.8
p= [Kp, Ki, Kd, Td]

y=np.zeros_like(t)
x0=np.zeros(2)
xp=np.zeros(2)

u[0], xp = pid_cont(r[0]-y[0],xp,p,Ts)

for k in range(1, len(t)):
    x0, yk = sim_lin_sys_tf(u[k-1], x0, num, den, Ts)
    e=r[k-1]-y[k-1]
    u[k], xp = pid_cont(e,xp,p,Ts)
    y[k] = yk

plt.figure()
plt.title('Second order system - PID controller')
plt.subplot(211)
plt.xlabel('time')
plt.ylabel('y, r')
plt.plot(t,y,label=r'$y(t)$')
plt.plot(t,r,label=r'$r(t)$')
plt.legend()
plt.subplot(212)
plt.xlabel('time')
plt.ylabel('u')
plt.plot(t,u,label=r'$u(t)$')
plt.legend()
