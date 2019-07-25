"""

Here, step step responses of PID controller are plotted

all gains or controller parameters are equal to one and
derivative time constant is equal to sampling time (simulation step)

"""

import numpy as np
import matplotlib.pyplot as plt

from controller_fcns import pid_cont

Ts=0.05
t=np.arange(0,3,Ts)
e=np.ones_like(t)
u=np.zeros_like(t)


Td=Ts*1


# --- step responses ---

plt.figure()
plt.title(r'Step responses of P, PI, PD and PID controllers with unit gains (parameters) and derivative time constant $T_D=0.1$',fontsize=8)

# --- P controller ---
Kp=1
Ki=0
Kd=0
p= [Kp, Ki, Kd, Td]
xp=np.zeros(2)
for k in range(0,len(t)):
    u[k], xp = pid_cont(e[k],xp,p,Ts)
    
plt.plot(t,u,'r',label='P controller')    

# --- PI controller ---
Kp=1
Ki=1
Kd=0
p= [Kp, Ki, Kd, Td]
xp=np.zeros(2)
for k in range(0,len(t)):
    u[k], xp = pid_cont(e[k],xp,p,Ts)
    
plt.plot(t,u,'b',label='PI controller')    
    
# --- PD controller ---
Kp=1
Ki=0
Kd=1
p= [Kp, Ki, Kd, Td]
xp=np.zeros(2)
for k in range(0,len(t)):
    u[k], xp = pid_cont(e[k],xp,p,Ts)
    
plt.plot(t,u,'k',label='PD controller')  
    
# --- PID controller ---
Kp=1
Ki=1
Kd=1
p= [Kp, Ki, Kd, Td]
xp=np.zeros(2)
for k in range(0,len(t)):
    u[k], xp = pid_cont(e[k],xp,p,Ts)
    
plt.plot(t,u,'m',label='PID controller')  
plt.xlabel('time [s]')
plt.ylabel('$u(t)$ [-]')
plt.legend()
plt.tight_layout()
