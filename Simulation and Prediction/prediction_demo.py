"""
Here is the demonstration of the identification and simulation of discrete 
stochastic LTI systems
"""
import numpy as np
import matplotlib.pyplot as plt
from pred_fcns import arx_model

t=np.arange(0,10,0.05)

u=2*np.sin(2*np.pi*2*t)
y1=0.5*np.sin(2*np.pi*2*t-1)
y2=0.1*np.sin(2*np.pi*2*t-1)



ny = [2]
nu = [2]


model1 = arx_model(ny, nu)
model1.parameter_estimation(y1, u)

model1.outputCount

ysim = model1.simulation(u, y1[0])

plt.plot(t, u, label = 'u')
plt.plot(t, y1, label = 'y')
plt.plot(t, ysim, label = 'ysim')
plt.legend()



ny = [2, 2]
nu = [2]

model2 = arx_model(ny, nu)

model2.outputCount

model2.h

model2.theta

y = np.transpose(np.vstack((y1,y2)))
model2.parameter_estimation(y, u)

ysim = model2.simulation(u, y[0,:])


plt.plot(t, u, label = 'u')
plt.plot(t, y, label = 'y')
plt.plot(t, ysim, label = 'ysim')
plt.legend()



