"""

In this script, simple demo of classic responses of
one-compartmental fractional order system is showed.
System is in form: Y(s)=   b / (s^alpha+b)  U(s)
where 0 < alpha < 1

"""
# initial value problem (IVP), impulse and step response

from scipy.special import binom
from scipy.special import gamma
from Mittag_Leffler import mlf
import matplotlib.pyplot as plt
import numpy as np       

# these function are needed for Grunwald-Letnikov
# fractional order integration approximation
def omega(j,alpha):
    return (-1)**j*binom(alpha,j)

def omegasum(alpha,y,k):
    suma=0
    for i in range(0,k+1):
        suma+=omega(i,alpha)*y[k-i]
    return suma

# R^2 metric for quantification of numeric solution accuracy
def r_squared(y,y_hat):
    y=np.squeeze(y)
    y_hat=np.squeeze(y_hat)
    return 1-np.sum((y-y_hat)**2)/np.sum((y-np.mean(y))**2)

# --- IVP   
# analytic solution (through Laplace transform)
def ivp_ana(t,x0,alpha,a):
    idx_final=len(t)
    y=np.zeros(idx_final)
    for k in range(0,idx_final):
        y[k]=x0*mlf(alpha,1,-a*t[k]**alpha)    
    return y

# numeric solution (Grunwald-Letnikov)
def ivp_num(t,x0,alpha,a):
    idx_final=len(t)
    Ts=t[1]-t[0]
    y=np.zeros(idx_final)
    y[0]=x0
    for k in range(1,idx_final):
        y[k]=(Ts**alpha)*(-a*y[k-1])-omegasum(alpha,y,k) + x0*(Ts**alpha)*t[k]**(-alpha)/gamma(1-alpha)
    return y




alpha=0.5 # order of system
x0=1      # initial condition
a=1       # this gives us the pole
# no need for b parameter in IVP.

Ts=0.05
Tsim=5
t=np.arange(0,Tsim+Ts,Ts)

plt.figure()
plt.subplot(311)
y1=ivp_ana(t,x0,alpha,a)
y2=ivp_num(t,x0,alpha,a)
r2=r_squared(y1,y2)
plt.title(r'initial value problem($\alpha=0.5$), $R^2=$'+str(np.round(r2,3)))
plt.plot(t,y1,label='Analytic solution')
plt.plot(t,y2,label='Numeric solution')
plt.legend()
plt.grid()


# ---- impulse response
# we assume that initial condition is zero
# analytic solution
def imp_ana(t,alpha,a,b):
    idx_final=len(t)
    y=np.zeros(idx_final)
    for k in range(0,idx_final):
        if t[k]==0:
            y[k]=np.inf
        else:
            y[k]=b*t[k]**(alpha-1)*mlf(alpha,alpha,-a*t[k]**alpha)
    return y

# numeric solution (Grunwald-Letnikov)
def imp_num(t,alpha,a,b):
    idx_final=len(t)
    Ts=t[1]-t[0]
    y=np.zeros(idx_final)
    y[0]=b*Ts**alpha*1/Ts
    for k in range(1,idx_final):
        y[k]=(Ts**alpha)*(-a*y[k-1])-omegasum(alpha,y,k) #+ x0*(Ts**alpha)*t[k]**(-alpha)/gamma(1-alpha)
    return y

b=1
plt.subplot(312)
y1=imp_ana(t,alpha,a,b)
y2=imp_num(t,alpha,a,b)
r2=r_squared(y1[1:],y2[1:])
plt.title(r'impulse response($\alpha=0.5$), $R^2=$'+str(np.round(r2,3)))
plt.plot(t,y1,label='Analytic solution')
plt.plot(t,y2,label='Numeric solution')
plt.legend()
plt.grid()

# ---- step response
# we assume that initial condition is zero
# analytic solution
def step_ana(t,alpha,a,b):
    idx_final=len(t)
    y=np.zeros(idx_final)
    for k in range(0,idx_final):
        y[k]=b/a*(1-mlf(alpha,1,-a*t[k]**alpha))
    return y

# numeric solution (Grunwald-Letnikov)
def step_num(t,alpha,a,b):
    idx_final=len(t)
    Ts=t[1]-t[0]
    y=np.zeros(idx_final)
    u=np.ones_like(t)
    for k in range(1,idx_final):
        y[k]=(Ts**alpha)*(-a*y[k-1]+b*u[k])-omegasum(alpha,y,k)
    return y


plt.subplot(313)
y1=step_ana(t,alpha,a,b)
y2=step_num(t,alpha,a,b)
r2=r_squared(y1,y2)
plt.title(r'step response($\alpha=0.5$), $R^2=$'+str(np.round(r2,3)))
plt.plot(t,y1,label='Analytic solution')
plt.plot(t,y2,label='Numeric solution')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('figs/FODE_demo_1_output.pdf')
plt.close()  
