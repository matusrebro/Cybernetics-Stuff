"""

In this script, frequency response of one-compartmental
fractional order system is shown: both analytic 
(via Laplace transform) and numeric (via simulations) 
solutions. Also the effect of fractional order on
the frequency response is shown on Bode plots.

"""
from scipy.special import binom
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

def zcross(y):
    ind=0
    for k in range(len(y)):
        if np.sign(y[k])>np.sign(y[k+1]):
            ind=k
            break
    return ind


alpha=0.5
a=1
b=1

# system response for periodic input signal
plt.figure()
sbn=311
for w in [1,5,10]:
    Ts=0.05/w
    Tsim=int(30/w)
    tt=np.arange(0,Tsim+Ts,Ts)
    idx_final=int(Tsim/Ts+1)

    yy=np.zeros(idx_final)
    u=np.sin(w*tt)
    yy[0]=0
    for i in range(1,idx_final):
        yy[i]=(Ts**alpha)*(-a*yy[i-1]+b*u[i])-omegasum(alpha,yy,i)
    plt.subplot(sbn)
    plt.title(r'$\omega=$'+str(w)+' [rad/s]')
    plt.plot(tt,u,label=r'$u(t)$')
    plt.plot(tt,yy,label=r'$x(t)$')
    plt.legend(loc=0,fontsize=15)
    plt.xlabel(r'$t$',fontsize=12)
    plt.ylabel(r'$x(t),u(t)$',fontsize=12)
    plt.legend(loc=0,fontsize=11)
    sbn=sbn+1
    
plt.tight_layout()
plt.savefig(r'figs/FODE_freqresp_demo_output.pdf')
plt.close()


# frequency response
# comparison of analytic and numeric solution
w=np.logspace(-1,3,1000)

plt.figure()
# amplitude
N=20*np.log10(np.abs(b/((1j*w)**alpha+1)))
plt.subplot(211)
plt.semilogx(w,N,label='Transfer function')
# phase
w=np.logspace(-1,3,1000)
N=np.rad2deg(np.angle(b/((1j*w)**alpha+1)))
plt.subplot(212)
plt.semilogx(w,N,label='Transfer function')


for w in [1,5,10]:
    Ts=0.05/w
    Tsim=int(30/w)
    tt=np.arange(0,Tsim+Ts,Ts)
    idx_final=int(Tsim/Ts+1)

    yy=np.zeros(idx_final)
    u=np.sin(w*tt)
    yy[0]=0
    for i in range(1,idx_final):
        yy[i]=(Ts**alpha)*(-a*yy[i-1]+b*u[i])-omegasum(alpha,yy,i)
    
    plt.subplot(211)
    # calculation of magnitude
    M=20*np.log10(np.max(yy[int(len(yy)/2):]))
    
    if w==1:      
        plt.semilogx(w,M,'ro',label='Simulation')
    else:
        plt.semilogx(w,M,'ro')
    
    plt.subplot(212)
    # calculation of phase
    T=2*np.pi/w
    deg=(zcross(u[int(len(u)/2):])-zcross(yy[int(len(yy)/2):]))*Ts/T*360
    if w==1:      
        plt.semilogx(w,deg,'ro',label='Simulation')
    else:
        plt.semilogx(w,deg,'ro')

plt.subplot(211)
plt.legend(loc=0,fontsize=15)
plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$M$ [dB]',fontsize=16)
plt.subplot(212)
plt.legend(loc=0,fontsize=15)
plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$M$ [dB]',fontsize=16)
plt.tight_layout()
plt.savefig('figs/FODE_freqresp_demo_Bode.pdf')
plt.close()



# frequency response comparison of different system orders
w=np.logspace(-1,3,1000)

# amplitude
plt.figure()
for alpha in [0.2,0.4,0.6,0.8,1.0]:
    N=20*np.log10(np.abs(b/((1j*w)**alpha+1)))
    plt.semilogx(w,N,label=r'$\alpha=$'+str(alpha))

plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$M$ [dB]',fontsize=16)
plt.legend(loc=0,fontsize=15)
plt.tight_layout()
plt.savefig('figs/FODE_freqresp_demo_Bode_mag_order_effect.pdf')
plt.close()


# phase
w=np.logspace(-1,3,1000)
plt.figure()
for alpha in [0.2,0.4,0.6,0.8,1.0]:
    N=np.rad2deg(np.angle(b/((1j*w)**alpha+1)))
    plt.semilogx(w,N,label=r'$\alpha=$'+str(alpha))

plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$\Delta\varphi$ [deg]',fontsize=16)
plt.legend(loc=0,fontsize=15)
plt.tight_layout()
plt.savefig('figs/FODE_freqresp_demo_Bode_phase_order_effect.pdf')
plt.close()
