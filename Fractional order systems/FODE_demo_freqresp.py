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
plt.savefig(r'FODE_freqresp_demo_output_'+str(w)+'.pdf')
plt.close()


# frequency response
# comparison of analytic and numeric solution
w=np.arange(0,1000.1,0.1)

# amplitude
N=20*np.log10(np.abs(b/((1j*w)**alpha+1)))
plt.figure()
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
    
    M=20*np.log10(np.max(yy[int(len(yy)/2):]))
    if w==1:      
        plt.semilogx(w,M,'ro',label='Simulation')
    else:
        plt.semilogx(w,M,'ro')
    

plt.legend(loc=0,fontsize=15)

plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$M$ [dB]',fontsize=16)
plt.legend(loc=0,fontsize=15)
plt.tight_layout()
plt.savefig('FODE_freqresp_demo_Bode_mag.pdf')
plt.close()


# phase
w=np.arange(0,1000.1,0.1)
N=np.rad2deg(np.angle(b/((1j*w)**alpha+1)))
plt.figure()
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
    
    T=2*np.pi/w
    deg=(zcross(u[int(len(u)/2):])-zcross(yy[int(len(yy)/2):]))*Ts/T*360
    if w==1:      
        plt.semilogx(w,deg,'ro',label='Simulation')
    else:
        plt.semilogx(w,deg,'ro')

plt.legend(loc=0,fontsize=15)

plt.xlabel(r'$\omega$ [rad/s]',fontsize=16)
plt.ylabel(r'$\Delta\varphi$ [deg]',fontsize=16)
plt.legend(loc=0,fontsize=15)
plt.tight_layout()
plt.savefig('FODE_freqresp_demo_Bode_phase.pdf')
plt.close()


