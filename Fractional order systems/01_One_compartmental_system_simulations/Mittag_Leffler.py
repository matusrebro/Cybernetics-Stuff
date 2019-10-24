import numpy as np

from numpy import pi, log, log10, abs, min, max, exp, sin, cos, floor, fix, angle
from scipy.special import gamma

# Mittag-Leffler function implementation
# alf and bet are the function parameters
# z is the function argument
# fi is the order of calculation precision
def mlf(alf,bet,z,fi=6):
    if bet<0:
        rc=(-2*log(10**(-fi)*pi/(6*(abs(bet)+2)*(2*abs(bet))**(abs(bet)))))**alf
    else:
        rc=(-2*log(10**(-fi)*pi/6))**alf
        
    r0=max([1,2*abs(z),rc])
        
    
    if alf==1 and bet==1:
        e=exp(z)
    elif (alf<1 and abs(z)<=1) or ((1<=alf and alf<2) and abs(z)<=floor(20/(2.1-alf)**(5.5-2*alf))) or (alf>=2 and abs(z)<=50):
        oldsum=0
        k=0
        while alf*k+bet<=0:
            k+=1
        newsum=z**k/gamma(alf*k+bet)
        
        while newsum!=oldsum:
            oldsum=newsum
            k+=1
            term=z**k/gamma(alf*k+bet)
            newsum=newsum+term
            k+=1
            term=z**k/gamma(alf*k+bet)
            newsum=newsum+term;
            
        e=newsum

    else:
        if (alf<=1 and abs(z)<=fix(5*alf+10)):
            if ((abs(angle(z))>pi*alf) and (abs(abs(angle(z))-(pi*alf))>10**(-fi))):
                if bet<=1:
                    e=rombint(K,(alf,bet,z),0,r0,fi)
                else:
                    eps=1
                    e=rombint(K,(alf,bet,z),0,r0,fi)+rombint(P,(alf,bet,z,eps),-pi*alf,pi*alf,fi)
            elif (abs(angle(z))<pi*alf and abs(abs(angle(z))-(pi*alf))>10**(-fi)):
                if bet<=1:
                    e=rombint(K,(alf,bet,z),0,r0,fi) + (z**((1-bet)/alf))*(exp(z**(1/alf))/alf)
                else:
                    eps=abs(z)/2
                    e=rombint(K,(alf,bet,z),0,r0,fi)+rombint(P,(alf,bet,z,eps),-pi*alf,pi*alf,fi)+(z**((1-bet)/alf))*(exp(z**(1/alf))/alf)
            else:
                eps=abs(z)+0.5
                e=rombint(K,(alf,bet,z),0,r0,fi)+rombint(P,(alf,bet,z,eps),-pi*alf,pi*alf,fi)
        else:
            if alf<=1:
                if (abs(angle(z))<(pi*alf/2+min([pi,pi*alf]))/2):
                    newsum=(z**((1-bet)/alf))*exp(z**(1/alf))/alf
                    
                    for k in range(1,int(floor(fi/log10(abs(z))))+1):
                        newsum-=((z**(-k))/gamma(bet-alf*k))
                    e=newsum
                else:
                    newsum=0
                    
                    for k in range(1,int(floor(fi/log10(abs(z))))+1):
                        newsum-=((z**(-k))/gamma(bet-alf*k))
                    e=newsum
            else:
                if alf>=2:
                    m=int(floor(alf/2))
                    sum1=0
                    for h in range(0,m+1):
                        zn=(z**(1/(m+1)))*exp((2*pi*1j*h)/(m+1))
                        sum1+=mlf(alf/(m+1),bet,zn,fi)
                    e=1/(m+1)*sum1
                else:
                    e=(mlf(alf/2,bet,z**(1/2),fi)+mlf(alf/2,bet,-z**(1/2),fi))/2
                    
    return e

# here are some auxiliary functions, which are needed in mlf function

# Romberg integration method
def rombint(funfcn,args,a,b,order=6,):
    rom=np.zeros([2,order])
    h=b-a
    rom[0,0]=h*(funfcn(a,*args)+funfcn(b,*args))/2
    
    ipower=1
    for i in range(1,order):
        sum1=0
        for j in range(1,ipower+1):
            sum1+=funfcn(a+h*(j-0.5),*args)
        rom[1,0]=(rom[0,0]+h*sum1)/2
        for k in range(1,i+1):
            rom[1,k]=((4**k)*rom[1,k-1]-rom[0,k-1])/((4**k)-1)
        
        for j in range(-1,i):
            rom[0,j+1]=rom[1,j+1]
        ipower*=2
        h/=2
    
    res=rom[0,-1]
    
    return res
        

    
def K(r,alfa,beta,z):
    res=r**((1-beta)/alfa)*exp(-r**(1/alfa))*(r*sin(pi*(1-beta))-z*sin(pi*(1-beta+alfa)))/(pi*alfa*(r**2-2*r*z*cos(pi*alfa)+z**2))  
    return res
            

def P(r,alfa,beta,z,eps):
    w=(eps**(1/alfa))*sin(r/alfa)+r*(1+(1-beta)/alfa)
    res=((eps**(1+(1-beta)/alfa))/(2*pi*alfa))*((exp((eps**(1/alfa))*cos(r/alfa))*(cos(w)+1j*sin(w))))/(eps*exp(1j*r)-z)
    return res      
    

# this should be mlf derivative
# not sure if it works properly    
def dmlf(alfa,beta,z,fi=6):
    if z==0:
        z=1e-10
    if np.abs(z)<=0.5:
        ro=10**(-fi)
        D=alfa**2-4*alfa*beta+6*alfa+1
        omega=alfa+beta-3/2
        k1=0
        if alfa>1:
            k1=int((2-alfa-beta)/(alfa-1))+1
        elif 0<alfa and alfa<=1 and D<=0:
            k1=int((3-alfa-beta)/alfa)+1
        elif 0<alfa and alfa<=1 and D>0:
            k1=np.max([int((3-alfa-beta)/alfa)+1,int((1-2*omega*alfa+np.sqrt(D))/(2*alfa**2))+1])
    
        k0=np.max([k1,int(np.log(ro*(1-np.abs(z)))/np.log(np.abs(z)))])
        
        newsum=0
        for k in range(0,k0+1):
            newsum+=(k+1)*(z**k)/(gamma(alfa+beta+alfa*k))
        
        e=newsum
        
    elif np.abs(z)>0.5:
        e=(mlf(alfa,beta-1,z)-(beta-1)*mlf(alfa,beta,z))/(alfa*z)
        
    return -e
    
# n-th order derivative of mlf
# also not sure about its functionality
def mlfn(alfa,beta,z,n):
    if n==0:
        return mlf(alfa,beta,z)
    else:
        h=mlfn(alfa,beta-1,z,n-1)
        A=mlfn(alfa,beta,z,n-1)
        h-=(beta-float(n-1)*alfa-1)*float(A)
        return (1/z/alfa)*h