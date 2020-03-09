"""

packaged control systems module

"""


import numpy as np
from scipy.integrate import odeint
from scipy.signal import tf2ss, ss2tf
import matplotlib.pyplot as plt

def fun_lin_sys_odes(x,t,u,A,B):
    return np.squeeze(np.dot(A,x)+np.dot(B,u))

def sim_lin_sys_ss(u, x0, A, B, C, D, Ts):
    x = odeint(fun_lin_sys_odes, x0, np.linspace(0,Ts),
             args=(u,A,B), rtol=1e-5
             )[-1,:]
    
    y = np.dot(C ,x) + np.dot(D, u)
    return x, y


class lin_model:
    
    A = 0
    B = 0
    C = 0
    D = 0
    
    num = []
    den = []
    
    input_count = 0
    output_count = 0
    
    x0 = 0
    
    
    def __init__(self, parameters):
        
        if len(parameters) == 4:
            self.A, self.B, self.C, self.D = parameters
            self.A = np.asarray(self.A)
            self.B = np.asarray(self.B)
            self.C = np.asarray(self.C)
            self.D = np.asarray(self.D)
            self.num, self.den = ss2tf(self.A, self.B, self.C, self.D)
            
        elif len(parameters) ==2:
            self.num, self.den = parameters
            self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)

        else:
            raise ValueError("Invalid parameter format")

        if len(self.C.shape)==1:
            self.output_count = 1
        else:
            self.output_count = self.C.shape[0]

        if len(self.B.shape)==1:
            self.input_count = 1
        else:
            self.input_count = self.B.shape[1]

        if self.input_count = 1:
            self.num, self.den = ss2tf(self.A, self.B, self.C, self.D)
        else:
            self.num = []
            for k in range(self.input_count):
                numk, self.den = ss2tf(self.A, self.B, self.C, self.D, input = k)
                self.num.append(numk)

    def set_parameters(self, parameters):
    
        if len(parameters) == 4:
            self.A, self.B, self.C, self.D = parameters
            self.A = np.asarray(self.A)
            self.B = np.asarray(self.B)
            self.C = np.asarray(self.C)
            self.D = np.asarray(self.D)
            
        elif len(parameters) ==2:
            self.num, self.den = parameters
            self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)
    
        else:
            raise ValueError("Invalid parameter format")
            
        if len(self.C.shape)==1:
            self.output_count = 1
        else:
            self.output_count = self.C.shape[0]

        if len(self.B.shape)==1:
            self.input_count = 1
        else:
            self.input_count = self.B.shape[1]
            
        if self.input_count = 1:
            self.num, self.den = ss2tf(self.A, self.B, self.C, self.D)
        else:
            self.num = []
            for k in range(self.input_count):
                numk, self.den = ss2tf(self.A, self.B, self.C, self.D, input = k)
                self.num.append(numk)
            
    def simulation(self, x0, t, u):
        self.x0 = x0
        
        Ts = t[1] - t[0]
        
        x=np.zeros([len(t), len(x0)])

        y = np.zeros([len(t), self.output_count])

        if len(u.shape) == 1:
            u.shape = (u.shape[0], 1)
        
        for k in range(1, len(t)):
            self.x0, y[k] = sim_lin_sys_ss(u[k-1,:], self.x0, self.A, self.B, self.C, self.D, Ts)
            x[k, :] = self.x0
            
        return x, y
    
    
    def freq_response(self, omega, plot = True):
        jomega = 1j * omega
        if self.input_count==1 and self.output_count==1:
            nyquist = np.polyval(self.num, jomega) / np.polyval(self.den, jomega)
            mag = 20 * np.log10(nyquist)
            phase = np.rad2deg(np.angle(nyquist))
            
            if plot:
                plt.figure(1)
                plt.title('Nyquist plot')
                plt.plot(np.real(nyquist), np.imag(nyquist))
                plt.xlabel(r'Re{G(j$\omega$)}')
                plt.ylabel(r'Im{G(j$\omega$)}')
                plt.grid()
                plt.tight_layout()
                
                plt.figure(2)
                plt.suptitle('Bode plot')
                plt.subplot(211)
                plt.semilogx(omega,mag)
                plt.xlabel(r'$\omega$ [rad\s]')
                plt.ylabel(r'Magnitude [dB]')
                plt.grid()
                plt.subplot(212)
                plt.semilogx(omega,phase)
                plt.xlabel(r'$\omega$ [rad\s]')
                plt.ylabel(r'Phase [deg]')
                plt.grid()
                plt.tight_layout()
                plt.subplots_adjust(top=0.92)
        else:
            nyquist = []
            mag = []
            phase = []
            index = 1
            
            for j in range(self.output_count):
                for k in range(self.input_count):
                    nuquistk = np.polyval(self.num[k][j], jomega) / np.polyval(self.den, jomega)
                    magk = 20 * np.log10(np.abs(nuquistk))
                    phasek = np.rad2deg(np.angle(nuquistk))
                    
                    plt.figure(1)
                    plt.suptitle('Nyquist plot')
                    plt.subplot(self.output_count,self.input_count,index)
                    plt.title('u'+str(k+1)+' -> y'+str(j+1))
                    plt.plot(np.real(nuquistk), np.imag(nuquistk))
                    plt.xlabel(r'Re{G(j$\omega$)}')
                    plt.ylabel(r'Im{G(j$\omega$)}')
                    plt.grid()
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    
                    plt.figure(index+1)
                    plt.suptitle('Bode plot: u'+str(k+1)+' -> y'+str(j+1))
                    plt.subplot(211)
                    plt.semilogx(omega,magk)
                    plt.xlabel(r'$\omega$ [rad\s]')
                    plt.ylabel(r'Magnitude [dB]')
                    plt.grid()
                    plt.subplot(212)
                    plt.semilogx(omega,phasek)
                    plt.xlabel(r'$\omega$ [rad\s]')
                    plt.ylabel(r'Phase [deg]')
                    plt.grid()
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.92)
                    
                    index+=1
                    nyquist.append(nuquistk)
                    mag.append(magk)
                    phase.append(phasek)
            
        return mag, phase, nyquist
    
    