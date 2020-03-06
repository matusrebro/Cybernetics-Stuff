import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss, ss2tf

from control_systems import lin_model


t  = np.arange(0, 10, 0.1)
u = np.ones_like(t)


num = [1]
den = [1, 2, 1]

model1 = lin_model([num, den])

model1.input_count

x0 = [0, 0]

x, y = model1.simulation(x0, t, u)

plt.figure()
plt.plot(t, x)

omega = np.logspace(-3, 3, 100)

mag, phase, nyquist = model1.freq_response(omega)



A = [[-2, -1], [1, 0]]
B = [[1, 2], [0, 0]]
C = [[1, 2], [6, 2]] 
D = [[0, 0], [0, 0]]



model2 = lin_model([A, B, C, D])

model2.input_count



num = []

for k in range(2):
    numk, den = ss2tf(A, B, C, D, input = k)
    num.append(numk)


nyquist = []
mag = []
phase = []
index = 1
for j in range(2):
    for k in range(2):
        nuquistk = np.polyval(num[j][k], jomega) / np.polyval(den, jomega)
        magk = 20 * np.log10(nuquistk)
        phasek = np.rad2deg(np.angle(nuquistk))
        
        plt.figure(1)
        plt.suptitle('Nyquist plot')
        plt.subplot(2,2,index)
        plt.plot(np.real(nuquistk), np.imag(nuquistk))
        plt.xlabel(r'Re{G(j$\omega$)}')
        plt.ylabel(r'Im{G(j$\omega$)}')
        plt.grid()
        plt.tight_layout()
        
        index+=1
        nyquist.append(nuquistk)
        mag.append(magk)
        phase.append(phasek)


omega = np.logspace(-3, 3, 100)
jomega = 1j * omega
nyquist = np.polyval(num, jomega) / np.polyval(den, jomega)
mag = 20 * np.log10(nyquist)
phase = np.rad2deg(np.angle(nyquist))

