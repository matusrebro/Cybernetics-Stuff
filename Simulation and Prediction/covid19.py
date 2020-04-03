import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

confirmed_cases_global_link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
deaths_cases_global_link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_cases_global_link = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

response_confirmed = requests.get(confirmed_cases_global_link)
response_deaths = requests.get(deaths_cases_global_link)
response_recovered = requests.get(recovered_cases_global_link)

file = open("time_series_covid19_confirmed_global.csv", "w")
file.write(response_confirmed.text)
file.close()

file = open("time_series_covid19_deaths_global.csv", "w")
file.write(response_deaths.text)
file.close()

file = open("time_series_covid19_recovered_global.csv", "w")
file.write(response_recovered.text)
file.close()

df_confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('time_series_covid19_deaths_global.csv')
df_recovered = pd.read_csv('time_series_covid19_recovered_global.csv')

dates = df_confirmed.columns.values[4:]
t = np.arange(0, len(dates)) # days from 22.1.2020

confirmed_data = df_confirmed[df_confirmed.columns[4:]]
confirmed_data_global = confirmed_data.sum(0)

deaths_data = df_deaths[df_deaths.columns[4:]]
deaths_data_global = deaths_data.sum(0)

recovered_data = df_recovered[df_recovered.columns[4:]]
recovered_data_global = recovered_data.sum(0)

plt.figure()
plt.plot(t, confirmed_data_global, 'ro', label = 'confirmed')
plt.plot(t, deaths_data_global, 'ko', label = 'deaths')
plt.plot(t, recovered_data_global, 'go', label = 'recovered')
plt.legend()


N = 7.8e9
I = np.asarray(confirmed_data_global)
R = np.asarray(deaths_data_global) + np.asarray(recovered_data_global)
S = N - I - R

plt.figure()
plt.semilogy(t, S, 'k', label = 'S')
plt.semilogy(t, I, 'r', label = 'I')
plt.semilogy(t, R, 'b', label = 'R')
plt.legend()

def fcn_sir(x, t, par, N):
    S, I, R = x
    beta, gamma = par
    
    S_dot = -beta/N * I*S
    I_dot = beta/N * I*S - gamma * I
    R_dot = gamma * I
    return np.array([S_dot, I_dot, R_dot])


def sim_sir(t, x0, par, N):
    
    Ts = t[1]-t[0] 
    idx_final = int(t[-1]/Ts)+1  
    
    x = np.zeros([len(t), len(x0)])
    x[0, :] = x0
 
    for i in range(1,idx_final):
        y = odeint(fcn_sir, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                 args=(par,
                       N, )
                 )
        x[i,:] = y[-1,:]
    return x
    


def lsq_sir(par, N, t, sir_data):
    
    Sdata = sir_data[:, 0]
    Idata = sir_data[:, 1]
    Rdata = sir_data[:, 2]
    
    x0 = [Sdata[0], Idata[0], Rdata[0]]
    x = sim_sir(t, x0, par, N)
    
    S = x[:,0]
    I = x[:,1]
    R = x[:,2]
    
    return np.sum(S - Sdata)**2 + np.sum(I - Idata)**2 + np.sum(R - Rdata)**2


sir_data = np.vstack((S, I))
sir_data = np.vstack((sir_data, R))
sir_data = np.transpose(sir_data)

p = [0, 0]
res = minimize(lsq_sir, p, args=(N, t, sir_data), method='Nelder-Mead')

par = res.x

# par = [100, 1]
x0 = [S[0], I[0], R[0]]
x = sim_sir(t, x0, par, N)

S_sim = x[:,0]
I_sim = x[:,1]
R_sim = x[:,2]

plt.figure()
plt.subplot(311)
plt.title('Susceptible')
plt.plot(t, S, 'k.', label = 'data')
plt.plot(t, S_sim, label = 'simulation')
plt.legend()
plt.subplot(312)
plt.title('Infected')
plt.plot(t, I, 'k.', label = 'data')
plt.plot(t, I_sim, label = 'simulation')
plt.legend()
plt.subplot(313)
plt.title('Removed = recovered + deaths')
plt.plot(t, R, 'k.', label = 'data')
plt.plot(t, R_sim, label = 'simulation')
plt.legend()
plt.tight_layout()
