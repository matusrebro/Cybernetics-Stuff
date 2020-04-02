import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

t = t + 1
t = np.insert(t, 0, 0)

S = np.insert(S, 0, N)
I = np.insert(I, 0, 0)
R = np.insert(R, 0, 0)

plt.figure()
plt.semilogy(t, S, 'k', label = 'S')
plt.semilogy(t, I, 'r', label = 'I')
plt.semilogy(t, R, 'b', label = 'R')
plt.legend()

def fcn_sir(t, x, par, N):
    S, I, R = x
    beta, gamma = par
    
    S_dot = -beta/N * I*S
    I_dot = beta/N * I*S - gamma * I
    R_dot = gamma * I
    return np.array([S_dot, I_dot, R_dot])

