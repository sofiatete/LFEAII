import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# import de todos os ficheiros
data328 = pd.read_csv('som/328Hz.csv')
data334 = pd.read_csv('som/334Hz.csv')
data340 = pd.read_csv('som/340Hz.csv')
data346 = pd.read_csv('som/346Hz.csv')
data352 = pd.read_csv('som/352Hz.csv')
data354 = pd.read_csv('som/354Hz.csv')
data358 = pd.read_csv('som/358Hz.csv')
data360 = pd.read_csv('som/360Hz.csv')
data362 = pd.read_csv('som/362Hz.csv')
data364 = pd.read_csv('som/364Hz.csv')
data370 = pd.read_csv('som/370Hz.csv')
data376 = pd.read_csv('som/376Hz.csv')
data382 = pd.read_csv('som/382Hz.csv')
data388 = pd.read_csv('som/388Hz.csv')
data394 = pd.read_csv('som/394Hz.csv')
data400 = pd.read_csv('som/400Hz.csv')


def func328(t, a, w, b, c):
    return a*np.sin(w*t+b) + c 


t = data328['t']
luma = data328['luma']
p0 = [-49.6, -3.9, -9.9, 75]
# Fit the function to the data
params, covariance = curve_fit(func, t, luma, p0=p0)

print(params)

# Extract the parameters
a, w, b, c = params

plt.plot(data328['t'], data328['luma'])
plt.plot(t, func(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('428Hz')
plt.grid(True)
plt.savefig("428Hz.png")
plt.show()