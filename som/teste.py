import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func352(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

data352 = pd.read_csv('som/352Hz.csv')

t = data352['t']
luma = data352['luma']
p0 = [-49.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func352, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params

plt.plot(data352['t'], data352['luma'])
plt.plot(t, func352(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('352Hz')
plt.grid(True)
plt.show()
