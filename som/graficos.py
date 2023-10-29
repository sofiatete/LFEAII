import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

data = pd.read_csv('som/428Hz.csv')

print(data)

def func(t, a, w, b, c):
    return a*np.sin(w*t+b) + c 


t = data['t']
luma = data['luma']
p0 = [-49.6, -3.9, -9.9, 75]
# Fit the function to the data
params, covariance = curve_fit(func, t, luma, p0=p0)

print(params)

# Extract the parameters
a, w, b, c = params

plt.plot(data['t'], data['luma'])
plt.plot(t, func(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('428Hz')
plt.grid(True)
plt.savefig("428Hz.png")
plt.show()