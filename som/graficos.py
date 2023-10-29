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

### Gráfico 328Hz

def func328(t, a, w, b, c):
    return a*np.sin(w*t+b) + c 


t = data328['t']
luma = data328['luma']
p0 = [-49.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func328, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_328 = w

plt.plot(data328['t'], data328['luma'])
plt.plot(t, func328(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('328Hz')
plt.grid(True)
plt.savefig('som/328Hz.png')
plt.show()

### Gráfico 334Hz

def func334(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data334['t']
luma = data334['luma']
p0 = [-120.6, -36.9, -9.9, 75]

params, covariance = curve_fit(func334, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_334 = w

plt.plot(data334['t'], data334['luma'])
plt.plot(t, func334(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('334Hz')
plt.grid(True)
plt.savefig('som/334Hz.png')
plt.show()

### Gráfico 340Hz

def func340(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data340['t']
luma = data340['luma']
p0 = [-89.6, -3.9, -9.9, 95]

params, covariance = curve_fit(func340, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_340 = w

plt.plot(data340['t'], data340['luma'])
plt.plot(t, func340(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('340Hz')
plt.grid(True)
plt.savefig('som/340Hz.png')
plt.show()

### Gráfico 346Hz

def func346(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data346['t']
luma = data346['luma']
p0 = [-89.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func346, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_346 = w

plt.plot(data346['t'], data346['luma'])
plt.plot(t, func346(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('346Hz')
plt.grid(True)
plt.savefig('som/346Hz.png')
plt.show()

### Gráfico 352Hz

def func352(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data352['t']
luma = data352['luma']
p0 = [-89.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func352, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_352 = w

plt.plot(data352['t'], data352['luma'])
plt.plot(t, func352(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('352Hz')
plt.grid(True)
plt.savefig('som/352Hz.png')
plt.show()

### Gráfico 354Hz

def func354(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data354['t']
luma = data354['luma']
p0 = [-100.6, -9.9, -9.9, 80]

params, covariance = curve_fit(func354, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_354 = w

plt.plot(data354['t'], data354['luma'])
plt.plot(t, func354(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('354Hz')
plt.grid(True)
plt.savefig('som/354Hz.png')
plt.show()

### Gráfico 358Hz

def func358(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data358['t']
luma = data358['luma']
p0 = [-99.6, -35.9, -20.9, 90]

params, covariance = curve_fit(func358, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_358 = w

plt.plot(data358['t'], data358['luma'])
plt.plot(t, func358(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('358Hz')
plt.grid(True)
plt.savefig('som/358Hz.png')
plt.show()

### Gráfico 360Hz

def func360(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data360['t']
luma = data360['luma']
p0 = [-100.6, -75.9, -9.9, 75]

params, covariance = curve_fit(func360, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_360 = w

plt.plot(data360['t'], data360['luma'])
plt.plot(t, func360(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('360Hz')
plt.grid(True)
plt.savefig('som/360Hz.png')
plt.show()

### Gráfico 362Hz

def func362(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data362['t']
luma = data362['luma']
p0 = [-100.6, -16.9, -9.9, 150]

params, covariance = curve_fit(func362, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_362 = w

plt.plot(data362['t'], data362['luma'])
plt.plot(t, func362(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('362Hz')
plt.grid(True)
plt.savefig('som/362Hz.png')
plt.show()

### Gráfico 364Hz

def func364(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data364['t']
luma = data364['luma']
p0 = [-49.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func364, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_364 = w

plt.plot(data364['t'], data364['luma'])
plt.plot(t, func364(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('364Hz')
plt.grid(True)
plt.savefig('som/364Hz.png')
plt.show()

### Gráfico 370Hz

def func370(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data370['t']
luma = data370['luma']
p0 = [-99.6, -34.9, -9.9, 75]

params, covariance = curve_fit(func370, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_370 = w

plt.plot(data370['t'], data370['luma'])
plt.plot(t, func370(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('370Hz')
plt.grid(True)
plt.savefig('som/370Hz.png')
plt.show()

### Gráfico 376Hz

def func376(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data376['t']
luma = data376['luma']
p0 = [-99.6, -9.4, -9.9, 75]

params, covariance = curve_fit(func376, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_376 = w

plt.plot(data376['t'], data376['luma'])
plt.plot(t, func376(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('376Hz')
plt.grid(True)
plt.savefig('som/376Hz.png')
plt.show()

### Gráfico 382Hz

def func382(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data382['t']
luma = data382['luma']
p0 = [-99.6, -46.9, -10.9, 75]

params, covariance = curve_fit(func382, t, luma, p0=p0)

# extrair parametros

a, w, b, c = params
w_382 = w

plt.plot(data382['t'], data382['luma'])

plt.plot(t, func382(t, a, w, b, c), label='fit')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('382Hz')
plt.grid(True)
plt.savefig('som/382Hz.png')
plt.show()

### Gráfico 388Hz

def func388(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data388['t']
luma = data388['luma']

p0 = [-49.6, -3.9, -9.9, 75]

params, covariance = curve_fit(func388, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_388 = w

plt.plot(data388['t'], data388['luma'])
plt.plot(t, func388(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('388Hz')
plt.grid(True)
plt.savefig('som/388Hz.png')
plt.show()

### Gráfico 394Hz

def func394(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data394['t']
luma = data394['luma']
p0 = [-150.6, -34.9, -9.9, 75]

params, covariance = curve_fit(func394, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_394 = w

plt.plot(data394['t'], data394['luma'])
plt.plot(t, func394(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('394Hz')
plt.grid(True)
plt.savefig('som/394Hz.png')
plt.show()

### Gráfico 400Hz

def func400(t, a, w, b, c):
    return a*np.sin(w*t+b) + c

t = data400['t']
luma = data400['luma']
p0 = [-809.6, -5.9, -9.9, 75]

params, covariance = curve_fit(func400, t, luma, p0=p0)

# extrair parametros
a, w, b, c = params
w_400 = w

plt.plot(data400['t'], data400['luma'])
plt.plot(t, func400(t, a, w, b, c), label='fit')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('400Hz')
plt.grid(True)
plt.savefig('som/400Hz.png')
plt.show()

print('w (328Hz) = ', w_328)
print('w (334Hz) = ', w_334)
print('w (340Hz) = ', w_340)
print('w (346Hz) = ', w_346)
print('w (352Hz) = ', w_352)
print('w (354Hz) = ', w_354)
print('w (358Hz) = ', w_358)
print('w (360Hz) = ', w_360)
print('w (362Hz) = ', w_362)
print('w (364Hz) = ', w_364)
print('w (370Hz) = ', w_370)
print('w (376Hz) = ', w_376)
print('w (382Hz) = ', w_382)
print('w (388Hz) = ', w_388)
print('w (394Hz) = ', w_394)
print('w (400Hz) = ', w_400)
