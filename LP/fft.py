#%% Playing with 2d Fouier Transform

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from LightPipes import *
from scipy.fftpack import fft2 as fft
from numpy.fft import fftshift, ifftshift
from matplotlib.colors import LinearSegmentedColormap


# Define the colors for your custom colormap
colors = ['#FFFF00', '#0000FF']  # Yellow to Blue

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list('YellowBlue', colors, N=256)


wavelength = 792*nm
size = 100*mm
N = 1000
w0=3*mm

# Fields
sq = np.zeros([100,100])
sq[25:75, 25:75] = 1
F=Begin(size,wavelength,N)
I0 = Intensity(0,GaussBeam(F, w0, LG=True, n=0, m=0))
I1 = Intensity(0,GaussBeam(F, w0, LG=False, n=0, m=1))+Intensity(0,GaussBeam(F, w0, LG=False, n=1, m=0))

# Plot transforms
f = sq
F = fftshift(fft(ifftshift(f)))
F_F = fftshift(fft(ifftshift(F)))

plt.subplot(331), plt.imshow(f,cmap = cmap)
plt.title(r'f'), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(np.abs(F),cmap = cmap)
plt.title(r'F\{f\}'), plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(np.abs(F_F),cmap = cmap)
plt.title('F\{F\{f\}\}'), plt.xticks([]), plt.yticks([])

f = I0
F = fftshift(fft(ifftshift(f)))
F_F = fftshift(fft(ifftshift(F)))

plt.subplot(334), plt.imshow(f[450:550,450:550],cmap = cmap)
plt.title(r'f'), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(np.abs(F)[450:550,450:550],cmap = cmap)
plt.title(r'F\{f\}'), plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(np.abs(F_F)[450:550,450:550],cmap = cmap)
plt.title('F\{F\{f\}\}'), plt.xticks([]), plt.yticks([])

f = I1
F = fftshift(fft(ifftshift(f)))
F_F = fftshift(fft(ifftshift(F)))

plt.subplot(337), plt.imshow(f[450:550,450:550],cmap = cmap)
plt.title(r'f'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(np.abs(F)[450:550,450:550],cmap = cmap)
plt.title(r'F\{f\}'), plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(np.abs(F_F)[450:550,450:550],cmap = cmap)
plt.title('F\{F\{f\}\}'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()