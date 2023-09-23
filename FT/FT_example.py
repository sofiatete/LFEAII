import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Define the parameters of the step function
t = np.linspace(0, 2 * np.pi, 1000)  # Time values from 0 to 2*pi
frequency = 15  # Frequency of the step function in Hz
step_function = np.sign(np.sin(2 * np.pi * frequency * t))

# Perform the Fourier transform
fft_result = fft(step_function)

# Frequency values corresponding to the FFT result
frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

# Normalize the Fourier transform to match the scale of the square wave
normalized_fft = np.abs(fft_result) / max(np.abs(fft_result))

# Plot the original step function and its Fourier transform
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, step_function)
plt.title('Pattern of the diffraction grating')
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.xlim([0, 2 * np.pi])

plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_result), color='orange')
plt.title('Fourier Transform')
plt.xlabel('Spacial Frequency of the object in the x direction')
plt.ylabel('Intensity of the frequency in the fourier image')

plt.tight_layout()
plt.savefig('FT_example' + str(frequency) + '.png')
plt.show()

# lightpipes
