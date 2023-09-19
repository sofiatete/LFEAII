import numpy as np
from scipy.fft import fft2, fftshift, fftfreq
import matplotlib.pyplot as plt
import cv2

# Load the image (grayscale)
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the 2D Fourier Transform
f_transform = fft2(image)

# Shift the zero frequency component to the center
f_transform_shifted = fftshift(f_transform)

# Calculate the magnitude spectrum
magnitude_spectrum = np.abs(f_transform_shifted)

# Generate the frequency axes
rows, cols = image.shape
frequencies_x = fftfreq(cols, 1)  # Frequency values for columns
frequencies_y = fftfreq(rows, 1)  # Frequency values for rows

# Create a meshgrid for plotting
frequencies_x, frequencies_y = np.meshgrid(frequencies_x, frequencies_y)

# Display the original image and magnitude spectrum
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.show()
