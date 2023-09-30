import matplotlib.pyplot as plt
from LightPipes import *

# Define the parameters of the optical system
wavelength = 632.8 * nm   # Wavelength of the light (e.g., HeNe laser)
size = 5 * mm             # Size of the grid (5 mm)
N = 500                   # Number of grid points
f = 100 * cm              # Focal length of the lens (e.g., 100 cm)
z = f                    # Propagation distance to the Fourier plane (focal length)

# Create an instance of the Field class for the input field
F = Begin(size, wavelength, N)

# Define the object as two rectangular apertures (A and B)
width_A = 0.4 * mm  # Width of aperture A
height_A = 1.5 * mm  # Height of aperture A
x_offset_A = -0.7 * mm  # Horizontal offset for aperture A
F_A = RectAperture(width_A, height_A, x_offset_A, 0, 1, F)

width_B = 0.4 * mm  # Width of aperture B
height_B = 1.5 * mm  # Height of aperture B
x_offset_B = 0.7 * mm  # Horizontal offset for aperture B
F_B = RectAperture(width_B, height_B, x_offset_B, 0, 1, F)

# Combine the two apertures to form the object
F_object = BeamMix(F_A, F_B)

# Propagate the field through the lens
F = Lens(f, 0, 0, F_object)

# Calculate the intensity distribution at the object plane and Fourier plane
I_object = Intensity(0, F_object)
I_FT = Intensity(0, F)

# Display the object (A and B apertures) and its Fourier Transform (AB image) separately
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(I_object, cmap='gray', extent=[-size/2, size/2, -size/2, size/2])
plt.title('Object (A and B Apertures)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.subplot(122)
plt.imshow(I_FT, cmap='gray', extent=[-size/2, size/2, -size/2, size/2])
plt.title('Fourier Transform (AB Image) of Object')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

plt.tight_layout()
plt.show()
