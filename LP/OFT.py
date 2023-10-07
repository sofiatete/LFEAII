from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

f=0.2*m # focal length of the lens
gridsize=8*mm # size of the grid to adjust the resolution
wavelength=633*nm





def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def Fouriertransform(F):
    # F is a field
    F=Lens(Forvard(f,F),f,0,0)
    return F







# Import the oject whose FT is to be calculated
image = 'ABI'
A=rgb2gray(mpimg.imread(image + '.png'))

N=A.shape[0]
X=range(N)
Z=range(N)
X, Z=np.meshgrid(X,Z)




# Begin the field with the same size as the object
F1=Begin(gridsize,wavelength,N) # N is the GridDimension

# Fill the field with the object
F1=MultIntensity(A,F1)

# Calculate the FT of the field
F1=Fouriertransform(F1)

# coloco os filtros espaciais aqui ou quando faço a intensidade?
aperture_diameter = 2*mm # Diameter of the aperture (in mm)
# F1 = CircAperture(F1, aperture_diameter / 2, x_shift=0, y_shift=0)
# F1 = CircScreen(F1, aperture_diameter / 2, 0, 0)



# Calculate the intensity of the FT

filter = True

if filter:
    # Com filtro espacial
    I_FT=Intensity(0,CircScreen(Fouriertransform(F1), aperture_diameter / 2, 0, 0)) # porque é que ao guardar a intensidade voltamos a fazer a FT? 
else:
    # Sem filtro espacial
    I_FT=Intensity(0,Fouriertransform(F1)) # porque é que ao guardar a intensidade voltamos a fazer a FT? I_FT=Intensity(0,F1)



# Plot the intensity of the FT
vmax_value = 0.9
plt.imshow(I_FT,cmap='hot', extent=[0,gridsize*1000,0,gridsize*1000], vmin=0, vmax=vmax_value)
plt.xlabel('X (mm)')  # Label for the X-axis
plt.ylabel('Y (mm)')  # Label for the Y-axis
plt.title('Intensity Pattern of FT')
plt.colorbar()
plt.savefig('OFT.' + image + '.png', dpi=300)
if filter:
    plt.savefig('OFT_' + image + '_filtered_' + str(aperture_diameter) + '_vmax_' + str(vmax_value) + '.png', dpi=300)
else:
    plt.savefig('OFT_' + image + '_vmax_' + str(vmax_value) + '.png', dpi=300)


plt.show()
# Command + W to close the window

