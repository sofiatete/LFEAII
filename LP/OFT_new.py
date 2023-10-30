from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

f=0.2475*m # focal length of the lens
gridsize=9*mm # size of the grid to adjust the resolution
wavelength=633*nm
lambda_1 = 633 * (10**(-9)) # m

# calcular as distâncias certas
pixeis = 4096


# Tamanho real de AB em metros
alturatotal_unidadespc = 724
larguratotal_unidadespc = 964
altura_AB_unidades_pc = 290.03
largura_AB_unidades_pc = 248.03
pixeis_altura_AB = pixeis * altura_AB_unidades_pc / alturatotal_unidadespc
pixeis_largura_AB = pixeis * largura_AB_unidades_pc / larguratotal_unidadespc
print('Altura de AB em pixeis para uma altura de imagem de ' + str(pixeis) + ' pixeis: ' + str(pixeis_altura_AB) + ' pixeis') 
print('Largura de AB em pixeis para uma largura de imagem de ' + str(pixeis) + ' pixeis: ' + str(pixeis_largura_AB) + ' pixeis')

# em metros
altura = 5.5 * (10**(-3)) # m
largura = 5 * (10**(-3)) # m
alturatotal_metros = alturatotal_unidadespc * altura / altura_AB_unidades_pc
alturatotal_metro_medidaccraveira = 7 * (10**(-3)) # é aproximadamente a distância de abertura da íris - a abertura da zona iluminada do objeto - altura e largura do quadradro da simulação
print('A altura total da imagem em metros é ' + str(alturatotal_metros) + ' m')







# calcular o tamanho para AB em pixeis

# 1 é o índice da calibração - foram feitas várias medições para obter a calibração
craveira_1 = 0.004 # m
len_1 = 491.22
len_B_1 = 407.46
len_A_1 = 398.76
distancia_real_B = len_B_1 * craveira_1 /len_1 # distância no eixo dos xx em metros
distancia_real_A = len_A_1 * craveira_1 /len_1 # distância no eixo dos yy em metros
# conversão para unidades de frequência:
frequencia_B_x = distancia_real_B / (lambda_1 * f)
frequência_A_y = distancia_real_A / (lambda_1 * f)
print('Frequência espacial das riscas horizontais de A = ' + str(frequência_A_y) + ' ciclos/m')
print('Número de riscas horizontais de A = ' + str(altura/2 * frequência_A_y) + ' ciclos')
print('Largura de riscas horizontais de A = ' + str(1/frequência_A_y) + ' m = ' + str(pixeis_altura_AB / altura / frequência_A_y) + ' pixeis')

print('Frequência espacial das riscas verticais de B = ' + str(frequencia_B_x) + ' ciclos/m')
print('Número de riscas verticais de B = ' + str(largura/2 * frequencia_B_x) + ' ciclos')
print('Largura de riscas verticais de B = ' + str(1/frequencia_B_x) + ' m = ' + str(pixeis_largura_AB / largura / frequencia_B_x) + ' pixeis')


# largura quadrado filtro = 23mm - 4096 pixeis
# espaçamento  - x pixeis
frequencia_espacial_AB = (frequencia_B_x + frequência_A_y) / 2
espaçamento = 1 / frequencia_espacial_AB
espaçamento_pixeis = pixeis * espaçamento / alturatotal_metro_medidaccraveira
print('\n Espaçamento entre riscas = ' + str(espaçamento) + ' m = ' + str(espaçamento_pixeis) + ' pixeis')
# 4096 - 7
# x - 5.5
altura_AB_pixeis = altura * pixeis / alturatotal_metro_medidaccraveira
largura_AB_pixeis = largura * pixeis / alturatotal_metro_medidaccraveira
print('Altura de AB em pixeis para uma altura de imagem de ' + str(pixeis) + ' pixeis: ' + str(altura_AB_pixeis) + ' pixeis')
print('Largura de AB em pixeis para uma largura de imagem de ' + str(pixeis) + ' pixeis: ' + str(largura_AB_pixeis) + ' pixeis \n')




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def Fouriertransform(F):
    # F is a field
    F=Forvard(f,Lens(Forvard(f,F),f,0,0))
    return F







# Import the oject whose FT is to be calculated
image = 'ABS6'
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
aperture_diameter = 1*mm # Diameter of the aperture (in mm)
# F1 = CircAperture(F1, aperture_diameter / 2, x_shift=0, y_shift=0)
# F1 = CircScreen(F1, aperture_diameter / 2, 0, 0)



# Calculate the intensity of the FT

filter = True

if filter:
    # Com filtro espacial
    I_FT=Intensity(0,CircScreen(F1, aperture_diameter / 2, 0, 0)) # porque é que ao guardar a intensidade voltamos a fazer a FT? 
else:
    # Sem filtro espacial
    I_FT=Intensity(0,F1) # porque é que ao guardar a intensidade voltamos a fazer a FT? I_FT=Intensity(0,F1)



# Plot the intensity of the FT
vmax_value = 1
plt.imshow(I_FT,cmap='hot', extent=[0,gridsize*1000,0,gridsize*1000], vmin=0, vmax=vmax_value)
plt.xlabel('X (mm)')  # Label for the X-axis
plt.ylabel('Y (mm)')  # Label for the Y-axis
plt.title('Intensity Pattern of FT')
plt.colorbar()
# plt.savefig('OFT.' + image + '.png', dpi=300)
if filter:
    plt.savefig('OFT_' + image + '_filtered_' + str(aperture_diameter) + '_vmax_' + str(vmax_value) + '.png', dpi=300)
else:
    plt.savefig('OFT_' + image + '_vmax_' + str(vmax_value) + '.png', dpi=300)


# plt.show()
plt.close()
# Command + W to close the window


# propagate the field f more in order to se AB again
F2=Forvard(1*f,F1)
I2=Intensity(f,F2)
vmax_value_2 = 1.3
plt.imshow(I2,cmap='gray', extent=[0,gridsize*1000,0,gridsize*1000], vmin=0, vmax=vmax_value_2)
plt.xlabel('X (mm)')  # Label for the X-axis
plt.ylabel('Y (mm)')  # Label for the Y-axis
plt.title('Intensity Pattern of FT')
plt.colorbar()
plt.savefig('OFT_object.' + image + '.png', dpi=300)
plt.show()
plt.close()

F2=Forvard(0.8*f,F2)
I2=Intensity(f,F2)
vmax_value_2 = 0.7
plt.imshow(I2,cmap='gray', extent=[0,gridsize*1000,0,gridsize*1000], vmin=0, vmax=vmax_value_2)
plt.xlabel('X (mm)')  # Label for the X-axis
plt.ylabel('Y (mm)')  # Label for the Y-axis
plt.title('Intensity Pattern of FT')
plt.colorbar()
plt.savefig('OFT_object2.' + image + '.png', dpi=300)
plt.show()
plt.close()

