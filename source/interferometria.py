import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Set up paths

DATA_PATH = Path('../data')
DATA_PATH.mkdir(exist_ok=True,
                    parents=True)

c=299792458
def func(x, f, p):
    return abs(np.cos(f*x+p))

def linear(x, m, b):
    return m*x+b
def divcos(theta,lam):
    return lam/np.sin(theta)
def k(d, a, b, e1, e2, f1, f2):
    return 0
bound = np.array([[0.0109, 0.0124],
                 [0.02, 0.05],
                 [0.04, 0.08],
                 [0.08, 0.17]])*264.5/2
print(bound.shape)
dlinhas=np.zeros(4)
for images in DATA_PATH.iterdir():
    for k in range(4):
        if images.name == f"riscas{k}_aula6.pgm":
            print(images.name)
            # Load Image and convert to numpy array
            IMAGE = images.name
            img = Image.open(DATA_PATH/IMAGE)
            img_arr = np.array(img)
            #print(f"Image shape: {img_arr.shape}")

            # Plot image
            plt.imshow(img_arr, cmap='gray')
            plt.axis('off')
            plt.savefig(f'../interferometria/riscas{k}/padrao.png')
            plt.clf()

            #2D images
            data = np.zeros((5,10))
            Nlines = img_arr.shape[0]
            Ncols = img_arr.shape[1]
            x=np.mgrid[:Nlines:1]/264.5*2
            for i in range(10):
                img2d = img_arr[:,(Ncols*i)//10]
                img2d_norm = (img2d - np.min(img2d)) / (np.max(img2d) - np.min(img2d))
                y=img2d_norm
                popt, pcov = curve_fit(func, x, y, bounds=([bound[k,0],0],[bound[k,1], np.pi/2]))

                plt.plot(x, y, '.-', label="data")
                #print(popt)
                for j in range(5):
                    data[j,i] = (np.pi*((5+j)/2)-popt[1])/popt[0]
                plt.plot(x, func(x, *popt), 'r-', label='fit')
                plt.title(f"Oscillation {i}")
                plt.xlabel('mm')
                plt.ylabel('K')
                plt.savefig(f"../interferometria/riscas{k}/oscillation{i}.png")
                plt.clf()

            x = x[-1]*np.arange(10)/10
            ax = plt.gca()
            b = np.zeros(5)
            for i in range(5):
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(x, data[i,:], '.', color=color)
                popt, pcov = curve_fit(linear, x, data[i,:])
                plt.plot(x, linear(x, *popt), '-',color=color)
                b[i] = popt[1]
            dlinhas[k]=round(np.mean([b[1]-b[0], b[2]-b[1], b[3]-b[2], b[4]-b[3]]),3)
            ddlinhas = np.max(np.abs(np.array([b[1]-b[0], b[2]-b[1], b[3]-b[2], b[4]-b[3]])-np.mean([b[1]-b[0], b[2]-b[1], b[3]-b[2], b[4]-b[3]])))*1e-3
            print("Distance between lines: ", dlinhas[k], "mm")
            plt.title("Stripes fit")
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.savefig(f"../interferometria/riscas{k}/linear.png")
            plt.clf()
    
    """
    for i in range(2):
        for j in range(2):
            if images.name == f"centro{i}{j+1}_angulos_aula6.pgm":
                print(images.name)
                # Load Image and convert to numpy array
                IMAGE = images.name
                img = Image.open(DATA_PATH/IMAGE)
                img_arr = np.array(img)
                
                plt.imshow(img_arr, cmap='gray')
                plt.axis('off')
                plt.title(images.name)
                plt.show()
                plt.clf()
    for i in range(2):
        if images.name == f"centros{i+2}_angulos_aula6.pgm":
            print(images.name)
            # Load Image and convert to numpy array
            IMAGE = images.name
            img = Image.open(DATA_PATH/IMAGE)
            img_arr = np.array(img)
            
            plt.imshow(img_arr, cmap='gray')
            plt.axis('off')
            plt.title(images.name)
            plt.show()
            plt.clf()
    """
pontos=np.array([[932,695,924,593],[960,491,967,698],[960,385,967,698],[983,698,979,193]])
distancia=np.zeros(4)
ddistancia=np.zeros(4)
for i in range(4):
    ddistancia[i]=abs((pontos[i,0] - pontos[i,2]) / 1.3225e+5 / np.sqrt((pontos[i,0] - pontos[i,2]) ** 2 + (pontos[i,1] - pontos[i,3]) ** 2)) + abs((pontos[i,1] - pontos[i,0]) / 1.3225e+5 / np.sqrt((pontos[i,0] - pontos[i,2]) ** 2 + (pontos[i,1] - pontos[i,3]) ** 2)) + abs((pontos[i,2] - pontos[i,3]) / 1.3225e+5 / np.sqrt((pontos[i,0] - pontos[i,1]) ** 2 + (pontos[i,2] - pontos[i,3]) ** 2)) + abs((pontos[i,3] - pontos[i,2]) / 1.3225e+5 / np.sqrt((pontos[i,0] - pontos[i,1]) ** 2 + (pontos[i,2] - pontos[i,3]) ** 2))
    distancia[i]=np.sqrt((pontos[i,0]-pontos[i,2])**2+(pontos[i,1]-pontos[i,3])**2)*2/264.5/1000
angulos=np.arctan(distancia/1.11)
dangulos=np.abs(100 / 111 / (distancia**2 * 10000 / 12321 + 1)) * ddistancia
print("d Distância entre linhas: ", ddlinhas)
print("dÂngulos: ", dangulos)

print("Distâncias: ",distancia)
print("Distância entre linhas", dlinhas)
dlinhas = dlinhas*1e-3
print("Ângulos: ", angulos)
popt, pcov = curve_fit(divcos, angulos, dlinhas)

plt.plot(angulos, dlinhas, '.')
plt.errorbar(angulos, dlinhas, 
             yerr = (0.00019, 0.00008, 0.00008, 0.00012), 
             xerr = dangulos*3, 
             fmt ='o')
print("Comprimento de onda: ", *popt)
print(633e-9)
#plt.plot(0.004*np.arange(100)/100, divcos(0.004*np.arange(100)/100,633e-9),'-', label='633 nm')
plt.plot(0.004*np.arange(100)/100, divcos(0.004*np.arange(100)/100,5.584643765004666e-07),'-', label=f"fit: [{round(popt[0] *10e8)}+- 5] nm")

plt.xlabel('ângulo entre os centros [rad]')
plt.ylabel('distância entre linhas [m]')
plt.title('Espaçamento entre riscas em função do ângulo entre os feixes')
plt.xlim([0.0005,0.004])
plt.ylim([0,0.001])
plt.legend()
plt.savefig(f"../interferometria/espacamento.png")
plt.show()

plt.clf()





