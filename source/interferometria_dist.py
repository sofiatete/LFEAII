import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import os

# Set up paths

DATA_PATH = Path('../data_others')
DATA_PATH.mkdir(exist_ok=True,
                    parents=True)

def func(x, f, p):
    return abs(np.cos(f*x+p))
def linear(x, m, b):
    return m*x+b
dlinhas=np.zeros(4)
di=0
k=np.zeros(17)
i=0
def fk(d, a, b, e1, e2, f1, f2):
    return a/(1+e1+e2)*np.sqrt(1+e1*e1+e2*e2+2*(e1*np.cos(f1*d)+e2*np.cos(f2*d)+e1*e2*np.cos((f1-f2)*d)))+b
for images in DATA_PATH.iterdir():
    if images.name == "calib.png":
        continue
    print(images.name)
    # Load Image and convert to numpy array
    IMAGE = images.name
    img = Image.open(DATA_PATH/IMAGE)
    img_arr = np.array(img)
    #print(f"Image shape: {img_arr.shape}")

    # Plot image
    print("grayscale maximum: ",np.max(img_arr))
    print("grayscale minimum: ",np.min(img_arr))
    img_norm = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
    ma=0
    mi=0
    countma=0
    countmi=0
    for lista in img_norm:
        for val in lista:
            if val < 0.25:
                mi+=val
                countmi+=1
            if val > 0.75:
                ma+=val
                countma+=1
    ma= ma/countma*(np.max(img_arr) - np.min(img_arr))+np.min(img_arr)
    mi= mi/countmi*(np.max(img_arr) - np.min(img_arr))+np.min(img_arr)
    

    k[i]=(ma-mi)/(ma+mi)

    i+=1
    plt.imshow(img_arr, cmap='gray')
    plt.axis('off')
    plt.savefig(f'../interferometria/distancias/padrao.png')
    #plt.show()
    plt.clf()
    """
    if not os.path.exists(f'../interferometria/distancias/{IMAGE[:-3]}'):
        os.mkdir(f'../interferometria/distancias/{IMAGE[:-3]}')
    #2D images
    data = np.zeros((5,10))
    Nlines = img_arr.shape[0]
    Ncols = img_arr.shape[1]
    x=np.mgrid[:Nlines:1]/264.5*2
    for i in range(10):
        img2d = img_arr[:,(Ncols*i)//10]
        img2d_norm = (img2d - np.min(img2d)) / (np.max(img2d) - np.min(img2d))
        y=img2d_norm
        popt, pcov = curve_fit(func, x, y, bounds=([2,0],[6, np.pi/2]))

        plt.plot(x, y, '-', label="data")
        #print(popt)
        for j in range(5):
            data[j,i] = (np.pi*((5+j)/2)-popt[1])/popt[0]
        #plt.plot(x, func(x, *popt), 'r-', label='fit')
        plt.title(f"Oscillation {i}")
        plt.xlabel('mm')
        plt.ylabel('K')
        plt.savefig(f"../interferometria/distancias/{IMAGE[:-3]}/oscillation{i}.png")
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

    plt.title("Stripes fit")
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.savefig(f"../interferometria/distancias/{IMAGE[:-3]}/linear.png")
    plt.clf()
    """


#plt.plot(np.arange(0,3*17,3),k,".")
plt.errorbar(np.arange(0,3*17,3)/100,k,xerr=0.1/100, yerr=0.005,fmt=".-")
#popt, pcov = curve_fit(fk, np.arange(0,3*17,3)/100, k,bounds=([1,-5,0.1,0.1,0,0],[5,0,0.2,0.2,100,100]))
#plt.plot(np.arange(0,3*17,0.1)/100, fk(np.arange(0,3*17,0.1)/100, *popt), '-', label="fit")
#plt.plot(np.arange(0,3*17,0.1)/100, fk(np.arange(0,3*17,0.1)/100, 3,-2,0.18,0.125,17.7,23.5), '-', label="ideal")
#print("valores: ", *popt)
plt.xlabel("offset from equal lenghts [m]")
plt.ylabel("K")
plt.title("Variação do contraste com a distância entre os braços")
plt.legend()
plt.savefig("../interferometria/distancias.png")
plt.show()



