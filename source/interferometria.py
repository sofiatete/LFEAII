import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Set up paths

DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True,
                    parents=True)

c=299792458
def func(x, f, p):
    return abs(np.cos(f*x+p))

def linear(x, m, b):
    return m*x+b
bound = np.array([[0.0109, 0.0124],
                 [0.02, 0.05],
                 [0.04, 0.08],
                 [0.08, 0.15]])
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
            x=np.mgrid[:Nlines:1]
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
                plt.xlabel('pixels')
                plt.ylabel('K')
                plt.savefig(f"../interferometria/riscas{k}/oscillation{i}.png")
                plt.clf()
            
            #print(data)

            x = x[-1]*np.arange(10)
            ax = plt.gca()
            b = np.zeros(5)
            for i in range(5):
                color = next(ax._get_lines.prop_cycler)['color']
                plt.plot(x, data[i,:], '.', color=color)
                popt, pcov = curve_fit(linear, x, data[i,:])
                plt.plot(x, linear(x, *popt), '.-',color=color)
                b[i] = popt[1]
            print("Distance between lines: ", np.mean([b[1]-b[0], b[2]-b[1], b[3]-b[2], b[4]-b[3]]))
            plt.savefig(f"../interferometria/riscas{k}/linear.png")
            plt.clf()


        """
        # Gaussian 3d from image 
        gaussian_img = img_arr
        gaussian_img_norm = (gaussian_img - np.min(gaussian_img)) / (np.max(gaussian_img) - np.min(gaussian_img))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.mgrid[:gaussian_img_norm.shape[0], :gaussian_img_norm.shape[1]]
        print("x: ", np.mgrid[:gaussian_img_norm.shape[0]:1])
        ax.plot_surface(x, y, gaussian_img_norm, cmap='viridis')
        plt.show()
        """






