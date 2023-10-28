import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Set up paths
IMAGES_PATH = Path('../images')
IMAGES_PATH.mkdir(exist_ok=True,
                    parents=True)
DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True,
                    parents=True)

for images in IMAGES_PATH.iterdir():
    if images.name == 'ab1.pgm':
        print(images.name)
        
        # Load Image and convert to numpy array
        IMAGE = images.name
        img = Image.open(IMAGES_PATH/IMAGE)
        img_arr = np.array(img)
        print(f"Image shape: {img_arr.shape}")

        # Plot image
        plt.imshow(img_arr, cmap='gray')
        plt.axis('off')
        plt.show()

        # Gaussian 3d from image 
        gaussian_img = img_arr
        gaussian_img_norm = (gaussian_img - np.min(gaussian_img)) / (np.max(gaussian_img) - np.min(gaussian_img))
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.mgrid[:gaussian_img_norm.shape[0], :gaussian_img_norm.shape[1]]
        ax.plot_surface(x, y, gaussian_img_norm, cmap='viridis')
        plt.show()






