import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
import os

# Set up paths
RAW_IMAGES_PATH = Path('../raw_images')
RAW_IMAGES_PATH.mkdir(exist_ok=True,
                    parents=True)
PNG_IMAGES_PATH = Path('../png_images')
PNG_IMAGES_PATH.mkdir(exist_ok=True,
                    parents=True)
IMAGES_PATH = Path('../images')
IMAGES_PATH.mkdir(exist_ok=True,
                    parents=True)
DATA_PATH = Path('../data')
DATA_PATH.mkdir(exist_ok=True,
                    parents=True)
GRAPH_PATH = Path('../graphs')
GRAPH_PATH.mkdir(exist_ok=True,
                    parents=True)
RAW_VIDEOS_PATH = Path('../raw_videos')
RAW_VIDEOS_PATH.mkdir(exist_ok=True,
                    parents=True)

# ------------------------ All to PNG ------------------------ #
for images in DATA_PATH.iterdir():
    IMAGE = images.name
    if IMAGE[-3:] != 'pgm':
        os.system('cp ' + str(DATA_PATH/IMAGE) + ' ' + str(RAW_VIDEOS_PATH/IMAGE))
        continue
    img = Image.open(DATA_PATH/IMAGE)
    img = np.array(img)

    # Plot image
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(PNG_IMAGES_PATH/f"{IMAGE[:-4]}.png")

