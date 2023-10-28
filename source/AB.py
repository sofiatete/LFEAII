import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Set up paths
IMAGES_PATH = Path('../data')
IMAGES_PATH.mkdir(exist_ok=True,parents=True)

def analyse_AB_5aula(*args, **kwargs):

    IMAGE1 = 'AB_5aula.pgm'

    img = Image.open(IMAGES_PATH/IMAGE1)
    img_arr = np.array(img)
    print(f"Image shape: {img_arr.shape}")

    # array with intensitie and coordinates
    imag_arr_coord = np.array([np.array([i,j,img_arr[i,j]]) for i in range(img_arr.shape[0]) for j in range(img_arr.shape[1])])
    print(imag_arr_coord)

    # Find Maximum Values
    # sort imag_arr_coord by the third column and reverse the order
    sorted_img = imag_arr_coord[imag_arr_coord[:,2].argsort()][::-1]
    print(sorted_img)
    # Find Coordinates of Maximum Values
    x_coords = []
    y_coords = []
    points = 1000
    for point in sorted_img[:points]:
            x_coords.append(point[0])
            y_coords.append(point[1])

    # Define multiple x and y coordinates for red points
    # x_coords = [530, 529, 523]  
    # y_coords = [320, 729, 1144]  
    coordinates = list(zip(x_coords, y_coords))  # List of x, y coordinates
    # point_labels = ['A', 'B', 'C']  # Labels for the points


    # Plot image
    plt.imshow(img_arr, cmap='gray')
    plt.scatter(x_coords, y_coords, color='red', s=1, label='Points')
    # Add labels to the points
    # for i, label in enumerate(point_labels):
    #     plt.text(x_coords[i], y_coords[i] - 40, label, color='white', fontsize=6, ha='center', va='bottom')
    plt.axis('off')
    plt.show()





# # Gaussian 3d from image
# gaussian_img = img_arr
# gaussian_img_norm = (gaussian_img - np.min(gaussian_img)) / (np.max(gaussian_img) - np.min(gaussian_img))
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.mgrid[:gaussian_img_norm.shape[0], :gaussian_img_norm.shape[1]]
# ax.plot_surface(x, y, gaussian_img_norm, cmap='viridis')
# # plt.show()

if __name__ == '__main__':
    analyse_AB_5aula()







