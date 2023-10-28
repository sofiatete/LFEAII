import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import itertools
import string


# Set up paths
IMAGES_PATH = Path('../data')
IMAGES_PATH.mkdir(exist_ok=True,parents=True)

def analyse_image(name: str, points: int, *args, **kwargs):
    """
    Analyse an image and find the clusters of points
    name: name of the image
    points: number of points to be used to find the clusters
    """

    IMAGE = name + '.pgm'

    img = Image.open(IMAGES_PATH/IMAGE)
    img_arr = np.array(img)
    # print(f"Image shape: {img_arr.shape}")

    # center of the image
    center = (img_arr.shape[1]//2 - 30, img_arr.shape[0]//2 + 10)

    # array with intensitie and coordinates
    imag_arr_coord = np.array([np.array([j,i,img_arr[i,j]]) for i in range(img_arr.shape[0]) for j in range(img_arr.shape[1])])
    # print(imag_arr_coord)

   
    # Find Maximum Values
    # sort imag_arr_coord by the third column and reverse the order
    sorted_img = imag_arr_coord[imag_arr_coord[:,2].argsort()][::-1]
    # print(sorted_img)

    # Remove the points that are in a circle of radius from the center
    radius = 350
    sorted_img = sorted_img[np.sqrt((sorted_img[:,0] - center[0])**2 + (sorted_img[:,1] - center[1])**2) > radius]
    # Remove the points that are in a circle of radius outside the center
    radius = 450
    sorted_img = sorted_img[np.sqrt((sorted_img[:,0] - center[0])**2 + (sorted_img[:,1] - center[1])**2) < radius]

    # Find Coordinates of Maximum Values
    x_coords = []
    y_coords = []
    for point in sorted_img[:points]:
            x_coords.append(point[0])
            y_coords.append(point[1])

    
    # Find Clusters of points 
    kmeans = KMeans(n_clusters=4, random_state=0).fit(sorted_img[:points])

    # sort kmeans.cluster_centers_ by the first column
    kmeans.cluster_centers_ = kmeans.cluster_centers_[kmeans.cluster_centers_[:,0].argsort()]

    point_labels = ['A', 'B', 'C', 'D']


    # Plot image
    plt.imshow(img_arr, cmap='gray')
    # plot axis
    plt.xlabel('$x$ (pixels)')
    plt.ylabel('$y$ (pixels)')

    # plot title
    plt.title(f'Image {name} ')

    plt.legend()
    plt.savefig(f'../graphs/{name}_image.png', dpi=400)
    # plot the center in green
    plt.scatter(center[0], center[1], color='green', s=10, label='Center \n {}'.format(center))
    # plot the points used to find the clusters
    plt.scatter(x_coords, y_coords, color='red', s=1, label='Points')
    # plot the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='orange', s=10, label='Centroids')
    # plot a label for each centroid with its letter and coordinates values
    for i, label in enumerate(point_labels):
        plt.text(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1] - 40, label + ' ' + str((kmeans.cluster_centers_[i, 0].round(0), kmeans.cluster_centers_[i, 1].round(0)))
                 , color='white', fontsize=7, ha='center', va='bottom')

    # plot title
    plt.title(f'Image {name} with {points} points')
    
    plt.legend()
    # plt.show()
    plt.savefig(f'../graphs/{name}_clusters.png', dpi=400)
    plt.close()

    # return the coordinates of the clusters centroids
    return kmeans.cluster_centers_

if __name__ == '__main__':

    coordinates_1 = analyse_image('AB1_5aula', points = 2000)
    coordinates_2 = analyse_image('AB2_5aula', points = 2000)
    coordinates_3 = analyse_image('AB3_5aula', points = 10)
    coordinates_4 = analyse_image('AB4_5aula', points = 2000)
    
    # mean of the coordinates of the clusters centroids
    mean = np.mean([coordinates_1, coordinates_2, coordinates_3, coordinates_4], axis=0)
    
    print('Mean of the coordinates of the clusters centroids:')
    print('point A: (' + str(mean[0, 0].round(0)) + ', ' + str(mean[0, 1].round(0)) + ')   ' + 'Intensity: ' + str(mean[0, 2].round(0)))
    print('point B: (' + str(mean[1, 0].round(0)) + ', ' + str(mean[1, 1].round(0)) + ')   ' + 'Intensity: ' + str(mean[1, 2].round(0)))
    print('point C: (' + str(mean[2, 0].round(0)) + ', ' + str(mean[2, 1].round(0)) + ')   ' + 'Intensity: ' + str(mean[2, 2].round(0)))
    print('point D: (' + str(mean[3, 0].round(0)) + ', ' + str(mean[3, 1].round(0)) + ')   ' + 'Intensity: ' + str(mean[3, 2].round(0)) + '\n')

    # Distance of each point considering A and D as being located in the x axis and B and C in the y axis
    center = np.array([934, 734])
    print('Distance in pixels along x from center to point A: ' + str(mean[0, 0].round(0) - center[0]) + ' pixels')
    print('Distance in pixels along x from center to point D: ' + str(mean[3, 0].round(0) - center[0]) + ' pixels')
    print('Distance in pixels along y from center to point C: ' + str(mean[2, 1].round(0) - center[1]) + ' pixels')
    print('Distance in pixels along y from center to point B: ' + str(mean[1, 1].round(0) - center[1]) + ' pixels')

    # Calibration





