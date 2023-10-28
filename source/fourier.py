import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from numpy import sinc
from scipy.optimize import curve_fit
from math import ceil


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


# ------------------------ Redes de Ronchi ------------------------ #
RONCHI_1 = np.array(Image.open(DATA_PATH/'ronchi_iris_3_50mm.pgm'))
RONCHI_2 = np.array(Image.open(DATA_PATH/'ronchi_iris_5_75mm.pgm'))
RONCHI_3 = np.array(Image.open(DATA_PATH/'ronchi_iris_8_55mm.pgm'))
RONCHI_4 = np.array(Image.open(DATA_PATH/'ronchi_iris_10_35mm.pgm'))
RONCHI_5 = np.array(Image.open(DATA_PATH/'ronchi_iris_aberta_aula2.pgm'))
RONCHI_6 = np.array(Image.open(DATA_PATH/'ronchi1_2aula.pgm'))
RONCHI_7 = np.array(Image.open(DATA_PATH/'ronchi2_2aula.pgm'))

def ronchi_plot(image, k, points, odd=True):
    # Normalize Image
    image = image / max(image.flatten())

    # Find Maximum Values 
    sorted_ronchi_1 = np.sort(image.flatten())

    # Find Coordinates of Maximum Values
    coordinates = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] in sorted_ronchi_1[-points:]:
                coordinates.append((x, y))

    # Remove Outliers points with low coordinates
    coordinates = [x for x in coordinates if x[0] > 200 and x[1] > 200]

    # Find Clusters of points 
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(coordinates)

    # Plot Cluster Centers with diferent colors side with image
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.scatter([x[0] for x in kmeans.cluster_centers_], [x[1] for x in kmeans.cluster_centers_], c='red', s=10, label='Points')
    plt.show()


    # List of center of clusters rounded
    cluster_centers = [(round(x[0]), round(x[1])) for x in kmeans.cluster_centers_]

    # Sort cluster_centers by y coordinate
    cluster_centers.sort(key=lambda x: x[0])
    print(f'centers: {cluster_centers}')

    # Matrix with distance from each cluster center to each cluster center
    distance_matrix = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i < j:
                distance_matrix[i,j] = np.sqrt((cluster_centers[i][0] - cluster_centers[j][0])**2 + (cluster_centers[i][1] - cluster_centers[j][1])**2)
            elif i > j:
                distance_matrix[i,j] = - np.sqrt((cluster_centers[i][0] - cluster_centers[j][0])**2 + (cluster_centers[i][1] - cluster_centers[j][1])**2)


    distance_center_point = distance_matrix[round(len(distance_matrix)/2),:]

    start = - ceil(k / 2) + 1
    x = np.array([start + i for i in range(k)])

    # Fit a linear model
    model = np.polyfit(x, distance_center_point, 1)
    # Plot linear model
    plt.figure(figsize=(8,6))
    plt.plot(x, distance_center_point, 'o')
    plt.plot(x, model[0]*x + model[1])
    plt.xlabel('n')
    plt.ylabel('Distance')
    plt.title('Distance between Maximum Points')
    plt.show()


    # All cordinates in a center I define
    mean_per_cluster = []
    for i in range(k):
        points = []
        x_points = range(cluster_centers[i][0] - 10, cluster_centers[i][0] + 10)
        y_points = range(cluster_centers[i][1] - 10, cluster_centers[i][1] + 10)
        points = []
        for x in x_points:
            for y in y_points:
                points.append((x,y))
        mean_per_cluster.append(np.mean([image[co[1], co[0]] for co in points]))
    
    mean_per_cluster.sort()
    mean_odd = [mean_per_cluster[i] for i in range(len(mean_per_cluster)) if i % 2 == 0]
    mean_even = [mean_per_cluster[i] for i in range(len(mean_per_cluster)) if i % 2 == 1]
    mean_even.sort(reverse=True)
    mean_per_cluster = mean_odd + mean_even


    #mean_per_cluster = [image[co[1], co[0]] for co in cluster_centers]
    
    cluster_mean = mean_per_cluster.copy()
    start = - ceil(k / 2) + 1
    x = np.array([start + i for i in range(k)])
    # Plot Mean Value per cluster
    plt.figure(figsize=(8,6))
    plt.plot(x, cluster_mean, 'o')
    plt.xlabel('n')
    plt.ylabel('Mean Value')
    plt.title('Mean Value per Cluster')
    plt.show()

    # Fit a custom model
    # Define the sinc function
    def sinc_function(x, A, B):
        return A * np.sinc(B * x) * np.sinc(B * x)
    
    # Fit the model
    popt, pcov = curve_fit(sinc_function, x, cluster_mean, p0=[1, 1])

    # Plot the result
    plt.figure(figsize=(8,6))
    plt.plot(x, cluster_mean, 'o')
    plt.plot(x, sinc_function(x, *popt))
    plt.xlabel('n')
    plt.ylabel('Mean Value')
    plt.title('Mean Value per Cluster')
    plt.show()


ronchi_plot(RONCHI_1, 5, 200)
ronchi_plot(RONCHI_2, 5, 200)
ronchi_plot(RONCHI_3, 5, 200)
ronchi_plot(RONCHI_4, 5, 200)
ronchi_plot(RONCHI_5, 7, 1000)
ronchi_plot(RONCHI_6, 7, 1000)
ronchi_plot(RONCHI_7, 7, 1000)












# --------------------------- Redes TEM --------------------------- #
# ---------------------------- Slide AB --------------------------- #
# ---------------------------- Resolução --------------------------- #

