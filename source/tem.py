import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, BisectingKMeans, DBSCAN
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
GRAPH_RONCHI_PATH = Path('../graphs/ronchi')
GRAPH_RONCHI_PATH.mkdir(exist_ok=True,
                    parents=True)

# ------------------------ TEM ------------------------ #
TEM_1 = np.array(Image.open(DATA_PATH/'RedeTEM1_1_2aula.pgm'))
# Define the radius of the circle and its center coordinates
radius = 115
center = (TEM_1.shape[1] // 2, TEM_1.shape[0] // 2)
for x in range(TEM_1.shape[1]):
    for y in range(TEM_1.shape[0]):
        if np.sqrt((x - center[0] + 40)**2 + (y - center[1])**2) < radius:
            TEM_1[y, x] = 0
TEM_2 = np.array(Image.open(DATA_PATH/'RedeTEM2_1_2aula.pgm'))
TEM_3 = np.array(Image.open(DATA_PATH/'RedeTEM2_2_2aula.pgm'))
TEM_4 = np.array(Image.open(DATA_PATH/'RedeTEM3_1_2aula.pgm'))

def separate_list(main_list, separator_list):
    result = []
    sublist = []
    
    for value in separator_list:
        while main_list and value > 0:
            sublist.append(main_list.pop(0))
            value -= 1
        result.append(sublist)
        sublist = []

    return result


def tem_plot(image, k, points, center_dot=True):
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

    # Order the cluster centers by the x coordinate and suborder by the y coordinate
    cluster_centers = sorted(cluster_centers, key=lambda x: (x[0], x[1]))

    print(f"Cluster Centers: {cluster_centers}")

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
    print(f"Mean per Cluster: {mean_per_cluster}")
    
    list_all_points = []
    for i in range(len(cluster_centers)):
        list_all_points.append(list([cluster_centers[i][0], cluster_centers[i][1], mean_per_cluster[i]]))
    print(f"List All Points: {list_all_points}")

    # Select one then 3 then 5 then 3 then 1 points from the center
    points_per_cluster = []
    sequence = [1]  # Start with the initial value

    for i in range(1, k//2 - 1):
        if i <= k // 4 - 1:
            sequence.append(sequence[-1] + 2)
        else:
            sequence.append(sequence[-1] - 2)
    # subraction of 1 to the middle value
    sequence[len(sequence)//2] -= 1

    list_all_points_sorted_by_x = sorted(list_all_points, key=lambda x: x[0])
    list_all_points_sorted_by_y = sorted(list_all_points, key=lambda x: x[1])

    x_points = separate_list(list_all_points_sorted_by_x, sequence)
    y_points = separate_list(list_all_points_sorted_by_y, sequence)

    print(f"X Points: {x_points}\n")
    print(f"Y Points: {y_points}\n")

    # Choose only the x points and for distance and to the mean 
    x_points_updated = []
    for groups in x_points:
        value = 0
        for points in groups:
            value += points[0]
        x_points_updated.append(value / len(groups))

    # Choose only the y points and for distance and to the mean
    y_points_updated = []
    for groups in y_points:
        value = 0
        for points in groups:
            value += points[1]
        y_points_updated.append(value / len(groups))

    print(f"X Points: {x_points_updated}\n")
    print(f"Y Points: {y_points_updated}\n")

    # Create a list from -k//2 to k//2
    start = - ceil(k / 4) + 1
    maximums = np.array([start + i for i in range(ceil(k//2 - 1))])
    print(f"Maximums: {maximums}")

    # Matrix with distance from x points to x points
    distance_matrix_x = np.zeros((len(maximums),len(maximums)))

    for i in range(len(maximums)):
        for j in range(len(maximums)):
            if i < j:
                distance_matrix_x[i,j] = np.sqrt((x_points_updated[i] - x_points_updated[j])**2 + (y_points_updated[i] - y_points_updated[j])**2)
            elif i == j:
                distance_matrix_x[i,j] = 0
            elif i > j:
                distance_matrix_x[i,j] = - np.sqrt((x_points_updated[i] - x_points_updated[j])**2 + (y_points_updated[i] - y_points_updated[j])**2)

    # Matrix with distance from y points to y points
    distance_matrix_y = np.zeros((len(maximums),len(maximums)))

    for i in range(len(maximums)):
        for j in range(len(maximums)):
            if i < j:
                distance_matrix_y[i,j] = np.sqrt((y_points_updated[i] - y_points_updated[j])**2 + (x_points_updated[i] - x_points_updated[j])**2)
            elif i == j:
                distance_matrix_y[i,j] = 0
            elif i > j:
                distance_matrix_y[i,j] = - np.sqrt((y_points_updated[i] - y_points_updated[j])**2 + (x_points_updated[i] - x_points_updated[j])**2)
    
    # Plot maximums in x and center point of distance matrix in y 
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.plot(maximums, distance_matrix_x[ceil(len(distance_matrix_x)/2) - 1,:], 'o', label='Pontos Experimentais')
    plt.xlabel('n')
    plt.xticks(maximums)
    plt.ylabel(r'Distance ($\mu m$)')
    plt.title('Distance between Maximum Points')
    plt.show()

    # Plot maximums in y and center point of distance matrix in x
    plt.figure(figsize=(8,6))
    plt.grid()
    plt.plot(maximums, distance_matrix_y[ceil(len(distance_matrix_y)/2) - 1,:], 'o', label='Pontos Experimentais')
    plt.xlabel('n')
    plt.xticks(maximums)
    plt.ylabel(r'Distance ($\mu m$)')
    plt.title('Distance between Maximum Points')
    plt.show()

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data_points = np.array(list_all_points)

    # Extract x, y, and intensity values from the data points
    x = np.array([point[0] for point in data_points])
    y = np.array([point[1] for point in data_points])
    intensity = np.array([point[2] for point in data_points])

    # Find the center of the data points
    center = (np.mean(x), np.mean(y))

    # Subtract the center from the data points
    x -= center[0]
    y -= center[1]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define marker properties based on intensity values
    colors = intensity  # Use intensity for color mapping
    marker_size = [20 * i for i in intensity]  # Use intensity for marker size

    # Plot the 3D scatter plot
    scatter = ax.scatter(x, y, intensity, c=colors, s=marker_size, cmap='viridis')

    # Add labels and colorbar
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Intensity')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity')

    plt.title('3D Scatter Plot with Intensity')
    plt.show()
        

    




    



tem_plot(TEM_1, 12, 500)