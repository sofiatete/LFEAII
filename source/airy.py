import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from numpy import sinc
from scipy.optimize import curve_fit
from math import ceil, sqrt, pi
from scipy.stats import multivariate_normal
import matplotlib.patches as patches


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
GRAPH_AIRY_PATH = Path('../graphs/airy')
GRAPH_AIRY_PATH.mkdir(exist_ok=True,
                    parents=True)
CALIBRATION_PATH = Path('../graphs/calibration.txt')

AIRY_1 = np.array(Image.open(DATA_PATH/'Bessel_diam_iris_3,85mm_2aula.pgm'))
AIRY_1 = AIRY_1[703:753, 900:950]
AIRY_2 = np.array(Image.open('../png_images/Bessel_diam_iris_3_5mm_2aula.png'))
AIRY_2 = AIRY_2[703:753, 900:950]
AIRY_3 = np.array(Image.open(DATA_PATH/'Bessel_diam_iris_2mm_2aula.tif'))
AIRY_3 = AIRY_3[703:753, 900:950]
AIRY_4 = np.array(Image.open(DATA_PATH/'Bessel_diam_iris_3mm_2aula.tif'))
AIRY_4 = AIRY_4[703:753, 900:950]

def airy_plot(image, k, points, title=None):
    # Extract Parameters from calibration image
    with open(CALIBRATION_PATH, 'r') as f:
        m = float(f.readline().split(' ')[1])
        b = float(f.readline().split(' ')[1])

    # Plot image for visualization
    plt.imshow(image, cmap='gray')
    plt.xlabel(r'x ($pixels$)')
    plt.ylabel(r'y ($pixels$)')
    plt.title('Airy')
    plt.savefig(GRAPH_AIRY_PATH/f'{title}_image.png', dpi=400)
    plt.show()

    # Normalize Image using min-max normalization
    image = image / max(image.flatten())

    # Find Maximum Values 
    sorted_ronchi_1 = np.sort(image.flatten())

    # Find Coordinates of Maximum Values
    coordinates = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] in sorted_ronchi_1[-points:]:
                coordinates.append((x, y))

    # Intensity Values
    intensity = []
    for x, y in coordinates:
        intensity.append(image[y, x])
    
    # Plot Intensity Values
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define marker properties based on intensity values
    colors = intensity  # Use intensity for color mapping
    marker_size = [20 * i for i in intensity]  # Use intensity for marker size

    # Coordinates
    # for values in coordinates:
    #     print(values)
    #     values = values[0] * m + b, values[1] * m + b

    x = [i[0] for i in coordinates]
    y = [i[1] for i in coordinates]
    # Plot the 3D scatter plot
    scatter = ax.scatter(x, y, intensity, c=colors, s=marker_size, cmap='viridis')

    # Add labels and colorbar
    ax.set_xlabel('X-axis (pixels)')
    ax.set_ylabel('Y-axis (pixels)')
    ax.set_zlabel('Intensity (Gray Scale)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Intensity (Gray Scale)')

    plt.title('3D Scatter Plot with Intensity')
    # plt.show()

    # Define the function to fit

    estimated_mean = np.mean(coordinates, axis=0)
    estimated_cov = np.cov(coordinates, rowvar=False)
    print(f'Estimated Mean: {estimated_mean}')
    print(f'Estimated Covariance: {estimated_cov}')

    mvn = multivariate_normal(mean=estimated_mean, cov=estimated_cov, allow_singular=True)

    # Meshgrid for plotting

    x, y = np.mgrid[0:50:1, 0:60:1]
    pos = np.dstack((x, y))
    zeta = mvn.pdf(pos)
    # Normalize zeta
    zeta = zeta / max(zeta.flatten())
    #ax.scatter(x, y, zeta, c='red')
    ax.plot_surface(x, y, zeta, cmap='viridis', alpha=0.5)
    plt.savefig(GRAPH_AIRY_PATH/f'{title}_3d.png', dpi=400)
    plt.show()


def resolution_plot(points, title=None):
    # Extract Parameters from calibration image
    with open(CALIBRATION_PATH, 'r') as f:
        m = float(f.readline().split(' ')[1])
        b = float(f.readline().split(' ')[1])
    x_res = [i[0] for i in points]
    y_res = [(i[1] * m + b)/2 * 1000 for i in points]
    # Fit a resolution function 
    # Fit a custom model
    # Define the sinc function
    def resolution(x, A, B):
        return A / x + B
    

    # Fit the model
    popt, pcov = curve_fit(resolution, x_res, y_res, p0=[1, 0.09])

    print(f'popt: {popt}')
    print(f'pcov: {pcov}')

    # X grid
    x_grid = np.linspace(min(x_res), max(x_res) + 1, 100)

    # Plot the result
    plt.figure(figsize=(8,6))
    plt.plot(x_grid, resolution(x_grid, *popt), label='Fit Function')
    plt.plot(x_res, y_res, label='Experimental Points', marker='o', linestyle='None')
    plt.xlabel('D (mm)')
    plt.ylabel('R (mm)')
    plt.title('Resolution')
    plt.legend()
    plt.savefig(GRAPH_AIRY_PATH/f'{title}_resolution.png', dpi=400)
    plt.show()

def airy_resolution_plot(images, points, iris_diameter=None):
    # Extract Parameters from calibration image
    with open(CALIBRATION_PATH, 'r') as f:
        m = float(f.readline().split(' ')[1])
        b = float(f.readline().split(' ')[1])
    print(f'{m}, {b}')
    for x_y in points:
        print(((x_y[0]*10e-6) - b) / m)

    x_radius_points = []
    for image in images:
        # Plot image for visualization
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

        # Copy image
        img_copy = image.copy()

        # Extract Parameters from calibration image
        with open('../graphs/calibraton.txt', 'r') as f:
            m = float(f.readline().split(' ')[1])
            b = float(f.readline().split(' ')[1])

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
        
        # Find Clusters of points 
        kmeans = KMeans(n_clusters=1, random_state=0, n_init='auto').fit(coordinates)

        # Plot Cluster Centers with diferent colors side with image
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,2)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1,2,1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.scatter([x[0] for x in kmeans.cluster_centers_], [x[1] for x in kmeans.cluster_centers_], c='red', s=10, label='Points')

        # Plot a circle with cluster center and cluster radius

        # Get the single cluster center
        cluster_center = kmeans.cluster_centers_[0][0]

        # Calculate the distances of data points to the cluster center
        distances_to_center = np.abs(coordinates - cluster_center)

        # Calculate the cluster radius as the standard deviation of the distances
        cluster_radius = np.std(distances_to_center)
        x_radius_points.append(cluster_radius)

        print(f'Cluster Radius: {cluster_radius}')

        # # Plot the cluster center and and a circle representing the cluster radius
        # # Define the center and radius
        # center = cluster_center  # Replace with your actual center coordinates
        # radius = cluster_radius  # Replace with your actual radius


        # # Create a drawing context
        # draw = ImageDraw.Draw(Image.fromarray(img_copy))

        # # Define the center and radius of the circle
        # center = (kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1])  # Replace with your circle's center coordinates

        # # Define the circle's outline color
        # outline_color = (255, 0, 0)  # Red color (RGB)

        # # Draw the circle on the image
        # draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline='red')

        # # Plot the image
        # plt.imshow(img_copy, cmap='gray')
        # plt.axis('off')
        # plt.show()





if __name__ == '__main__':
    airy_plot(AIRY_1, 30, 300, title='Bessel_diam_iris_3,85mm_2aula')
    resolution_plot([(1.20, 70), (1.40, 50), (2.10, 36), (3.85, 10)], title='Resolution')
