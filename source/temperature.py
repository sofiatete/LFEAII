import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, MeanShift
from numpy import sinc
from scipy.optimize import curve_fit
from math import ceil, sin
from sklearn.cluster import DBSCAN
import math
from scipy.ndimage import convolve1d

# Set Up Paths
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
GRAPH_TEMPERATURE_PATH = Path('../graphs/airy')
GRAPH_TEMPERATURE_PATH.mkdir(exist_ok=True,
                    parents=True)


TEMPERATURE_1 = np.array(Image.open(DATA_PATH/'ferro150.pgm'))
# Open Image and Give Coordinates for Points 
line = [ (431, 338), (543, 343), (1452, 458), (1689, 500), (19, 306), (1786, 532)]
curve_points = [
    (779.8, 374.4), (812.5, 385.5), (857.3, 401.6), (893.0, 417.7), (926.7, 437.8),
    (954.9, 454.4), (989.7, 433.8), (1022.4, 421.2), (1067.2, 411.6), (1111.4, 410.6)
]

TEMPERATURE_2 = np.array(Image.open(DATA_PATH/'ferro300.pgm'))
# Open Image and Give Coordinates for Points
line_2 = [
    (60, 280), (187, 283), (325, 302), (423, 305), (520, 317), (637, 325), (1311, 412),
    (1392, 412), (1392, 422), (1498, 436), (1609, 456), (1759, 479)
]
curve_points_2 = [
    (60, 280), (187, 283), (325, 302), (423, 305), (520, 317), (637, 325), (1311, 412),
    (1392, 412), (1392, 422), (1498, 436), (1609, 456), (1759, 479)
]

TEMPERATURE_3 = np.array(Image.open(DATA_PATH/'ferro450.pgm'))
# Open Image and Give Coordinates for Points
line_3 = [
    (42, 258), (157, 265), (239, 272), (330, 277), (418, 282), (520, 287), (632, 301),
    (1328, 390), (1434, 403), (1522, 417), (1623, 429), (1711, 449)
]
curve_points_3 = [(60, 280), (187, 283), (325, 302), (423, 305), (520, 317), (637, 325), (1311, 412),
    (1392, 412), (1392, 422), (1498, 436), (1609, 456), (1759, 479)
]


# ------------------------ Temperatura ------------------------ #
def temperature_plot(image, line_points, curve_points, adjusted_x=0, box_y=None, box_x=None, Rad=None):
    # Visualize image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Plot line points
    x, y = zip(*line_points)

    # Make a regression line
    m, b = np.polyfit(x, y, 1)

    # Plot regression line
    plt.imshow(image, cmap='gray')
    plt.plot(x, y, 'o')
    plt.plot(x, m*np.array(x) + b)
    plt.axis('off')
    plt.show()

   # Plot curve points
    x_curve, y_curve = zip(*curve_points)
    plt.plot(x_curve, y_curve, 'o')

    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


    # Normalize Image
    image = image / max(image.flatten())

    # Find Coordinates of Maximum Values
    coordinates = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] < 0.2:
                coordinates.append((x, y))

    # Plot Points
    plt.imshow(image, cmap='gray')
    plt.scatter([x[0] for x in coordinates], [x[1] for x in coordinates], c='red', s=10, label='Points')
    plt.axis('off')
    plt.show()
    x_curve, y_curve = zip(*coordinates)


    # Define the angle of rotation in radians (e.g., 45 degrees)
    angle_radians = - math.atan(m)

    # Determine the rotation center (origin) as the intersection with the line
    rotation_center_x = - b / m
    rotation_center_y = 0.0

    # Translate the points to align the rotation center with the origin
    translated_x = [x - rotation_center_x for x in x_curve]
    translated_y = [y - rotation_center_y for y in y_curve]

    # Perform the rotation using the rotation matrix
    rotated_x = [x * np.cos(angle_radians) - y * np.sin(angle_radians) for x, y in zip(translated_x, translated_y)]
    rotated_y = [x * np.sin(angle_radians) + y * np.cos(angle_radians) for x, y in zip(translated_x, translated_y)]

    # Translate the rotated points back to their original position
    # rotated_x = [x + rotation_center_x for x in rotated_x]
    # rotated_y = [y + rotation_center_y for y in rotated_y]

    # Plot the rotated points in a new figure
    plt.plot(rotated_x, rotated_y, 'o')
    plt.grid()
    plt.show()



    # Use box to select
    new_rotated_y = []
    new_rotated_x = []
    for i in range(len(rotated_x)):
        if rotated_y[i] > box_y[0] and rotated_y[i] < box_y[1] and rotated_x[i] > box_x[0] and rotated_x[i] < box_x[1]:
            new_rotated_y.append(rotated_y[i])
            new_rotated_x.append(rotated_x[i])

    # Plot the rotated points in a new figure
    plt.plot(new_rotated_x, new_rotated_y, 'o')
    plt.grid()
    plt.show()

    # Perform Sliding Average
    # Define the size of the sliding window (kernel)
    window_size = 1000  # Adjust as needed

    # Create a kernel for the moving average
    kernel = np.ones(window_size) / window_size

    # Perform the sliding average on the y_array
    smoothed_y_array = convolve1d(new_rotated_y, kernel, mode='nearest')

    x_points = []
    y_points = []
    # X Range
    minimum = min(new_rotated_x)
    maximum = max(new_rotated_x)
    for interval in range(int(minimum), int(maximum), 10):
        x_points.append(interval)
        max_find = []
        for i in range(len(new_rotated_x)):
            if new_rotated_x[i] > interval and new_rotated_x[i] < interval + 10:
                max_find.append(new_rotated_y[i])
        y_points.append(np.mean(max_find))

    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    # Find the x with largest y value 
    min_y = min(y_points)
    max_y = max(y_points)
    max_x = x_points[np.where(y_points == max_y)]

    # Subtract the x with largest y value from all x values
    x_points = x_points - max_x + adjusted_x

    # Sum the y value with the largest y value
    y_points = y_points - min_y

    # Adjust for Rad 
    y_points = y_points


    # Intervale to remove
    pixeis_point_iron = (902 - 997) / 2

    # Define the condition to keep points outside the interval
    condition = np.logical_and(x_points >= pixeis_point_iron, x_points <= -pixeis_point_iron)

    # Apply the condition to the x and y arrays
    x_points = np.delete(x_points, np.where(condition))
    y_points = np.delete(y_points, np.where(condition))

    # Divide by two * pi to get it in radians
    y_points = y_points / 2 / np.pi

    # Plot the rotated points in a new figure
    plt.plot(x_points, y_points, 'o')
    plt.grid()
    plt.show()

    # Fit a 6th order degree polynomial
    model = np.polyfit(x_points, y_points, 6)

    # Plot the polynomial
    plt.plot(x_points, y_points, 'o', label='Pontos Experimentais')
    x_grid = np.linspace(min(x_points), max(x_points), 100)
    plt.plot(x_grid, np.polyval(model, x_grid), label='Função de Ajuste')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Points vs Polynomial')
    plt.legend()
    plt.show()
    g, f, e, d, c, b, a = model


    r = Rad - 0.5
    u = np.sqrt(Rad ** 2 - r ** 2)
    nr = - 1 / np.pi * (2 * c * u + 4 * e * (u * r**2 + u**3 / 3) + 6 * g * (u*r**4 + 2*u**3 /3 * r**2 + u**5/5)) + 1
    print(f"nr: {nr}")




if __name__ == '__main__':
    temperature_plot(TEMPERATURE_1, line, curve_points, adjusted_x=-10, box_y=(-75, 46), box_x=(2754, 3750), Rad=150)
    temperature_plot(TEMPERATURE_2, line_2, curve_points_2, adjusted_x=47, box_y=(-45, 65), box_x=(2900, 3506), Rad=248)
    temperature_plot(TEMPERATURE_3, line_3, curve_points_3, adjusted_x=-80, box_y=(-52, 67), box_x=(2800, 3289))