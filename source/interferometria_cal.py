import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter
#from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
import itertools
import string
import os
import math
from math import sqrt



# Set up paths
IMAGES_PATH = Path('../data')
IMAGES_PATH.mkdir(exist_ok=True,parents=True)


def calibrate_image(name: str, intensity_limit: int = 3500, *args, **kwargs) -> list:
    """
    Calibrate the image
    """

    IMAGE = name + '.pgm'

    img = Image.open(IMAGES_PATH/IMAGE)
    img_arr = np.array(img)
    # print(f"Image shape: {img_arr.shape}")

    # Plot image
    plt.imshow(img_arr, cmap='gray')
    # plot axis
    plt.xlabel('$x$ (pixels)')
    plt.ylabel('$y$ (pixels)')

    plt.title(f'Calibration {name} ')
    plt.legend()
    plt.savefig(f'../graphs/{name}_image.png', dpi=400)
    plt.show()

    center = (img_arr.shape[1]//2 - 30, img_arr.shape[0]//2 + 10)

    # lets go through col 400 and find the first point with intensity different from 0
    # and the last point with intensity different from 0
    point_11, point_12, point_21, point_22 = 0, 0, 0, 0
    col_1 = 600
    col_2 = 800
    limit = intensity_limit
    for i in range(img_arr.shape[0]):
        if img_arr[i, col_1] >= limit:
            point_11 = (col_1, i)
            break
    for i in range(img_arr.shape[0]-1, 0, -1):
        if img_arr[i, col_1] >= limit:
            point_12 = (col_1, i)
            break
    for i in range(img_arr.shape[0]):
        if img_arr[i, col_2] >= limit:
            point_21 = (col_2, i)
            break
    for i in range(img_arr.shape[0]-1, 0, -1):
        if img_arr[i, col_2] >= limit:
            point_22 = (col_2, i)
            break
    print('points: ', point_11, point_12, point_21, point_22)
    points_x = [point_11[0], point_12[0], point_21[0], point_22[0]]
    points_y = [point_11[1], point_12[1], point_21[1], point_22[1]]

    # slope of the line that goes through 11 and 21: y = slope_1*x + b_1
    slope_1 = (point_21[1] - point_11[1])/(point_21[0] - point_11[0])
    b_1 = point_11[1] - slope_1*point_11[0]
    # print('y_1 = ' + str(slope_1) + '*x + ' + str(b_1))
    # slope of the line that goes through 12 and 22: y = slope_2*x + b_2
    slope_2 = (point_22[1] - point_12[1])/(point_22[0] - point_12[0])
    b_2 = point_12[1] - slope_2*point_12[0]
    # print('y_2 = ' + str(slope_2) + '*x + ' + str(b_2))

    # slope perpendicular to slope_1 and goes through the middle point of 11 and 21
    middle_point_1 = ((point_11[0] + point_21[0])//2, (point_11[1] + point_21[1])//2)
    slope_perp_1 = -1/slope_1
    b_perp_1 = middle_point_1[1] - slope_perp_1*middle_point_1[0]

    # slope perpendicular to slope_2 and goes through the middle point of 12 and 22
    middle_point_2 = ((point_12[0] + point_22[0])//2, (point_12[1] + point_22[1])//2)
    slope_perp_2 = -1/slope_2
    b_perp_2 = middle_point_2[1] - slope_perp_2*middle_point_2[0]

    # point of intersection of the lines perp and 1
    x_1 = (b_perp_1 - b_1)/(slope_1 - slope_perp_1)
    y_1 = slope_1*x_1 + b_1
    # print('point of intersection of the lines perp and 1: ', x_1, y_1)
    # point of intersection of the lines perp and 2
    x_2 = (b_perp_2 - b_2)/(slope_2 - slope_perp_2)
    y_2 = slope_2*x_2 + b_2
    # print('point of intersection of the lines perp and 2: ', x_2, y_2)
    intersection_x = [x_1, x_2]
    intersection_y = [y_1, y_2]

    # distance in pixels between the points of intersection
    distance = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
    # print('distance in pixels between the points of intersection: ', distance)

    # error in the distance
    error = round(abs((x_1 - x_2) / sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)) + abs((x_2 - x_1) / sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)) + abs((y_1 - y_2) / sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)) + abs((y_2 - y_1) / sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)),1)

    # Plot image
    plt.imshow(img_arr, cmap='gray')
    # plot axis
    plt.xlabel('$x$ (pixels)')
    plt.ylabel('$y$ (pixels)')

    plt.title(f'Calibration {name} ')
    plt.legend()
    plt.savefig(f'../graphs/{name}_image.png', dpi=400)

    # plot the points used to find the lines
    plt.scatter(points_x, points_y, color='red', s=10, label='Points')

    # plot the lines
    x_values_1 = np.linspace(point_11[0], point_21[0], 100)
    x_values_2 = np.linspace(point_12[0], point_22[0], 100)
    x_values_perp_1 = np.linspace(1.1*col_1, 0.9*col_2, 100)
    x_values_perp_2 = np.linspace(1.1*col_1+15, 0.9*col_2+15, 100)

    y_values_1 = slope_1*x_values_1 + b_1
    y_values_2 = slope_2*x_values_2 + b_2
    y_values_perp_1 = slope_perp_1*x_values_perp_1 + b_perp_1
    y_values_perp_2 = slope_perp_2*x_values_perp_2 + b_perp_2

    plt.plot(x_values_1, y_values_1, color='orange', label='Line 1')
    plt.plot(x_values_2, y_values_2, color='green', label='Line 2')
    plt.plot(x_values_perp_1, y_values_perp_1, color='blue', label='Line 1 perp')
    plt.plot(x_values_perp_2, y_values_perp_2, color='red', label='Line 2 perp')

    # plot the intersection points
    plt.scatter(intersection_x, intersection_y, color='purple', s=40, label='Intersection', zorder=10)
    # add labels A and B
    plt.text(intersection_x[0] - 50, intersection_y[0] - 40, 'A', color='white', fontsize=7, ha='center', va='bottom')
    plt.text(intersection_x[1] + 50, intersection_y[1] - 40, 'B', color='white', fontsize=7, ha='center', va='bottom')

    # add the distance between the points in a label in the right below corner
    plt.text(0.8, 0.05, 'Distance AB: ' + str(distance.round(1)) + ' ± ' + str(error) + ' pixels', color='white', fontsize=7, ha='center', va='bottom', transform=plt.gca().transAxes)


    # plot title
    plt.title(f'Calibration {name} ')
    plt.legend()
    plt.savefig(f'../graphs/{name}_calibration.png', dpi=400)
    plt.show()
    plt.close()

    return [distance, error]


# calibração interferometro
calib_intf = calibrate_image('craveira2mm_aula4_2', intensity_limit=27500)[0]




