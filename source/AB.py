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
import os
import math
from math import sqrt

# Set up paths
IMAGES_PATH = Path('../data')
IMAGES_PATH.mkdir(exist_ok=True,parents=True)

def analyse_image(name: str, points: int = 2000, *args, **kwargs) -> np.ndarray:
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
    plt.savefig(f'../graphs/AB/{name}_image.png', dpi=400)
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
    plt.savefig(f'../graphs/AB/{name}_clusters.png', dpi=400)
    plt.close()

    # return the coordinates of the clusters centroids
    return kmeans.cluster_centers_

def calibrate_image(name: str, intensity_limit: int = 3500, *args, **kwargs) -> list:
    """
    Calibrate the image
    """

    IMAGE = name + '.pgm'

    img = Image.open(IMAGES_PATH/IMAGE)
    img_arr = np.array(img)
    # print(f"Image shape: {img_arr.shape}")

    center = (img_arr.shape[1]//2 - 30, img_arr.shape[0]//2 + 10)

    # lets go through line 400 and find the first point with intensity different from 0
    # and the last point with intensity different from 0
    point_11, point_12, point_21, point_22 = 0, 0, 0, 0
    line_1 = 400
    line_2 = 1000
    limit = intensity_limit
    for i in range(img_arr.shape[1]):
        if img_arr[line_1, i] >= limit:
            point_11 = (i, line_1)
            break
    for i in range(img_arr.shape[1]-1, 0, -1):
        if img_arr[line_1, i] >= limit:
            point_21 = (i, line_1)
            break
    for i in range(img_arr.shape[0]):
        if img_arr[line_2, i] >= limit:
            point_12 = (i, line_2)
            break
    for i in range(img_arr.shape[0]-1, 0, -1):
        if img_arr[line_2, i] >= limit:
            point_22 =  (i, line_2)
            break
    # print('points: ', point_11, point_12, point_21, point_22)
    points_x = [point_11[0], point_12[0], point_21[0], point_22[0]]
    points_y = [point_11[1], point_12[1], point_21[1], point_22[1]]

    # slope of the line that goes through 11 and 12: y = slope_1*x + b_1
    slope_1 = (point_12[1] - point_11[1])/(point_12[0] - point_11[0])
    b_1 = point_11[1] - slope_1*point_11[0]
    # print('y_1 = ' + str(slope_1) + '*x + ' + str(b_1))
    # slope of the line that goes through 21 and 22: y = slope_2*x + b_2
    slope_2 = (point_22[1] - point_21[1])/(point_22[0] - point_21[0])
    b_2 = point_21[1] - slope_2*point_21[0]
    # print('y_2 = ' + str(slope_2) + '*x + ' + str(b_2))

    # slope perpendicular to slope_1 and goes through the center
    slope_perp_1 = -1/slope_1
    b_perp_1 = center[1] - slope_perp_1*center[0]

    # slope perpendicular to slope_2 and goes through the center
    slope_perp_2 = -1/slope_2
    b_perp_2 = center[1] - slope_perp_2*center[0]

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
    plt.savefig(f'../graphs/AB/{name}_image.png', dpi=400)

    # plot the points used to find the lines
    plt.scatter(points_x, points_y, color='red', s=10, label='Points')

    # plot the lines
    x_values_1 = np.linspace(point_11[0], point_12[0], 100)
    x_values_2 = np.linspace(point_21[0], point_22[0], 100)
    x_values_perp_1 = np.linspace(250, 1750, 100)
    x_values_perp_2 = np.linspace(250, 1750, 100)

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
    plt.savefig(f'../graphs/AB/{name}_calibration.png', dpi=400)
    # plt.show()
    plt.close()

    return [distance, error]

def calibrator(meters: list, pixels: list) -> list:

    # plot the points
    plt.scatter(pixels, meters, color='orange', s=40, label='Points')

    # linear regression
    m, b = np.polyfit(pixels, meters, 1)
    # Residuals (difference between observed and predicted values)
    residuals = meters - (m * np.array(pixels) + b)

    # Number of observations
    n = len(pixels)

    # Standard deviation of the residuals
    residual_std = np.sqrt(np.sum(residuals**2) / (n - 2))  # n - 2 for degrees of freedom (2 parameters: slope and intercept)

    # Standard error of the regression slope (m) and intercept (b)
    std_error_slope = residual_std / np.sqrt(np.sum((pixels - np.mean(pixels))**2))
    std_error_intercept = residual_std * np.sqrt(1/n + (np.mean(pixels)**2) / np.sum((pixels - np.mean(pixels))**2))

    print("Standard error of the slope (m):", std_error_slope)
    print("Standard error of the intercept (b):", std_error_intercept)
    # print('m: ', m)
    # print('b: ', b)

    # plot the line
    x_values = np.linspace(0, 1.05*max(pixels), 100)
    y_values = m*x_values + b
    plt.plot(x_values, y_values, color='blue', label=f'Linear Regression \n y = mx + b \n m = {m:.3e} ± {std_error_slope:.3e}\n b = {b:.3e} ± {std_error_intercept:.3e}')
    # plt.plot(x_values, y_values, color='blue', label='Linear Regression \n y = mx + b \n m = ' + str(m.round(9)) + ' ± ' + str(std_error_slope.round(9)) + '\n b = ' + str(b.round(9)) + ' ± ' + str(std_error_intercept.round(9)))


    plt.title('Calibration')
    plt.xlabel('Distance (pixels)')
    plt.ylabel('Distance (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../graphs/AB/regr_calibration.png', dpi=400)
    # plt.show()
    plt.close()

    return [m, b, std_error_slope, std_error_intercept] # = [m, b, error_m, error_b]

def save_image_png(names: np.array, path: str, *args, **kwargs) -> None:
    """
    Save the images in png format
    """
    for name in names:
        IMAGE = name + '.pgm'
        img = Image.open(IMAGES_PATH/IMAGE)
        img_arr = np.array(img)
        plt.imshow(img_arr, cmap='gray')
        plt.xlabel('$x$ (pixels)')
        plt.ylabel('$y$ (pixels)')
        plt.title(name)
        plt.savefig(f"..{path}/{name}.png", dpi=400)
        plt.clf()
    plt.close()

    return None

if __name__ == '__main__':

    coordinates_1 = analyse_image('AB1_5aula', points = 2000)
    coordinates_2 = analyse_image('AB2_5aula', points = 2000)
    coordinates_3 = analyse_image('AB3_5aula', points = 10)
    coordinates_4 = analyse_image('AB4_5aula', points = 2000)
    
    # mean of the coordinates of the clusters centroids
    mean = np.mean([coordinates_1, coordinates_2, coordinates_3, coordinates_4], axis=0)
    print('\n Mean of the coordinates of the clusters centroids:')
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        print(label + ': (' + str(mean[i, 0]) + ', ' + str(mean[i, 1]) + ')   ' + 'Intensity: ' + str(mean[i, 2]))
    

    # Distance of each point considering A and D as being located in the x axis and B and C in the y axis
    center = np.array([934, 734])
    # distance in pixels between the points
    distance_ABCD = [abs(mean[0, 0] - center[0]), abs(mean[1, 1] - center[1]), abs(mean[2, 1] - center[1]), abs(mean[3, 0] - center[0])]
    print('\n Distance of each point considering A and D as being located in the x axis and B and C in the y axis:')
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        print(label + ': ' + str(distance_ABCD[i]) +  ' ± ' + str(1) + ' pixels')

    # Calibration
    calibration_2mm = calibrate_image('craveira2mm_AB_5aula', intensity_limit=3500)[0]
    calibration_3mm = calibrate_image('craveira3mm_AB_5aula', intensity_limit=6500)[0]
    calibration_4mm = calibrate_image('craveira4mm_AB_5aula', intensity_limit=5500)[0]
    calibration_4mm2 = calibrate_image('craveira4mm2_AB_5aula', intensity_limit=3500)[0]

    pixels = [calibration_2mm, calibration_3mm, calibration_4mm, calibration_4mm2]
    mm = [2, 3, 4, 4]
    m = [0.002, 0.003, 0.004, 0.004]

    regression = calibrator(m, pixels) # = [m, b, error_m, error_b]
    os.system('../graphs/AB/calibration.txt')
    with open('../graphs/AB/calibration.txt', 'w') as f:
        f.write('m: ' + str(regression[0]) + '\n')
        f.write('b: ' + str(regression[1]) + '\n')
        f.write('y = ' + str(regression[0]) + 'x + ' + str(regression[1]) + '\n y in m and x in pixels')
        f.close()
    
    
    print('\n Linear Regression: y = ' + str(regression[0].round(9)) + 'x + ' + str(regression[1].round(9)))
    print(f'm = {regression[0]:.3e} ± {regression[2]:.3e}')
    print(f'b = {regression[1]:.3e} ± {regression[3]:.3e}')
    calibrated_value = lambda x: abs(float(regression[0]) * float(x) + float(regression[1]))
    error_calib_value = lambda x: abs(x) * regression[2] + abs(regression[0]) + regression[3]

    # print all not calibrated and then calibrated values for distances A, B, C and D
    print('\n Distances A, B, C and D Calibrated:')
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        # print(label + f': {distance_ABCD[i]:.0f} ± {1:.0f} pixels' + f'   calibrated: {calibrated_value(distance_ABCD[i]):.3e} ± {error_calib_value(distance_ABCD[i]):.3e} m')
        # in milimeters
        print(label + f': {distance_ABCD[i]:.0f} ± {1:.0f} pixels' + f'   calibrated: {(calibrated_value(distance_ABCD[i])*10**3):.3f} ± {(error_calib_value(distance_ABCD[i])*10**3):.3f} mm')

    f=0.2475 # focal length of the lens in m
    wavelength=633*10**(-9) # wavelength of the laser in m

    # array with the calibrated values of the distances in m
    distance_ABCD_m = [(calibrated_value(distance), error_calib_value(distance)) for distance in distance_ABCD]


    # compute the spatial frequencies
    spacial_frequency_x_1 = distance_ABCD_m[0][0]/(f*wavelength) # distance of A to the center
    error_spacial_frequency_x_1 = distance_ABCD_m[0][1]/(f*wavelength)/spacial_frequency_x_1 * 100
    spacial_frequency_x_2 = distance_ABCD_m[3][0]/(f*wavelength) # distance of D to the center
    error_spacial_frequency_x_2 = distance_ABCD_m[3][1]/(f*wavelength)/spacial_frequency_x_2 * 100
    spacial_frequency_y_1 = distance_ABCD_m[1][0]/(f*wavelength) # distance of B to the center
    error_spacial_frequency_y_1 = distance_ABCD_m[1][1]/(f*wavelength)/spacial_frequency_y_1 * 100
    spacial_frequency_y_2 = distance_ABCD_m[2][0]/(f*wavelength) # distance of C to the center
    error_spacial_frequency_y_2 = distance_ABCD_m[2][1]/(f*wavelength)/spacial_frequency_y_2 * 100

    # the error of the spatial frequency is the error of the distance divided by f*wavelength
    print('\n Spatial Frequencies:')
    print('Spatial Frequency along X for point A: ' + f'{spacial_frequency_x_1:.2f} m^-1' + ' ± ' + f'{error_spacial_frequency_x_1:.2f} %')
    print('Spatial Frequency along X for point D: ' + f'{spacial_frequency_x_2:.2f} m^-1' + ' ± ' + f'{error_spacial_frequency_x_2:.2f} %')
    print('Spatial Frequency along Y for point B: ' + f'{spacial_frequency_y_1:.2f} m^-1' + ' ± ' + f'{error_spacial_frequency_y_1:.2f} %')
    print('Spatial Frequency along Y for point C: ' + f'{spacial_frequency_y_2:.2f} m^-1' + ' ± ' + f'{error_spacial_frequency_y_2:.2f} %')

    frequancy_x = (spacial_frequency_x_1 + spacial_frequency_x_2)/2
    error_frequancy_x = (error_spacial_frequency_x_1 + error_spacial_frequency_x_2)/2
    frequancy_y = (spacial_frequency_y_1 + spacial_frequency_y_2)/2
    error_frequancy_y = (error_spacial_frequency_y_1 + error_spacial_frequency_y_2)/2
    print('Spatial Frequency along X: ' f'{(frequancy_x):.2f} m^-1' + ' ± ' + f'{(error_frequancy_x):.2f} %')
    print('Spatial Frequency along Y: ' f'{(frequancy_y):.2f} m^-1' + ' ± ' + f'{(error_frequancy_y):.2f} %')

    print('\n Separation of the fringes:')
    separation_letterB = 1/(frequancy_y) 
    error_separation_B = error_frequancy_y/(frequancy_y**2)
    error_separation_B_percentage = error_separation_B/separation_letterB*100
    separation_letterA = 1/(frequancy_x)
    error_separation_A = error_frequancy_x/(frequancy_x**2)
    error_separation_A_percentage = error_separation_A/separation_letterA*100

    print('Separation of the fringes along X: ' + f'{(separation_letterA):.3e} ± {(error_separation_A):.3e} m')
    print('Separation of the fringes along Y: ' + f'{(separation_letterB):.3e} ± {(error_separation_B):.3e} m')

    print('\n Separation of the fringes in micrometers:')
    print('Separation of the fringes along X: ' + f'{(separation_letterA*10**6):.5f} ± {(error_separation_A*10**6):.5f} μm')
    print('Separation of the fringes along Y: ' + f'{(separation_letterB*10**6):.5f} ± {(error_separation_B*10**6):.5f} μm')

    # separation of the fringes with error in percentage
    print('\n Separation of the fringes with error in percentage:')
    print('Separation of the fringes along X: ' + f'{(separation_letterA*10**6):.5f} μm ± {(error_separation_A_percentage):.5f} %')
    print('Separation of the fringes along Y: ' + f'{(separation_letterB*10**6):.5f} μm ± {(error_separation_B_percentage):.5f} %')


    # Extra with filters on the fourier plane
    images_names = np.array(['aula5_moire', 'extra1_A', 'extra1_A2', 'extra1_B', 'extra1_B2', 'extra1_TFB', 'slideAB_aula5'])
    save_image_png(images_names, '/graphs/AB/Extra')

    

    

    





