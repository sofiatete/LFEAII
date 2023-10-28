import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from scipy.ndimage import gaussian_filter

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

print(f"Ronchi 1 shape: {RONCHI_1.shape}")
print(RONCHI_1 / max(RONCHI_1.flatten()))
# Find Maximum Values 
sorted_ronchi_1 = np.sort(RONCHI_1.flatten())
print(sorted_ronchi_1[-100:])
# Find Coordinates of Maximum Values
coordinates = []
for y in range(RONCHI_1.shape[0]):
    for x in range(RONCHI_1.shape[1]):
        if RONCHI_1[y, x] in sorted_ronchi_1[-200:]:
            coordinates.append((x, y))
print(coordinates)

# Plot image with a dot on each maximum value
plt.imshow(RONCHI_1, cmap='gray')
plt.axis('off')
plt.scatter([x[0] for x in coordinates], [x[1] for x in coordinates], c='red')
plt.show()

# Find Clusters of points 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(coordinates)

# Print Clusters Centers
print(kmeans.cluster_centers_)
# Plot Cluster Centers with diferent colors
plt.imshow(RONCHI_1, cmap='gray')
plt.axis('off')
plt.scatter([x[0] for x in kmeans.cluster_centers_], [x[1] for x in kmeans.cluster_centers_], c='red')
plt.show()

for points in kmeans.cluster_centers_:
    print(round(points[0]), round(points[1]))








# --------------------------- Redes TEM --------------------------- #
# ---------------------------- Slide AB --------------------------- #
# ---------------------------- Resolução --------------------------- #

