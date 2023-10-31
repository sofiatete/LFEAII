import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

points = []
for i in (150, 300, 400):
    # Indice de refração do ar para temp=0ºC
    n2 = 1.00029115
    T2 = 273.15 # temp em k

    T1 = i + 273.15 # temp em k

    # Formula para obter o n do ar para uma dada temp
    n1 = ((n2 - 1) * T2 + T1) / T1 
    points.append(n1)

print('n1 = ', n1)
print(points)
plt.figure(figsize=(10, 10))
plt.plot([1.57644, 2.338745, 3.095798], points, 'o-')
# Fit a line, y = ax + b
a, b = np.polyfit([1.57644, 2.338745, 3.095798], points, 1)
print(1/a)
print(b)
plt.plot([1.57644, 2.338745, 3.095798], a*np.array([1.57644, 2.338745, 3.095798]) + b)
plt.show()


