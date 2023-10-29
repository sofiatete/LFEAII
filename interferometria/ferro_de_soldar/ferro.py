

# Indice de refração do ar para temp=0ºC
n2 = 1.00029115
T2 = 273.15 # temp em k

T1 = 373.15 # temp em k

# Formula para obter o n do ar para uma dada temp
n1 = ((n2 - 1) * T2 + T1) / T1 

print('n1 = ', n1)