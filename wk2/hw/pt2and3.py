import matplotlib.pyplot as plt
import numpy as np

array = range(5, 32) # array of 0-30
for x in array: print(x)  # list it
threed = np.reshape(array,(3,3,3)) #3dit
for x in threed: print(x) #list it

array2 = [20, 18, 16, 14, 12, 10, 8, 6, 4] #new one
for x in array2: print(x) #list it
twod = np.reshape(array2,(3,3))  #2dit
for x in twod: print(x) #list it
