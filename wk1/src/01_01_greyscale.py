import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("duckhunt.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

greyed = np.zeros((nrows, ncols, nchannels), dtype=int)

for row in range(nrows):
    for col in range(ncols):
        avg = sum(img[row,col,:]) / 3
        greyed[row,col,:] = avg


plt.imsave("out/grey-dog.png", greyed)
