import matplotlib.pyplot as plt
import numpy as np

def upsample(input, output, stride=4):
    img = plt.imread(input)
    nrows, ncols = img.shape[0], img.shape[1]
    nfilters = img.shape[2]

    buffer = np.zeros((stride * nrows, stride * ncols, nfilters), dtype=int)

    s = stride
    for row in range(nrows):
        for col in range(ncols):
            for c in range(nfilters):
                buffer[row*s:(row+1)*s, col*s:(col+1)*s, c] = img[row, col, c]

    plt.imsave(output, buffer.astype(np.uint8))
    # plt.imshow(buffer)

upsample("notadoctor.jpg", "notadoctorx4.jpg")
upsample("notadoctor.jpg", "notadoctorx2.jpg", stride=2)
