import numpy as np
import dippykit as dip


def rgb2gray(rgb):
    r, g, b = rgb[0][:][:], rgb[1][:][:], rgb[2][:][:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def hist(image):
    dim = image.shape
    x = np.reshape(image, (dim[0] * dim[1], 1))
    dip.hist(x)
    dip.grid()
    dip.title('Histogram')
    dip.xlabel('Pixel value')
    dip.ylabel('Pixel frequency')


def add_contrast(image):
    L = 0.02  # Controls the slope of the sigmoid

    dim = image.shape
    height = dim[0]
    width = dim[1]

    output = image
    for i in range(0, height):
        for j in range(0, width):
            x = output[i][j]
            output[i][j] = x * (1.0 / 1 + np.exp(-L * (x - 127)))

    return output
