import numpy as np
import matplotlib.pyplot as plt

def flat2img(flat):
    """
    convert flat format image to HWC format
    :param flat: flatten format of image (read from files)
    :return: image in H W C format
    """
    r, g, b = flat[:1024], flat[1024:2048], flat[2048:]
    r, g, b = r.reshape(32, 32), g.reshape(32, 32), b.reshape(32, 32)
    return np.stack([r, g, b]).transpose((1, 2, 0))  # CHW to HWC


# convert rgb images to grayscale
def rgb2gray(image, gamma=1.):
    r, g, b = image[:, :1024], image[:, 1024:2048], image[:,2048:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray += abs(np.min(gray, axis=1)[:, None])
#     gray = np.maximum(r, g, b)
    return gray**gamma


def normalize(image):
    m = np.mean(image, axis=1)
    sd2 = np.mean((image-m[:,None])**2, axis=1)
    return (image - m[:, None]) / np.sqrt(sd2[:, None])


import cv2 as cv
def norm_rgb(rgb):
    return cv.normalize(rgb, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

def plotrgb(flat):
    rgb = flat2img(flat)
    rgb = norm_rgb(rgb)
    plt.imshow(rgb)
    plt.show()
    return


def one_hot_encoding(a, num_class=10):
    one_hot_vector = np.zeros((a.size[0], num_class))
    one_hot_vector[np.arange(a.size[0]), a] = 1
    return one_hot_vector
