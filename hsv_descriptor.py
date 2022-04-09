import matplotlib
import numpy as np


def hsv(img, nbin=10, xmin=0, xmax=255, normalize=True):
    """
    Calculate color histogram (HSV)
    :argument
    - img  : RGB in order of HxWxC
    - nbin :
    - xmin : min pixel value (default 0)
    - xmax : min pixel value (default 255)
    - normalize : normalize bins? (default=True)
    :return
    - imhist : 图像的颜色直方图

    """
    ndim = img.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(img / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalize)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist
