import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernel import RBF
from svm_model import KernelSVC
from utils import rgb2gray, normalize, flat2img, norm_rgb, plotrgb
from hog_descriptor import hog

from sklearn.model_selection import train_test_split
from sklearn import svm

Xtr = np.array(pd.read_csv('data/Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('data/Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('data/Ytr.csv',sep=',',usecols=[1])).squeeze()

#%%


# convert the RGB images into grayscale?
def rgb2gray(image, gamma=1.):
    r, g, b = image[:, :1024], image[:, 1024:2048], image[:,2048:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     gray[gray<0] = 0
    gray += abs(np.min(gray, axis=1)[:, None])
#     gray = np.maximum(r, g, b)
    return gray**gamma

def normalize(image):
    m = np.mean(image, axis=1)
    sd2 = np.mean((image-m[:,None])**2, axis=1)
    return (image - m[:, None]) / np.sqrt(sd2[:, None])
x = rgb2gray(Xtr, 1.)
# print(x)
plt.imshow(x[900].reshape(32, 32), cmap='gray')
plt.colorbar()
plt.show()
# x = normalize(xx)
# plt.imshow(x[119].reshape(32, 32), cmap='gray')
# plt.colorbar()
# plt.show()