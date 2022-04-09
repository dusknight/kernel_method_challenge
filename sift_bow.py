# import cv2
from sift import SIFT
# from sklearn.cluster import KMeans
from k_means import Kmeans
import pickle
from scipy.spatial.distance import cdist
import os
import numpy as np
from tqdm import tqdm

def extract_sift_features(list_image):

    image_descriptors = []
    # sift = cv2.SIFT_create(sigma=0.4, edgeThreshold=10, contrastThreshold=0.02)  # 0.27
    # sift = cv2.SIFT_create(sigma=0.4, edgeThreshold=10, contrastThreshold=0.001)  # 0.261
    # sift = cv2.SIFT_create(sigma=0.5, edgeThreshold=4, contrastThreshold=0.001)  # 0.284
    # sift = cv2.SIFT_create(sigma=0.5, edgeThreshold=4, contrastThreshold=0.001, nOctaveLayers=3)  # 0.289
    # sift = cv2.SIFT_create(sigma=0.8, edgeThreshold=4, contrastThreshold=0.001)  # 0.273
    # sift = cv2.SIFT_create(sigma=0.8, edgeThreshold=4, contrastThreshold=0.001, nOctaveLayers=3)  # testing
    sift = SIFT(sigma=0.8, edgeThreshold=4, contrastThreshold=0.001, nOctaveLayers=3)  # testing
    for image in tqdm(list_image):
        image = np.uint8(255*(image + 0.5))
        # _, descriptor = sift.detectAndCompute(image, None)
        descriptor = sift.calc_features_for_image(image, unflatten=True)
        image_descriptors.append(descriptor)

    return image_descriptors


def kmean_bow(all_descriptors, num_cluster):
    if not hasattr(all_descriptors, 'shape'):
        all_descriptors = np.array(all_descriptors)
    kmeans = Kmeans(nclusters=num_cluster)
    kmeans.fit(all_descriptors)
    bow_dict = kmeans.mu
    return bow_dict


def create_feature_bow(image_descriptors, BoW, num_cluster):
    X_features = []
    for i in tqdm(range(len(image_descriptors))):
        features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)
            argmin = np.argmin(distance, axis = 1)
            for j in argmin:
                features[j] += 1
        X_features.append(features)
    return X_features


if __name__=="__main__":
    import pandas as pd
    from utils import rgb2gray
    from sklearn.model_selection import train_test_split
    from sklearn import svm

    Xtr = np.array(pd.read_csv('data/Xtr.csv', header=None, sep=',', usecols=range(3072)))[220:325]
    Xte = np.array(pd.read_csv('data/Xte.csv', header=None, sep=',', usecols=range(3072)))[:5]
    Ytr = np.array(pd.read_csv('data/Ytr.csv', sep=',', usecols=[1])).squeeze()[220:325]

    Xall = np.concatenate([Xtr, Xte])
    image_size = (32, 32)
    cell_size = 4

    Xall = rgb2gray(Xall, 1).reshape(len(Xtr)+len(Xte), 32, 32)
    image_desctiptors = extract_sift_features(Xall)

    all_descriptors = []
    for descriptor in image_desctiptors:
        if descriptor is not None:
            for des in descriptor:
                all_descriptors.append(des)

    num_cluster = 100
    BoW = kmean_bow(all_descriptors, num_cluster)
    X_features = create_feature_bow(image_desctiptors, BoW, num_cluster)
    Xtr_sift, Xte_sift = X_features[:5000], X_features[5000:]
    X_train, X_test, y_train, y_test = train_test_split(Xtr_sift, Ytr, test_size=0.2, random_state=20)
    c = 1.5
    clf = svm.SVC(C=c, kernel='rbf')
    clf.fit(X_train, y_train)
    print('Overall accuracy on validation set (1000 samples):')
    print(clf.score(X_test, y_test))