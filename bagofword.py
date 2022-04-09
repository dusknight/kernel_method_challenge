import numpy as np
import pandas as pd
from utils import rgb2gray, normalize, flat2img
from hog_descriptor import hog
from k_means import Kmeans
import cv2
from sklearn.cluster import KMeans

class BagOfWords:
    def __init__(self, nclusters=256):
        self.kmeans = None
        self.nclusters = nclusters

    def fit(self, X):
        X_features = X.reshape(X.shape[0] * X.shape[1], -1)
        self.kmeans = Kmeans(self.nclusters)
        self.kmeans.fit(X_features)

    def predict(self, X):
        assert X.ndim == 3
        X_features = X.reshape(X.shape[0] * X.shape[1], -1)
        X_clustered = self.kmeans.predict(X_features)
        X_clustered = X_clustered.reshape(X.shape[0], X.shape[1])
        ret = np.zeros((X.shape[0], self.nclusters))

        for i, x in enumerate(X_clustered):
            for word in x:
                ret[i, word] += 1
        return ret

class BagOfWords2:
    def __init__(self, nclusters=256):
        self.kmeans = None
        self.nclusters = nclusters
        self.centers = np.zeros((nclusters, 9))

    def fit(self, Xtr_hog):
        batches = Xtr_hog.reshape(-1, 9)
        self.kmeans = KMeans(n_clusters=self.nclusters, max_iter=50).fit(batches)
        self.centers = self.kmeans.cluster_centers_
        print('kmeans done')

    def get_hist(self, Xtr_hog):
        n = Xtr_hog.shape[0]
        xx_hist = np.zeros((n, self.nclusters))

        for l in range(n):
            dd = Xtr_hog[l]
            for d in dd.reshape(-1, 9):
                aa = (abs(self.centers - d) ** 2).sum(1)
                id = np.argmin(aa)
                xx_hist[l, id] += 1
        return xx_hist

#%%
# if __name__ == '__main__':
#
#     Xtr = np.array(pd.read_csv('data/Xtr.csv', header=None, sep=',', usecols=range(3072)))
#     Xte = np.array(pd.read_csv('data/Xte.csv', header=None, sep=',', usecols=range(3072)))
#     Ytr = np.array(pd.read_csv('data/Ytr.csv', sep=',', usecols=[1])).squeeze()
#
#     # Set parameters in hog descriptor extraction
#     image_size = (32, 32)
#     cell_size = 4
#     hog_len = int(9*(image_size[0]/cell_size-1))
#     Xtr_gray = rgb2gray(Xtr, 1.)
#     Xtr_hog = np.zeros((Xtr.shape[0], 1764))
#     # Xtr_hog = np.zeros((Xtr.shape[0], 441))
#
#     for i, image in enumerate(Xtr_gray):
#         Xtr_hog[i] = hog(image.reshape(32, 32), im_size=image_size, cell_size=cell_size, overlap=True)
#     print('feature extract done for training')
#     print(Xtr_hog.shape)
#
# #%%
# # import matplotlib.pyplot as plt
# # plt.imshow(image8[3], cmap='gray')
# # plt.show()
#
#
# #%%
# nb_cluster = 256
# # print(Xtr_hog_batch.shape)
# bow = BagOfWords2(nclusters=nb_cluster)
# bow.fit(Xtr_hog)
# print('kmeans done')
# print(bow.centers.shape)
#
# xx_hist = bow.get_hist(Xtr_hog)
# print('predict done')
# print(xx_hist.shape)
#
#
# #%%
# from sklearn.model_selection import train_test_split
# from sklearn import svm
#
#
# X_train, X_test, y_train, y_test = train_test_split(xx_hist, Ytr, test_size=0.2, random_state=20)
# c = 1.5
# gamma = 1e-3
# clf = svm.SVC(C=c, gamma=gamma, kernel='rbf')
# clf.fit(X_train, y_train)
# print('Overall accuracy on validation set (1000 samples):')
# print(clf.score(X_test, y_test))
#
# #%%
# from sklearn.neighbors import KNeighborsClassifier
#
# neigh = KNeighborsClassifier(n_neighbors=50)
# neigh.fit(X_train, y_train)
# print('Accuracy KNN on validation set (1000 samples):')
# print(neigh.score(X_test, y_test))
