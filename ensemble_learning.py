import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kernel import Ghi, RBF, Poly
from utils import rgb2gray, normalize, flat2img
from hog_descriptor import hog
from hsv_descriptor import hsv

from sklearn.model_selection import train_test_split
from svm_model import OneOverOneSVC as SVC
from LBP import Lbp_hist
from bagofword import BagOfWords2
import sift_bow
from scipy.stats import mode

Xtr = np.array(pd.read_csv('data/Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('data/Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('data/Ytr.csv',sep=',', usecols=[1])).squeeze()

# Set parameters in hog descriptor extraction
image_size = (32, 32)
cell_size = 4
hsv_bins = 100

Xtr_gray = rgb2gray(Xtr, 1)
Xtr_hog = np.zeros((Xtr.shape[0], 1764))
Xtr_lbp = np.zeros((Xtr.shape[0], 144))
Xtr_hsv = np.zeros((len(Xtr), hsv_bins))

# Get HSV feature
for i, image in enumerate(Xtr):
    Xtr_hsv[i, :] = hsv(flat2img(image), nbin=hsv_bins, xmin=-1, xmax=1, normalize=True)

# Get HOG and LBP feature
for i, image in enumerate(Xtr_gray):
    Xtr_hog[i] = hog(image.reshape(32, 32), im_size=image_size, cell_size=cell_size, clip=None, overlap=True)
    Xtr_lbp[i] = Lbp_hist(image.reshape(32, 32), cells=4, mode='uniform', n=9)

# BOW + HOG
nb_cluster = 256
bow = BagOfWords2(nclusters=nb_cluster)
bow.fit(Xtr_hog)
bow_hog = bow.get_hist(Xtr_hog)
print('bow hog done')

# BOW + SIFT
image_desctiptors = sift_bow.extract_sift_features(Xtr_gray.reshape(-1, 32, 32))
all_descriptors = []
for descriptor in image_desctiptors:
    if descriptor is not None:
        for des in descriptor:
            all_descriptors.append(des)
print('number of words ')
print(len(all_descriptors))
num_cluster = 256
bow_S = sift_bow.kmean_bow(all_descriptors, num_cluster)
bow_sift = sift_bow.create_feature_bow(image_desctiptors, bow_S, num_cluster)
print('bow sift done')

Xtr_hog_lpb = np.hstack([Xtr_hog, Xtr_lbp])
Xtr_all = np.hstack([Xtr_hog_lpb, Xtr_hsv])
print('feature extract done for training')

# scaler = StandardScaler()
# Xtr_feat = scaler.fit_transform(Xtr_feat)
# print(Xtr_feat.shape)


# extract hog descriptor in test dataset
Xte_gray = rgb2gray(Xte, 1.)
Xte_hog = np.zeros((Xte.shape[0], 1764))
Xte_lbp = np.zeros((Xte.shape[0], 144))
Xte_hsv = np.zeros((len(Xte), hsv_bins))

for i, image in enumerate(Xte_gray):
    Xte_hog[i] = hog(image.reshape(32, 32), im_size=image_size, cell_size=cell_size)
    Xte_lbp[i] = Lbp_hist(image.reshape(32, 32), cells=4, mode='uniform', n=9)

for i, image in enumerate(Xte):
    Xte_hsv[i, :] = hsv(flat2img(image), nbin=hsv_bins, xmin=-1, xmax=1, normalize=True)

Xte_hog_lbp = np.hstack([Xte_hog, Xte_lbp])
Xte_all = np.hstack([Xte_hog_lbp, Xte_hsv])

image_desctiptors2 = sift_bow.extract_sift_features(Xte_gray.reshape(-1, 32, 32))
Xte_bow_sift = sift_bow.create_feature_bow(image_desctiptors2, bow_S, num_cluster)
Xte_bow_hog = bow.get_hist(Xte_hog)

print('feature extract done for test')

#%% Cluster 1 HOF + SVM (rbf)
X_train, X_test, y_train, y_test = train_test_split(Xtr_hog_lpb, Ytr, test_size=0.2, random_state=20)
# X_train, X_test, y_train, y_test = train_test_split(Xtr_all, Ytr, test_size=0.2, random_state=20)
print(X_train.shape)

c = 10.5
gamma = 1e-3
k2 = RBF(3.1)
clf = SVC(C=c, kernel=k2.kernel, num_class=10)
clf.fit(X_train, y_train)
# print('Overall accuracy on validation set (1000 samples):')
# print(clf.score(X_test, y_test))


#%% Cluster 2 HOG + SVM (poly degree 5)
c = 10.
k = Poly(degree=6)
clf2 = SVC(C=c, epsilon=gamma, kernel=k.kernel, num_class=10)
clf2.fit(X_train, y_train)
print('Overall accuracy on validation set (1000 samples):')
print(clf2.score(X_test, y_test))


#%% Cluster 3 KNN
from kNN import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X_train, y_train)
print('Overall accuracy on validation set (1000 samples):')
print(neigh.score(X_test, y_test))


#%% Cluster 4 BOW + HOG
X_train, X_test_hog, y_train, y_test = train_test_split(bow_hog, Ytr, test_size=0.2, random_state=20)
c = 1.
gamma = 1e-3
n_features = X_train.shape[1]
k = RBF(sigma=np.sqrt(n_features * X_train.var()))
clf4 = SVC(C=c,  kernel=k.kernel, num_class=10)
clf4.fit(X_train, y_train)
print('Overall accuracy on validation set (1000 samples):')
print(clf4.score(X_test_hog, y_test))


#%% CLuster 5 BOW + sift
X_train, X_test_sift, y_train, y_test = train_test_split(bow_sift, Ytr, test_size=0.2, random_state=20)
c = 1.
n_features = X_train.shape[1]
k = RBF(sigma=np.sqrt(n_features * X_train.var()))
clf5 = SVC(C=c, kernel=k.kernel, num_class=10)
clf5.fit(X_train, y_train)
print('Overall accuracy on validation set (1000 samples):')
print(clf5.score(X_test_sift, y_test))


#%%
# y1 = clf.predict(X_test)
# y2 = clf2.predict(X_test)
# y3 = neigh.predict(X_test)
# y4 = clf4.predict(X_test_hog)
# y5 = clf5.predict(X_test_sift)
#
# ensemble = np.array([y1, y2, y3, y4, y5])
# print(ensemble.shape)

# from scipy.stats import mode
#
# res = mode(ensemble, 0)[0][0]
# print('final score:')
# print(np.mean(res == y_test))

#%%
clf.fit(Xtr_hog_lpb, Ytr)
clf2.fit(Xtr_hog_lpb, Ytr)
neigh.fit(Xtr_hog_lpb, Ytr)
clf4.fit(bow_hog, Ytr)
clf5.fit(bow_sift, Ytr)

y1 = clf.predict(Xte_hog_lbp)
y2 = clf2.predict(Xte_hog_lbp)
y3 = neigh.predict(Xte_hog_lbp)
y4 = clf4.predict(Xte_bow_hog)
y5 = clf5.predict(Xte_bow_sift)
ensemble = np.array([y1, y2, y3, y4, y5])
print(ensemble.shape)

res = mode(ensemble, 0)[0][0]
print(str(res))
Yte = {'Prediction': res}
dataframe = pd.DataFrame(Yte)
dataframe.index += 1
import os

os.makedirs('./data', exist_ok=True)
dataframe.to_csv('./data/Yte_pred_ensemble.csv', index_label='Id')




