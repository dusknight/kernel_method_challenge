import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


class KNeighborsClassifier():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self._X_train = X
        self.classes_, self._y = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        dist_mat = cdist(X, self._X_train)
        neigh_ind = np.argsort(dist_mat, axis=1)[:, :self.n_neighbors]
        return self.classes_[mode(self._y[neigh_ind], axis=1)[0].ravel()]

    def predict_proba(self, X):
        dist_mat = cdist(X, self._X_train)
        neigh_ind = np.argsort(dist_mat, axis=1)[:, :self.n_neighbors]
        proba = np.zeros((X.shape[0], len(self.classes_)))
        pred_labels = self._y[neigh_ind]
        for idx in pred_labels.T:
            proba[np.arange(X.shape[0]), idx] += 1
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        return proba

    def score(self, X, y_true):
        assert len(y_true) == len(X)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)