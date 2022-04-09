import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import one_hot_encoding

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def kernel_ridge_regression(K, Y, W=None, lbda=1.0):
    n = len(K)

    if W is None:
        W = np.ones(n)

    Wh = np.sqrt(np.diag(W))
    L = Wh @ K @ Wh + n * lbda * np.eye(n)
    alpha = Wh @ np.linalg.solve(L, Wh @ Y)

    return alpha


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, iter_max=20, tolerance=1E-6, lbda=1.0, num_class=10):

        self.lbda = lbda/2
        self.tolerance = tolerance
        self.iter_max = iter_max
        self.num_class = num_class

    @property
    def _pairwise(self):
        return True

    def fit(self, K, y):
        alpha_t = np.zeros(y.size)
        y = one_hot_encoding(y)
        for iter in range(self.iter_max):
            for j in range(self.num_class-1):
                m = K @ alpha_t[:, j]
                p_j = - sigmoid(- y[:, j] * m)
                c_j = -y[:, j] + p_j + self.lbda * alpha_t[:, j]
                g_j = K @ c_j
                W_t = sigmoid(-m) * sigmoid(-m)

                z_t = m - P_t * y / W_t

                alpha_p = kernel_ridge_regression(K, z_t, W_t, lbda=self.lbda)

                delta = np.linalg.norm(alpha_p - alpha_t)
                alpha_t = alpha_p

                if delta < self.tolerance:
                    break

        self.alpha_ = alpha_t
        self.fitted_ = True
        self.K_fit_ = K

        return self

    def predict(self, K):
        return to_binary(np.sign(K @ self.alpha_))

