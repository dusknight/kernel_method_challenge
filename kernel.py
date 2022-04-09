import numpy as np


class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma
        self.name = 'RBF'

    def kernel(self, X, Y):
        squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
        return np.exp(-0.5*squared_norm/self.sigma**2)


class Linear:
    def __init__(self):
        self.name = 'linear'

    def kernel(self, X, Y):
        return np.einsum('nd,md->nm',X,Y)


# The Generalized Histogram Intersection kernel
class Ghi:
    def __init__(self, b=1.):
        self.name = 'GHI'
        self.beta = b

    def kernel(self, X, Y):
        K = np.zeros((len(X), len(Y)))
        x = abs(X) ** self.beta
        y = abs(Y) ** self.beta
        for i in range(len(X)):
            for j in range(len(Y)):
                K[i, j] = np.minimum(x[i, :], y[j, :]).sum()
        return K


class Poly:
    def __init__(self, degree=5):
        self.degree = degree

    def kernel(self, X, Y, gamma=None, coef0=1):
        if gamma is None:
            gamma = 1.0 / X.shape[1]

        K = np.dot(X, Y.T, dense_output=True)
        K *= gamma
        K += coef0
        K **= self.degree
        return K
