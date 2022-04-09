import numpy as np
from scipy import optimize
# from sklearn.svm import SVC
""" SVC and multiSVC """


# class KernelSVC:
#
#     def __init__(self, C, kernel, epsilon = 1e-3):
#         self.type = 'non-linear'
#         self.C = C
#         self.kernel = kernel
#         self.alpha = None
#         self.support = None
#         self.epsilon = epsilon
#         self.norm_f = None
#
#     def fit(self, X, y):
#         N = len(y)
#         hXX = self.kernel(X, X)
#         G = np.einsum('ij,i,j->ij', hXX, y, y)
#         A = np.vstack((-np.eye(N), np.eye(N)))
#         b = np.hstack((np.zeros(N), self.C * np.ones(N)))
#
#         # Lagrange dual problem
#         def loss(alpha):
#             return -alpha.sum() + 0.5 * alpha.dot(alpha.dot(G))
#
#         # Partial derivate of Ld on alpha
#         def grad_loss(alpha):
#             return -np.ones_like(alpha) + alpha.dot(G)
#
#         # Constraints on alpha of the shape :
#         fun_eq = lambda alpha:  np.dot(alpha, y)
#         jac_eq = lambda alpha:   y
#         fun_ineq = lambda alpha:  b - np.dot(A, alpha)
#         jac_ineq = lambda alpha:  -A
#
#         constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
#                        {'type': 'ineq',
#                         'fun': fun_ineq ,
#                         'jac': jac_ineq})
#
#         optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
#                                    x0=np.ones(N),
#                                    method='SLSQP',
#                                    jac=lambda alpha: grad_loss(alpha),
#                                    constraints=constraints)
#         self.alpha = optRes.x
#
#         ## Assign the required attributes
#
#         margin_pointsIndices = (self.alpha > self.epsilon)
#         boundaryIndices = (self.alpha > self.epsilon) * (self. C- self.alpha > self.epsilon )
#
#         self.support = X[boundaryIndices]
#
#         self.margin_points = X[margin_pointsIndices]
#         self.margin_points_AlphaY = y[margin_pointsIndices] * self.alpha[margin_pointsIndices]
#         if max(boundaryIndices) == False:
#             print("Warning: No boundary points found!")
#             self.b = 0
#         else:
#             self.b = y[boundaryIndices][0] - self.separating_function(np.expand_dims(X[boundaryIndices][0], axis=0))
#         K_margin_points = self.kernel(self.margin_points, self.margin_points)
#         self.norm_f = np.einsum('i,ij,j->', self.margin_points_AlphaY , K_margin_points, self.margin_points_AlphaY)
#
#     def separating_function(self, x):
#         x1 = self.kernel(self.margin_points, x)
#         return np.einsum('ij,i->j', x1, self.margin_points_AlphaY)
#
#     def predict_proba(self, X):
#         return self.separating_function(X) + self.b
#
#     def predict(self, X):
#         """ Predict y values in {-1, 1} """
#         d = self.separating_function(X)
#         return 2 * (d + self.b > 0) - 1
class KernelSVC:

    def __init__(self, C, kernel, epsilon=1e-3):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None

    def fit(self, X, y):
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        K = self.kernel(X, X)
        y_diag = np.diag(y)
        yKy = y_diag @ K @ y_diag

        # Lagrange dual problem
        def loss(alpha):
            target = - 0.5 * alpha.T @ yKy @ alpha + np.sum(alpha)
            return - target  # '''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            grad = - yKy @ alpha + np.ones(N)
            return - grad  # '''----------------partial derivative of the dual loss wrt alpha-----------------'''

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        fun_eq = lambda alpha: np.dot(y,
                                      alpha)  # '''----------------function defining the equality constraint------------------'''
        jac_eq = lambda alpha: np.array(
            y)  # '''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha: np.array(
            self.C - alpha)  # '''---------------function defining the ineequality constraint-------------------'''
        jac_ineq = lambda alpha: -np.diag(
            np.ones(N))  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        fun_ineq2 = lambda alpha: np.array(alpha)
        jac_ineq2 = lambda alpha: np.diag(np.ones(N))

        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq',
                        'fun': fun_ineq,
                        'jac': jac_ineq},
                       {'type': 'ineq',
                        'fun': fun_ineq2,
                        'jac': jac_ineq2})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes

        supportIndices = np.logical_and((self.alpha > self.epsilon),
                                        (self.C - self.alpha > self.epsilon))
        # '''------------------- A matrix with each row corresponding to a support vector ------------------'''
        self.support = X[supportIndices]
        # margin_upper = (y[supportIndices] - K[supportIndices].dot(self.alpha * y)).min()
        # margin_lower = (y[supportIndices] - K[supportIndices].dot(self.alpha * y)).max()
        # ''' -----------------offset of the classifier------------------ '''
        if max(supportIndices) == False:
            self.b = 0
        else:
            self.b = (y[supportIndices] - K[supportIndices].dot(self.alpha * y)).mean()
        # '''------------------------RKHS norm of the function f ------------------------------'''
        self.norm_f = np.sqrt(self.alpha.T @ K @ self.alpha)
        #         self.support_alpha_y = self.alpha[supportIndices] * y[supportIndices]
        self.X_train = X
        self.alpha_y_train = self.alpha * y

    ### Implementation of the separting function $f$
    def separating_function(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        #         K_x = self.kernel(x, self.support)
        #         return K_x.dot(self.support_alpha_y) + self.b
        K_x = self.kernel(x, self.X_train)
        return K_x.dot(self.alpha_y_train)  # + self.b  # `b` to be added in the predict method

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d + self.b > 0) - 1

    def predict_proba(self, X):
        return self.separating_function(X)+ self.b > 0

"""
NOTE: there are two ways to do multi-class classification in SVM
cf. https://stackoverflow.com/questions/1958267/how-to-do-multi-class-classification-using-support-vector-machines-svm
- One vs One: decided by majority votes of all classifier.
- One vs Rest: decided by the highest score.

For reference, LibSVM's implementation use one-against-one, but authors state that performance are similar.
"""


class OneVSRestSVC:
    def __init__(self, num_class, C, kernel, epsilon=1e-3):
        self.num_class = num_class
        self.svcs = []
        for i in range(num_class):
            self.svcs.append(KernelSVC(C, kernel, epsilon))

    def fit(self, X, y):
        assert min(y) >= 0
        assert max(y) < self.num_class

        for i in range(self.num_class):
            # build train data
            print(f'OvR_SVM {i}')
            _y = np.where(y==i, 1, -1)
            self.svcs[i].fit(X, _y)
        return

    def predict(self, X):
        scores = np.zeros((self.num_class, len(X)))
        for i in range(self.num_class):
            scores[i, :] = self.svcs[i].predict_proba(X)
        print(scores)
        return scores.argmax(axis=0)

    def score(self, X, y_true):
        assert len(y_true) == len(X)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


class OneOverOneSVC:
    def __init__(self, num_class, C, kernel, epsilon=1e-3):
        self.num_class = num_class
        self.pairs = []
        for i in range(num_class):
            for j in range(i + 1, num_class):
                self.pairs.append((i, j))

        self.svcs = {}
        for i, j in self.pairs:
            self.svcs[(i, j)] = KernelSVC(C, kernel, epsilon)


    def fit(self, X, y):
        assert min(y) >= 0
        assert max(y) < self.num_class

        for i, j in self.pairs:
            print(f"{i}, {j}...")
            # build train data
            _y_p = np.where(y==i, 1, 0)
            _y_n = np.where(y==j, -1, 0)
            _y = _y_n + _y_p
            _X = X[_y != 0]
            _y = _y[_y!= 0]
            self.svcs[(i, j)].fit(_X, _y)
        return

    def predict(self, X):
        scores = np.zeros((self.num_class, len(X)))
        for i, j in self.pairs:
            _pred = self.svcs[(i, j)].predict(X)
            scores[i, :] += (_pred > 0)
            scores[j, :] += (_pred < 0)
        # print(scores)
        return scores.argmax(axis=0)

    def score(self, X, y_true):
        assert len(y_true) == len(X)
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


