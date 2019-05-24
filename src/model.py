import numpy as np
from numpy.random import *


class SVM(object):

    def __init__(self, cost=1, _iter=1000):
        self.C = cost
        self._iter = _iter

    def fit(self, X, Y):
        x_bias = np.c_[X, np.ones(X.shape[0])]
        w = np.ones(X.shape[1] + 1)
        for _i in range(1, self._iter):
            target = randint(1, X.shape[0])
            _x = x_bias[target]
            _y = Y[target]
            eta = 1 / (self.C * _i)
            if _y * np.dot(w, _x) < 1:
                w = (1 - eta * self.C) * w + eta * _y * _x
            else:
                w = (1 - eta * self.C) * w
        self.w = w

    def predict(self, X):
        x = np.c_[X, np.ones(X.shape[0])]
        return np.where(np.dot(x, self.w) > 0, 1, -1)


class Kernel_SVM(object):

    def __init__(self, cost=1, _iter=1000, gamma=1):
        self.C = cost
        self._iter = _iter
        self.gamma = gamma

    def _gauss_kernel(self, x_1, x_2):
        return np.exp(-self.gamma * np.sum((x_1 - x_2) ** 2, axis=1))

    def fit(self, X, Y):
        self.Y = Y
        self.X_bias = np.c_[X, np.ones(X.shape[0])]
        n_row, n_col = X.shape
        a_vec = np.zeros(n_row)
        for _i in range(1, self._iter):
            # i
            target = randint(0, n_row)
            # i以外
            ind = np.ones(n_row, dtype=bool)
            ind[target] = False
            # iとi以外に分ける
            _x, _x_rest = self.X_bias[target], self.X_bias[ind]
            _y = Y[target]

            threshold = _y * 1 / (self.C * _i) * np.sum(
                a_vec[ind] * _y * self._gauss_kernel(_x, _x_rest))
            if threshold < 1:
                a_vec[target] = a_vec[target] + 1
            else:
                print("not")

        self.a_vec = a_vec

    def predict(self, X):
        predict_x_bias = np.c_[X, np.ones(X.shape[0])]
        result = []
        for i in range(len(predict_x_bias)):
            _x = predict_x_bias[i]
            threshold = 1 / (self.C * self._iter) * np.sum(
                self.a_vec * self.Y * self._gauss_kernel(_x, self.X_bias))
            result.append(1 if threshold > 0 else -1)
        return np.array(result)
