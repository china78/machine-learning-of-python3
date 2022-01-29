import numpy as np


class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1
        assert len(x_train) == len(y_train)
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) ** 2
            d += (x_i - x_mean)*(y_i - y_mean)

        a = d / num
        b = y_mean - a * x_mean
        self.a_ = a
        self.b_ = b
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1
        assert self.a_ is not None and self.b_ is not None
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self, single_x):
        return self.a_ * single_x + self.b_

# 向量化


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1
        assert len(x_train) == len(y_train)
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        d = (x_train - x_mean).dot(y_train - y_mean)
        num = (x_train - x_mean).dot(x_train - x_mean)

        a = d / num
        b = y_mean - a * x_mean
        self.a_ = a
        self.b_ = b
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1
        assert self.a_ is not None and self.b_ is not None
        return np.array([self._predict(i) for i in x_predict])

    def _predict(self, single_x):
        return self.a_ * single_x + self.b_
