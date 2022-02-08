import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coefficents_ = None
        self.theta_ = None

    # 正规方程法
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self.theta_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self.theta_[0]
        self.coefficents_ = self.theta_[1:]
        return self

    # 梯度下降法
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4, precision=1e-8):
        assert X_train.shape[0] == y_train.shape[0]

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2)/len(X_b)
            except:
                return float('inf')

        # 对theta求偏导
        def DJ(theta, X_b, y):
            # for循环的方式
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)

            # 向量化的方式
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters, precision):
            theta = initial_theta
            i_iters = 0

            while i_iters < n_iters:
                gredient = DJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gredient
                if (abs(J(last_theta, X_b, y) - J(theta, X_b, y)) < precision):
                    break
                i_iters += 1
            return theta
        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self.theta_ = gradient_descent(
            X_b, y_train, initial_theta, eta, n_iters, precision)
        self.intercept_ = self.theta_[0]
        self.coefficents_ = self.theta_[1:]
        return self

    def predict(self, X_predict):
        assert self.coefficents_ is not None and self.theta_ is not None
        assert X_predict.shape[1] == len(self.coefficents_)
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self.theta_)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
