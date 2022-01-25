import numpy as np
from math import sqrt
from collections import Counter
from accuracy_score import accuracy_score

raw_data_X = [
    [3.393533211, 2.331273381],
    [3.110073483, 1.781539638],
    [1.343808831, 3.368360954],
    [3.582294042, 4.679179110],
    [2.280362439, 2.866990263],
    [7.423436942, 4.696522875],
    [5.745051997, 3.533989803],
    [9.172168622, 2.511101045],
    [7.792783481, 3.424088941],
    [7.939820817, 0.791637231]
]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
x = np.array([8.093607318, 3.365731514]).reshape(1, -1)


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]
        assert self.k <= X_train.shape[0]
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None
        assert X_predict.shape[1] == self._X_train.shape[1]
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1]
        distance = [sqrt(np.sum(((x_train - x) ** 2)))
                    for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)


knn_clf = KNNClassifier(k=6)
knn_clf.fit(X_train=X_train, y_train=y_train)
predict = knn_clf.predict(X_predict=x)
# print(predict)
