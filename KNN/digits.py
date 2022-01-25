import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from train_test_split import train_test_split
from SimulationSkKnnClass import KNNClassifier
from accuracy_score import accuracy_score

digits = datasets.load_digits()

X = digits.data
y = digits.target

some_digit = X[666]

some_digit_image = some_digit.reshape(8, 8)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.show()

X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)
my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train, y_train)
# y_predict = my_knn_clf.predict(X_test)
# accuracy0 = accuracy_score(y_true=y_test, y_predict=y_predict)
accuracy = my_knn_clf.score(X_test, y_test)
# print(accuracy0)
print(accuracy)
