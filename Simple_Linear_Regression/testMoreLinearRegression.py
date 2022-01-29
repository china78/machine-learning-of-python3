import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50]
y = y[y < 50]

X_train, X_test, y_train, y_test = train_test_split(X, y)

reg = LinearRegression()

reg.fit_normal(X_train, y_train)

print(X_test.shape)
predict = reg.predict(X_test)
print(predict)

MAE = reg.score(X_test, y_test)

print('MAE: ', MAE)
