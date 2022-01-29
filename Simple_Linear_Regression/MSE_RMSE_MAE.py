import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from SimpleLinearRegression import SimpleLinearRegression2
from math import sqrt
from sklearn.metrics import r2_score

boston = datasets.load_boston()
X = boston.data[:, 5]
y = boston.target

X = X[y < 50]
y = y[y < 50]

plt.scatter(X, y)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = SimpleLinearRegression2()

reg.fit(X_train, y_train)

y_predict = reg.predict(X_test)
# x_predict = reg.predict(X_train)
print('y_predict: ', y_predict)

# plt.scatter(X_train, y_train)
# plt.plot(X_train, x_predict, color='r')
# plt.show()
MSE = np.sum((y_predict - y_test) ** 2) / len(y_test)
print('MSE: ', MSE)

RMSE = sqrt(MSE)
print('RMSE: ', RMSE)

MAE = np.sum(np.absolute(y_predict - y_test)) / len(y_test)
print('MAE: ', MAE)

print(X_test.shape)

R_2 = 1 - (np.sum((y_predict - y_test) ** 2) /
           np.sum((np.mean(y_test) - y_test) ** 2))
print('R_2: ', R_2)

r2_score = r2_score(y_test, y_predict)
print('r2_score: ', r2_score)
