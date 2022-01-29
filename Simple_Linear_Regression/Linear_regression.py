import numpy as np
import matplotlib.pyplot as plt
from SimpleLinearRegression import SimpleLinearRegression1

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 3.0, 2.0, 3.0, 5.0])

# plt.scatter(x, y)
# plt.axis([0, 6, 0, 6])
# plt.show()

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.0
d = 0.0

for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) ** 2
    d += (x_i - x_mean)*(y_i - y_mean)

a = d / num
b = y_mean - a * x_mean

x_predict = 6.0

y_hat = a * x + b

print(y_hat)

plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.plot(x, y_hat, color="r")
plt.show()

y_predict = a * x_predict + b

print(y_predict)

reg = SimpleLinearRegression1()
reg.fit(x, y)
print('a_:', reg.a_)
print('b_:', reg.b_)

y_predict_ = reg.predict(np.array([x_predict]))
print('y_predict_: ', y_predict_)
