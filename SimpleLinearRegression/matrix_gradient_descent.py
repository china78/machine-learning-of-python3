import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# np.random.seed(666)
x = 2 * np.random.random(size=100)
y = x * 3. + 4. + np.random.normal(size=100)
X = x.reshape(-1, 1)
plt.scatter(X, y)
plt.show()

print('X: ', X.shape)
print('y: ', y.shape)
print('X.len: ', len(X))
# 损失函数


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


def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, precision=1e-8):
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


X_b = np.hstack([np.ones((X.shape[0], 1)), X])
initial_theta = np.zeros(X_b.shape[1])
eta = 0.01
print('initial_thate: ', initial_theta)
theta = gradient_descent(X_b, y, initial_theta, eta)

print('theta: ', theta)


# 测试LinearRegression类中的fit_gd

lin_reg = LinearRegression()
lin_reg.fit_gd(X, y)
print('lin_reg.theta_: ', lin_reg.theta_)
