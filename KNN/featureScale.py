import numpy as np
import matplotlib.pyplot as plt

# 向量
x = np.random.randint(0, 100, size=100)

X = (x - np.min(x)) / (np.max(x) - np.min(x))
_X = (x - np.mean(x)) / np.std(x)

# 矩阵-最值
x1 = np.random.randint(0, 100, (50, 2))
x1 = np.array(x1, dtype=float)
x_10 = x1[:10, :]

print('pre_x1: ', x_10)

x_10[:, 0] = (x_10[:, 0] - np.min(x_10[:, 0])) / \
    (np.max(x_10[:, 0]) - np.min(x_10[:, 0]))
x_10[:, 1] = (x_10[:, 1] - np.min(x_10[:, 1])) / \
    (np.max(x_10[:, 1]) - np.min(x_10[:, 1]))

print('next_x1: ', x_10)
plt.scatter(x_10[:, 0], x_10[:, 1])
plt.show()
print('-------------------------------------------------------------')

# 矩阵-均值方差
x2 = np.random.randint(0, 100, (50, 2))
x2 = np.array(x2, dtype=float)
x2_10 = x2[:10, :]

print('pre_x2: ', x2_10)

x2_10[:, 0] = (x2_10[:, 0] - np.mean(x2_10[:, 0])) / \
    np.std(x2_10[:, 0])
x2_10[:, 1] = (x2_10[:, 1] - np.min(x2_10[:, 1])) / \
    np.std(x2_10[:, 1])

print('next_x2: ', x2_10)

plt.scatter(x2_10[:, 0], x2_10[:, 1])
plt.show()
