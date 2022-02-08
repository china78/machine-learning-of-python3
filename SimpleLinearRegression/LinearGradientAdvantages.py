import numpy as np
from LinearRegression import LinearRegression
import time

m = 1000
n = 5000

big_X = np.random.normal(size=(m, n))
true_theta = np.random.uniform(0.0, 100.0, size=n+1)
big_y = big_X.dot(true_theta[1:]) + true_theta[0] + \
    np.random.normal(0., 10., size=m)


# 正规方程
start_1 = time.time()
big_reg1 = LinearRegression()
big_reg1.fit_normal(big_X, big_y)
end_1 = time.time()

time_1 = end_1 - start_1
print('time_1: ', time_1)

# 梯度下降
start_2 = time.time()
big_reg2 = LinearRegression()
big_reg2.fit_gd(big_X, big_y)
end_2 = time.time()
time_2 = end_2 - start_2
print('time_2: ', time_2)
