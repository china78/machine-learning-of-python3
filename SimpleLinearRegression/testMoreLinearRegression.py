import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
from sklearn.preprocessing import StandardScaler

boston = datasets.load_boston()

X = boston.data
y = boston.target

X = X[y < 50]
y = y[y < 50]

X_train, X_test, y_train, y_test = train_test_split(X, y)

reg = LinearRegression()

# 正规方程法
reg.fit_normal(X_train, y_train)
score = reg.score(X_test, y_test)


MAE = reg.score(X_test, y_test)

print('MAE: ', MAE)


# 梯度下降法
lin_reg = LinearRegression()
lin_reg.fit_gd(X_train, y_train, eta=0.000001, n_iters=1e6)
lin_reg_score = lin_reg.score(X_test, y_test)
print('lin_score: ', lin_reg_score)

# 得到的score依然不理想, 如果继续增大n_iters的值能提升score，但是太消耗性能
## 根本原因: 数据的规模不同统一
## 解决方案: 归一化
standardScaler = StandardScaler()
# fit是为了取均值和方差
standardScaler.fit(X_train)
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

lin_reg_scale = LinearRegression()
lin_reg_scale.fit_gd(X_train_standard, y_train)
lin_reg_scale_score = lin_reg_scale.score(X_test_standard, y_test)
print('lin_reg_scale_score: ', lin_reg_scale_score)
