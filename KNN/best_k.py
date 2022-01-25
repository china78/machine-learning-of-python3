from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from train_test_split import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.2)

best_p = -1
best_score = 0.0
best_k = -1
best_method = ''
for p in [1, 2]:
    # for method in ['uniform', 'distance']:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            # best_method = method
            best_p = p
print('best_p: ', best_p)
print('best_k: ', best_k)
print('best_score: ', best_score)
# print('best_methods: ', best_method)
