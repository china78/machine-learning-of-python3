from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from train_test_split import train_test_split

digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, y_train, X_test, y_test = train_test_split(X, y)

params_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, params_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
knn_clf = grid_search.best_estimator_

accuracy = knn_clf.score(X_test, y_test)
print('accuracy: ', accuracy)
