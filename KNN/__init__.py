from sklearn import datasets
from train_test_split import train_test_split
from SimulationSkKnnClass import KNNClassifier

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, y_train, X_test, y_test = train_test_split(X, y)

my_knn_clf = KNNClassifier(k=3)
my_knn_clf.fit(X_train=X_train, y_train=y_train)

y_predict = my_knn_clf.predict(X_test)
print('y_predict: ', y_predict)
print('y____test: ', y_test)
accuracy = sum(y_predict == y_test) / len(y_test)
print('accuracy: ', accuracy)
