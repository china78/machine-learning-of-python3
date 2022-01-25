from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from train_test_split import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from handleStandardScale import StandardScaler

iris = datasets.load_iris()

X = iris.data
y = iris.target


X_train, y_train, X_test, y_test = train_test_split(X, y)
standard_scale = StandardScaler()
standard_scale.fit(X_train)

X_train = standard_scale.transform(X_train)
X_test = standard_scale.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
