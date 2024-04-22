from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron


iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)  # "Iris setosa" and "no iris setosa" labels


# Training
perceptron_clf = Perceptron(random_state=42)
perceptron_clf.fit(X, y)


# Testing
X_new_input = [[3, 0.5], [2, 0.5]]
y_predicted = perceptron_clf.predict(X_new_input)

print(y_predicted)
