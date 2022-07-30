from sklearn.datasets import load_breast_cancer

# breast cancer dataset
cancer_data = load_breast_cancer()

X = cancer_data.data
y = cancer_data.target
print("Print out cancer data")

X = X[:, :10]
print(X)
print(y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

accuracy_score(y, y_pred)