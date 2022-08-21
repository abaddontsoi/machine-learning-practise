from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# breast cancer dataset
cancer_data = load_breast_cancer()

X = cancer_data.data
y = cancer_data.target
print("Print out cancer data")

X = X[:, :10]
print(X)
print(y)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X, y)

y_pred = model.predict(X)

print(accuracy_score(y, y_pred))