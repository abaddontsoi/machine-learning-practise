import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer

data = load_iris()
X = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.DataFrame(data.target, columns=["Species"])

df = pd.concat([X,y], axis = 1)

# print head dataset
print('Print head dataset')
print(df.head(10))

# print X
print('Print X')
print(X)

# print y
print('Print y')
print(y)

# print data
print('Print data')
print(data)