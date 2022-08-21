from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


data = load_wine()

x = data.data[:, [0]]
y = data.data[:, [9]]

# plt.scatter(x, y)

df_x = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=["kind(target)"])


df = pd.concat([df_x, df_y], axis=1)
print(df.head(20))


fig, ax = plt.subplots()
ax.set_title("wine")
ax.set_ylabel('alcohol')
ax.boxplot(df.loc[:, "alcohol"])
print(df.corr())