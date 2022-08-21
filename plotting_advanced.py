import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(100)
y1 = x1 * np.random.rand(100)
print(y1)

fig, ax = plt.subplots()
print(fig)

ax.set_title("Sin")
ax.set_xlabel('rad')
ax.bar(x1, y1)

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles, labels)
# ax.legend("handles", "labels")

print(handles, labels)
plt.show()