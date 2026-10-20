import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1,1,1)
ax.set_title("Weekly evolution")
ax.set_xlabel("Day")
ax.set_ylabel("Value")
days=["M", "T", "W", "TH", "F", "S", "SU"]
values=[12,15,11,18,22,20,9]
ax.plot(days, values, marker="o", color="blue")
ax.axhline(np.mean(values), color="red", linestyle="--", label="Mean")

plt.show()

#2 scatter plot with labels

t = pd.DataFrame({
    "comp1": [-2.1,0.5,1.8,-0.3],
    "comp2": [0.4,-1.2,0.9,1.5],
}, index=["RO", "HU", "DE", "FR"])

t.plot.scatter(x="comp1", y="comp2", s=200, color="green", title="Scatterplot", xlabel="Component 1", ylabel="Component 2",)
print(t)

plt.show()

#3 heatmap
import seaborn as sns

r = np.array([
    [1.0, 0.72, -0.15],
    [0.72, 1.0, 0.31],
    [-0.15, 0.31, 1.0]
])

sns.heatmap(r, annot=True, cmap="coolwarm", xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])

plt.show()