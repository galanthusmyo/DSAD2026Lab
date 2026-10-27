import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

ax = t.plot.scatter(x="comp1", y="comp2", s=200, color="green", title="Scatterplot", xlabel="Component 1", ylabel="Component 2",)
for label, row in t.iterrows():
    ax.annotate(label, (row["comp1"], row["comp2"]))
print(t)

plt.show()

#3 heatmap
r = np.array([
    [1.0, 0.72, -0.15],
    [0.72, 1.0, 0.31],
    [-0.15, 0.31, 1.0]
])

sns.heatmap(r, annot=True, cmap="coolwarm", xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])

plt.show()


### PCA example

#1. import dataset

t = pd.read_csv(Path(__file__).resolve().parent / "iris.data.csv", header=None)

#2. select 4 numerical variables


t.columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

X = t.iloc[:, :4]

#3. standardize the variables
X_scaled = StandardScaler().fit_transform(X)

#4. apply PCA: reduce from 4 variables to 2 PC 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#5. Create a df with the new components
result = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
result["Species"] = t["Species"]

#6. See how much variance is preserved
print("Explained variance:", pca.explained_variance_ratio_)
print("Total variance preserved:", pca.explained_variance_ratio_.sum())

#7. visualize the results
for species, group in result.groupby("Species"):
    plt.scatter(group["PC1"], group["PC2"], label=species)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA1 vs PCA2")
plt.legend()
plt.show()
