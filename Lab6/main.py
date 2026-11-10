# 1. Bartell's Test of Sphericity (correlogram) in range [-1, 1]
# 2. KMO test in range [0, 1]
# 3. Factor loadings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import  (
    FactorAnalyzer,
    calculate_bartlett_sphericity,
    calculate_kmo
)

rng = np.random.default_rng(42)
n = 300

latent_factors = rng.normal(size=(n, 3))

questions = {
    "Price_1": (0,0.90),
    "Price_2": (0,0.85),
    "Price_3": (0,0.80),
    "Quality_1": (1,0.90),
    "Quality_2": (1,0.85),
    "Quality_3": (1,0.80),
    "Service_1": (2,0.90),
    "Service_2": (2,0.85),
    "Service_3": (2,0.80)
}

data = {}

for name, (factor_index, loading) in questions.items():
    data[name] = (
        loading *latent_factors[:, factor_index] + np.sqrt(1-loading**2) * rng.normal(size=n)
    )

df = pd.DataFrame(data)
print("First five customers:")
print(df.head(5))

# Correlation matrix
plt.figure(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Bartlett's Test of Sphericity

chi_square, p_value = calculate_bartlett_sphericity(df)
print("Bartlett's Test of Sphericity:")
print(f"Chi-square: {round(chi_square, 2)}")
print(f"P-value: {round(p_value, 5)}")

if(p_value < 0.05):
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")

# KMO Test
kmo_items, kmo_total = calculate_kmo(df)
print("KMO Test:")
print("Overall KMO: ", round(kmo_total, 3))

kmo_table = pd.Series(
    kmo_items, index=df.columns, name="KMO"
)

print(kmo_table.round(3))

# eigenvalues and scree plot

fa_initial = FactorAnalyzer(
    n_factors=df.shape[1], 
    rotation=None,
    method="minres"
)

fa_initial.fit(df)
eigenvalues, _ = fa_initial.get_eigenvalues()

print("Eigenvalues:")
print(np.round(eigenvalues, 3))

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues)
plt.title("Eigenvalues (Scree Plot)")
plt.axhline(y=1, color='red', linestyle='--')
plt.xlabel("Component number")
plt.ylabel("Eigenvalue")
plt.xticks(range(1, len(eigenvalues)+1))
plt.show()