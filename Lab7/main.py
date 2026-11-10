import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt

np.random.seed(0)
data = np.random.rand(50, 6)
df = pd.DataFrame(data, columns=[f'var{i+1}' for i in range(6)])
print(df)

chi_sq, p = calculate_bartlett_sphericity(df)
print(f'Chi-square: {chi_sq}, P-value: {p}')

kmo_all, kmo_model = calculate_kmo(df)
print(f'KMO: {kmo_model}')

fa = FactorAnalyzer(rotation="varimax")
fa.fit(df)
eigen_values, _ = fa.get_eigenvalues()
print(f'Eigenvalues: {eigen_values}')

plt.plot(range(1, df.shape[1]+1), eigen_values, marker='o')
plt.title('Scree Plot'); plt.xlabel('Factors'); plt.ylabel('Eigenvalue'); plt.grid(True); plt.show()

fa = FactorAnalyzer(n_factors=2, rotation="varimax")
fa.fit(df)

print("Factor Loadings:\n", fa.loadings_)
print("Factor Variance:\n", fa.get_factor_variance())
print("Factor Scores:\n", fa.transform(df))