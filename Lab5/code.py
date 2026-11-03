import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent / 'penguins.csv')
column_names = {
    'Species': 'species',
    'Island': 'island',
    'Culmen Length (mm)': 'culmen_length_mm',
    'Culmen Depth (mm)': 'culmen_depth_mm',
    'Flipper Length (mm)': 'flipper_length_mm',
    'Body Mass (g)': 'body_mass_g',
    'Sex': 'sex',
}
df = df.rename(columns=column_names)[list(column_names.values())]

print("Missing values by column before cleanup:")
print(df.isna().sum())

df = df.dropna()
df = df.loc[df['sex'] != '.'].copy()
df['species'] = df['species'].str.split().str[0]
df = df.reset_index(drop=True)

print(df.head())
print("Cleaned shape:", df.shape, df.isna().sum())

# Compare culmen length and depth, with each species shown as a cluster.
fig, ax = plt.subplots(figsize=(8, 6))
for species, group in df.groupby('species'):
    ax.scatter(
        group['culmen_length_mm'],
        group['culmen_depth_mm'],
        label=species,
        alpha=0.7,
    )
ax.set_xlabel('Culmen length (mm)')
ax.set_ylabel('Culmen depth (mm)')
ax.set_title('Penguin bill measurements by species')
ax.legend(title='Species')
ax.grid(alpha=0.2)
fig.tight_layout()

# Standardize the measurements before PCA because they use different units.
features = [
    'culmen_length_mm',
    'culmen_depth_mm',
    'flipper_length_mm',
    'body_mass_g',
]
scaled_features = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
pca_df['species'] = df['species'].to_numpy()

print("Explained variance by PC:", pca.explained_variance_ratio_.round(3))
print("Total variance explained:", round(pca.explained_variance_ratio_.sum(), 3))
print("PCA feature loadings:")
print(pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2']).round(3))
print("Interpretation: PC1 mainly tracks overall body size; PC2 mainly tracks bill measurements.")

fig, ax = plt.subplots(figsize=(8, 6))
for species, group in pca_df.groupby('species'):
    ax.scatter(group['PC1'], group['PC2'], label=species, alpha=0.7)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title('Penguins projected onto the first two principal components')
ax.legend(title='Species')
ax.grid(alpha=0.2)
fig.tight_layout()

plt.show()
