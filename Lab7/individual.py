import pandas as pd
import numpy as np
from pathlib import Path
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parent / "test.csv", index_col=0)
#print(df)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())

#sns.pairplot(df, diag_kind="kde")
#plt.show()

# Labeling numeric and catagorical columns 
def grab_column_names(df, cardinality_threshold=20):
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist() 
    
    cardinal_columns = [col for col in categorical_columns if df[col].nunique() > cardinality_threshold]
    
    return numeric_columns, categorical_columns, cardinal_columns 

# Select numerical columns
numeric_columns, categorical_columns, cardinal_columns = grab_column_names(df)     
print("Numeric columns :", numeric_columns,)
print("Catagorical Columns:", categorical_columns )

# Ploting all Numerical columns in histogram
'''
def plot_histograms(df, columns, rows=2, cols=3):
    """
    Plot histograms for specified columns in a dataframe.

    Args:
    df (pandas.DataFrame): The dataframe containing the data.
    columns (list): List of column names to plot.
    rows (int): Number of rows in the plot grid. Default is 2.
    cols (int): Number of columns in the plot grid. Default is 3.
    """ 
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, col in enumerate(columns):
        if i < len(axes):  # Check if we have enough subplots
            sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
            axes[i].set_title(col)
            axes[i].set_xlabel('')  # Remove x-label as it's redundant with title

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show() 

plot_histograms(df, numeric_columns, rows=4, cols=4)    

# Ploting all Numerical columns in histogram

def plot_box(df, columns, rows=2, cols=3):
    """
    Plot histograms for specified columns in a dataframe.

    Args:
    df (pandas.DataFrame): The dataframe containing the data.
    columns (list): List of column names to plot.
    rows (int): Number of rows in the plot grid. Default is 2.
    cols (int): Number of columns in the plot grid. Default is 3.
    """ 
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    for i, col in enumerate(columns):
        if i < len(axes):  # Check if we have enough subplots
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xlabel('')  # Remove x-label as it's redundant with title

    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show() 

plot_box(df, numeric_columns, rows=4, cols=4)
'''

# Standardize the data
df = df[numeric_columns].drop(columns="id")
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)
# Convert back to DataFrame if needed
df = pd.DataFrame(df_standardized, columns=df.columns)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated variables (threshold = 0.8)
high_corr = df.corr().abs().stack().reset_index()
high_corr = high_corr[high_corr[0] > 0.8]
high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
print("Highly Correlated Variables:")
print(high_corr)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each variable
vif_data = pd.DataFrame()
vif_data["Variable"] = df.columns
vif_data["VIF"] = [round(variance_inflation_factor(df.values, i), 4) for i in range(len(df.columns))]

# Sort the VIF values in descending order
print(vif_data.sort_values(by="VIF", ascending=False))

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
calculate_bartlett_sphericity(df)

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df)
if kmo_model >.6:
    print(f'The overall KMO for our data is {kmo_model: .2f}, which exceeds 0.6. You can proceed with factor analysis.')

else:
    
    print(f'The overall KMO for our data is {kmo_model: .2f}, which is  inadequate for Factor analysis.')

# instantiate the Factor Analyzer
fa = FactorAnalyzer() 

# Fit the dataframe using Factor Analyzer
fa.fit(df)

# Identify the eigenvalues
ev, v = fa.get_eigenvalues() #eigenvalues

# display the eigenvalues
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

'''


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
'''
