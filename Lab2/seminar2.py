import os
import pandas as pd
from scipy.spatial.distance import cdist

# Create output folder
os.makedirs("data_out", exist_ok=True)


# Requirement 1 - Load file
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "winequality-red.csv")
df = pd.read_csv(csv_path)

# Shape
print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn types:")
print(df.dtypes)


# Requirement 2 - Missing values

# Verify NaN before
print("\nNaN before:")
print(df.isna().sum())

# Replace NaN values in numeric columns
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    mean = df[column].mean()
    df[column] = df[column].fillna(mean)

# Check after cleanup
print("\nNaN values after:")
print(df.isna().sum())

print("\nTotal NaN remaining:")
print(df.isna().sum().sum())


# Requirement 3 - Filter wines with quality >= 7
quality_min7 = df[df["quality"] >= 7]

quality_min7.to_csv(
    "data_out/vinuri_calitate_min7.csv",
    index=False
)


# Requirement 4 - Top 50 wines by alcohol
top50_alcohol = df.sort_values(
    by="alcohol",
    ascending=False
).head(50)

top50_alcohol.to_csv(
    "data_out/top50_alcool.csv",
    index=False
)


# Requirement 5 - Average alcohol and pH by quality
average_by_quality = df.groupby("quality")[
    ["alcohol", "pH"]
].mean().reset_index()

average_by_quality.to_csv(
    "data_out/medii_pe_calitate.csv",
    index=False
)


# Requirement 6 - Derived acidity column
df["aciditate_totala"] = (
    df["fixed acidity"]
    + df["volatile acidity"]
    + df["citric acid"]
)

wines_with_acidity = df[
    [
        "quality",
        "alcohol",
        "aciditate_totala",
        "pH"
    ]
]

wines_with_acidity.to_csv(
    "data_out/vinuri_cu_aciditate.csv",
    index=False
)


# Requirement 7 - Quality with maximum average alcohol
average_alcohol_by_quality = df.groupby(
    "quality"
)["alcohol"].mean()

quality_max_alcohol = average_alcohol_by_quality.idxmax()

result_quality = pd.DataFrame({
    "quality": [quality_max_alcohol]
})

result_quality.to_csv(
    "data_out/calitate_max_alcool.csv",
    index=False
)

print("\nQuality with maximum average alcohol:")
print(quality_max_alcohol)


# Requirement 8 - Correlation matrix
correlation_columns = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "alcohol",
    "quality"
]

R = df[correlation_columns].corr()

R.to_csv(
    "data_out/R.csv",
    index=True
)

print("\nCorrelation matrix:")
print(R)


# Requirement 9 - Standardize alcohol

# Sample standard deviation is used: ddof = 1
alcohol_mean = df["alcohol"].mean()
alcohol_std = df["alcohol"].std(ddof=1)

df["alcohol_std"] = (
    df["alcohol"] - alcohol_mean
) / alcohol_std

standardized_alcohol = df[
    [
        "alcohol",
        "alcohol_std"
    ]
]

standardized_alcohol.to_csv(
    "data_out/alcohol_standardizat.csv",
    index=False
)


# Requirement 10 - Euclidean distance matrix

distance_data = df[
    [
        "fixed acidity",
        "volatile acidity",
        "alcohol"
    ]
].head(20).copy()

# Standardize each column using sample standard deviation
for column in distance_data.columns:
    mean = distance_data[column].mean()
    std = distance_data[column].std(ddof=1)

    distance_data[column] = (
        distance_data[column] - mean
    ) / std

# Calculate Euclidean distances
distance_matrix = cdist(
    distance_data,
    distance_data,
    metric="euclidean"
)

distance_df = pd.DataFrame(
    distance_matrix,
    index=range(20),
    columns=range(20)
)

distance_df.to_csv(
    "data_out/d_euclid_20.csv",
    index=True
)

print("\nEuclidean distance matrix:")
print(distance_df)