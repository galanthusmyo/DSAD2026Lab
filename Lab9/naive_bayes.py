
import io
import zipfile
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# ==========================================
# 1. DOWNLOAD THE REAL DATASET
# ==========================================

url = (
    "https://archive.ics.uci.edu/static/public/"
    "222/bank+marketing.zip"
)

print("Downloading UCI Bank Marketing dataset...")

with urlopen(url, timeout=60) as response:
    data = response.read()

# The UCI archive contains another ZIP archive.
with zipfile.ZipFile(io.BytesIO(data)) as outer:
    inner_zip = outer.read("bank-additional.zip")

with zipfile.ZipFile(io.BytesIO(inner_zip)) as inner:
    csv_data = inner.read(
        "bank-additional/bank-additional-full.csv"
    )

df = pd.read_csv(io.BytesIO(csv_data), sep=";")

print("\nDataset shape:", df.shape)
print(df.head())


# ==========================================
# 2. BUSINESS QUESTION AND VARIABLES
# ==========================================

# Target:
# y = yes -> customer subscribed
# y = no  -> customer did not subscribe

df["subscribed"] = (df["y"] == "yes").astype(int)

# Choose numerical variables known before a call.
# Keep the model simple for teaching.

features = [
    "age",           # Customer age
    "campaign",      # Contacts in current campaign
    "previous",      # Contacts before current campaign
    "euribor3m"      # Economic interest-rate indicator
]

X = df[features]
y = df["subscribed"]

print("\nSubscription distribution:")
print(y.value_counts())

print("\nSubscription percentages:")
print(y.value_counts(normalize=True) * 100)

# Plot the class distribution
counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar(
    ["Not subscribed", "Subscribed"],
    counts.values
)
plt.title("Bank Marketing - Subscription Outcomes")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))


# ==========================================
# 4. GAUSSIAN NAIVE BAYES MODEL
# ==========================================

model = GaussianNB()
model.fit(X_train, y_train)

# Class predictions (0 or 1)
y_pred = model.predict(X_test)

# Predicted probability of subscription
y_prob = model.predict_proba(X_test)[:, 1]


# ==========================================
# 5. MODEL EVALUATION
# ==========================================

print("\n--- GAUSSIAN NAIVE BAYES ---")

print("Accuracy:", round(
    accuracy_score(y_test, y_pred), 3
))

print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 3
))

print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 3
))

print("ROC AUC:", round(
    roc_auc_score(y_test, y_prob), 3
))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No", "Yes"],
    zero_division=0
))

# Compare against a naive baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("Baseline accuracy:", round(
    accuracy_score(y_test, baseline_pred), 3
))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No", "Yes"],
    cmap="Blues",
    values_format="d"
)

plt.title("Gaussian Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.show()


# ==========================================
# 6. PREDICT INDIVIDUAL CUSTOMERS
# ==========================================

# Use actual held-out customers.
# This avoids inventing unrealistic customer profiles.

examples = X_test.head(5).copy()

examples["P(subscribe)"] = model.predict_proba(
    examples[features]
)[:, 1]

examples["Prediction"] = model.predict(
    examples[features]
)

print("\n--- FIVE CUSTOMERS ---")
print(examples.round(3).to_string())


# ==========================================
# 7. BUSINESS APPLICATION:
# WHICH CUSTOMERS SHOULD WE CALL FIRST?
# ==========================================

# Simulate prioritizing the top 20% of
# test-set contacts according to predicted
# subscription probabilities.

results = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "probability": y_prob
})

results = results.sort_values(
    "probability", ascending=False
).reset_index(drop=True)

top_n = int(np.ceil(len(results) * 0.20))
top_customers = results.head(top_n)

overall_rate = results["actual"].mean()
targeted_rate = top_customers["actual"].mean()

lift = targeted_rate / overall_rate

print("\n--- MARKETING PERFORMANCE ---")
print("Overall subscription rate:",
      round(overall_rate * 100, 2), "%")

print("Top 20% subscription rate:",
      round(targeted_rate * 100, 2), "%")

print("Targeting lift:", round(lift, 2), "x")

# Plot business impact
plt.figure(figsize=(8, 5))

plt.bar(
    ["Random targeting", "Top 20% by model"],
    [overall_rate * 100, targeted_rate * 100]
)

plt.ylabel("Actual subscription rate (%)")
plt.title("Does Bayesian Targeting Improve Selection?")

plt.tight_layout()
plt.show()


# ==========================================
# 8. PREDICTED PROBABILITY DISTRIBUTIONS
# ==========================================

plt.figure(figsize=(9, 5))

plt.hist(
    y_prob[y_test.to_numpy() == 0],
    bins=30,
    alpha=0.6,
    label="Did not subscribe",
    density=True
)

plt.hist(
    y_prob[y_test.to_numpy() == 1],
    bins=30,
    alpha=0.6,
    label="Subscribed",
    density=True
)

plt.xlabel("Predicted probability of subscription")
plt.ylabel("Density")
plt.title("Gaussian Naive Bayes - Probability Distributions")
plt.legend()
plt.tight_layout()
plt.show()
