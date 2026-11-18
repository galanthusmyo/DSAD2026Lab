"""
Load hernia.csv, separate predictors vs class.
Counts per class → clase_count.csv.
Group means of predictors by class → medii_predictori_pe_clasa.csv.
One plot (e.g. degree_spondylolisthesis) by class.
train_test_split(0.3, stratify, random_state=42).
Fit LinearDiscriminantAnalysis on train; print classes_, priors_.
Predict on test → Predictii_test.csv.
Accuracy + confusion matrix → Acuratete_lda.csv, CM_lda.csv.
transform() on train → z.csv (Z1, Z2).
Scatter Z1 vs Z2, coloured by class + group centres.
Distributions of predictors by class (as in f_distributii).
Short interpretation: which predictor “looks” most discriminative?
"""


"""Linear discriminant analysis of the hernia dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# The first column identifies rows and is not a predictor.
df = pd.read_csv(Path(__file__).with_name('hernia.csv'))
X = df.drop(columns=['No.', 'class'])
y = df['class']

y.value_counts().rename_axis('class').to_csv('clase_count.csv', header=['count'])
df.groupby('class')[X.columns].mean().to_csv('medii_predictori_pe_clasa.csv')

# Compare a predictor across the three classes.
df.boxplot(column='degree_spondylolisthesis', by='class')
plt.title('Degree of spondylolisthesis by class')
plt.suptitle('')
plt.ylabel('degree_spondylolisthesis')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
print('Classes:', model.classes_)
print('Priors:', model.priors_)

predictions = model.predict(X_test)
pd.DataFrame({'actual': y_test, 'predicted': predictions}).to_csv('Predictii_test.csv', index=False)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)
pd.DataFrame({'accuracy': [accuracy]}).to_csv('Acuratete_lda.csv', index=False)
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
pd.DataFrame(cm, index=model.classes_, columns=model.classes_).rename_axis('actual').to_csv('CM_lda.csv')
print('Confusion matrix:\n', cm)

# Three classes produce two discriminant coordinates.
z = pd.DataFrame(model.transform(X_train), columns=['Z1', 'Z2'], index=X_train.index)
z.to_csv('z.csv', index=False)
plt.figure()
for class_name in model.classes_:
    points = z.loc[y_train == class_name]
    plt.scatter(points['Z1'], points['Z2'], alpha=0.7, label=class_name)
    centre = points.mean()
    plt.scatter(centre['Z1'], centre['Z2'], marker='X', s=150, edgecolors='black')
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.title('LDA training observations and class centres')
plt.legend()
plt.show()

# Compare all predictor distributions using the same class labels.
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, predictor in zip(axes.flat, X.columns):
    for class_name in model.classes_:
        ax.hist(df.loc[y == class_name, predictor], bins=20, alpha=0.4, label=class_name)
    ax.set_title(predictor)
    ax.set_ylabel('Count')
axes[0, 0].legend()
fig.tight_layout()
plt.show()

print('Interpretation: degree_spondylolisthesis appears particularly discriminative')
print('for Spondylolisthesis, although Hernia and Normal overlap substantially.')
