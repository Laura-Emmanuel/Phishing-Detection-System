import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 3A: Load the dataset ──────────────────────────────────────
df = pd.read_csv('phishing.csv')

# ── STEP 3B: Basic inspection ───────────────────────────────────────
print('=== DATASET SHAPE ===')
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')

print('\n=== FIRST 5 ROWS ===')
print(df.head())

print('\n=== COLUMN NAMES ===')
print(df.columns.tolist())

print('\n=== DATA TYPES ===')
print(df.dtypes)

print('\n=== MISSING VALUES PER COLUMN ===')
print(df.isnull().sum())

print('\n=== CLASS DISTRIBUTION ===')
# Target column is usually named 'Result' or 'phishing'
# -1 = Phishing, 0 or 1 = Legitimate (check your dataset)
print(df.iloc[:, -1].value_counts())
# ── STEP 4A: Class distribution chart ──────────────────────────────
target_col = df.columns[-1]   # Automatically selects the last column
class_counts = df[target_col].value_counts()

plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', color=['#2196F3', '#F44336'])
plt.title('Class Distribution: Phishing vs Legitimate')
plt.xlabel('Class (1=Phishing, -1 or 0=Legitimate)')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
print(f'Class counts:\n{class_counts}')

# ── STEP 4B: Correlation heatmap ────────────────────────────────────
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5, annot=False)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print('Charts saved as PNG files.')
