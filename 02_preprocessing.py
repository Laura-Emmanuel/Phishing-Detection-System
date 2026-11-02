import pandas as pd
import numpy as np
import os

df = pd.read_csv('phishing.csv')
target_col = df.columns[-1]

# ── 5A: Check and report missing values ─────────────────────────────
print('Missing values before cleaning:')
print(df.isnull().sum())
print(f'Total missing: {df.isnull().sum().sum()}')

# ── 5B: Drop rows with too many missing values (>50% missing) ───────
threshold = len(df.columns) * 0.5
df = df.dropna(thresh=threshold)
print(f'Rows after dropping high-missing rows: {len(df)}')

# ── 5C: Fill remaining missing numeric values with column mean ───────
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
print('Missing values after imputation:', df.isnull().sum().sum())

# ── 5D: Remove duplicate rows ────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f'Duplicates removed: {before - after}')
print(f'Clean dataset shape: {df.shape}')

df.to_csv('df_cleaned.csv', index=False)
print('Saved: df_cleaned.csv')
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('df_cleaned.csv')
target_col = df.columns[-1]

# ── 6A: Identify non-numeric (categorical) columns ───────────────────
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c != target_col]   # exclude target
print(f'Categorical columns to encode: {cat_cols}')
# ── 6B: Apply Label Encoding to each categorical column ─────────────
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    print(f'  Encoded: {col}')

# ── 6C: Standardize the target label to 0 and 1 ─────────────────────
# Many datasets use -1 for phishing; we convert to 0=legitimate, 1=phishing
unique_labels = df[target_col].unique()
print(f'Current target labels: {unique_labels}')

if -1 in unique_labels:
    df[target_col] = df[target_col].map({-1: 1, 1: 0})  # -1=phishing→1
    print('Target labels remapped: -1→1 (phishing), 1→0 (legitimate)')

print(f'\nFinal class distribution:')
print(df[target_col].value_counts())
df.to_csv('df_encoded.csv', index=False)
print('Saved: df_encoded.csv')
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('df_encoded.csv')
target_col = df.columns[-1]

# ── 7A: Separate features (X) from target label (y) ─────────────────
X = df.drop(columns=[target_col])
y = df[target_col]

print(f'Feature matrix shape: {X.shape}')
print(f'Target vector shape:  {y.shape}')
print(f'Features: {X.columns.tolist()}')

# ── 7B: Apply Min-Max normalization to all numeric features ──────────
# This implements Equation 3.1 from your methodology
scaler = MinMaxScaler()   # Scales everything to [0, 1]
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

print(f'\nFeature value ranges BEFORE normalization:')
print(X.describe().loc[['min','max']].T.head(8))
print(f'\nFeature value ranges AFTER normalization:')
print(X_scaled.describe().loc[['min','max']].T.head(8))

# Save scaler for use in the Flask app later
import joblib
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print('Scaler saved to models/scaler.pkl')
print('Feature names saved to models/feature_names.pkl')
from imblearn.over_sampling import SMOTE
from collections import Counter

# ── 8A: Check if imbalance exists ────────────────────────────────────
print('Class distribution BEFORE SMOTE:')
print(Counter(y))

# ── 8B: Apply SMOTE only if imbalance ratio > 1.5:1 ─────────────────
counts = Counter(y)
majority = max(counts.values())
minority = min(counts.values())
ratio = majority / minority
print(f'\nImbalance ratio: {ratio:.2f}')

if ratio > 1.5:
    print('Imbalance detected — applying SMOTE...')
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
    print(f'\nClass distribution AFTER SMOTE:')
    print(Counter(y_balanced))
    print(f'New dataset size: {X_balanced.shape[0]} samples')
else:
    print('Dataset is balanced — SMOTE not needed.')
    X_balanced, y_balanced = X_scaled, y

print(f'\nFinal balanced dataset: {X_balanced.shape}')
from sklearn.model_selection import train_test_split

# ── 9A: Perform 80/20 split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.20,         # 20% reserved for testing
    random_state=42,        # ensures reproducibility
    stratify=y_balanced     # maintains class ratio in both sets
)

print('=== TRAIN-TEST SPLIT COMPLETE ===')
print(f'Training set:  {X_train.shape[0]} samples ({X_train.shape[0]/len(X_balanced)*100:.1f}%)')
print(f'Testing set:   {X_test.shape[0]}  samples ({X_test.shape[0]/len(X_balanced)*100:.1f}%)')
print(f'Features:      {X_train.shape[1]}')

print(f'\nTraining class distribution:')
print(Counter(y_train))
print(f'Testing class distribution:')
print(Counter(y_test))

# ── 9B: Save processed data for use in next phase ────────────────────
import joblib
joblib.dump((X_train, X_test, y_train, y_test), 'processed_data.pkl')
print('\nPreprocessed data saved to processed_data.pkl')
print('Preprocessing pipeline COMPLETE.')
