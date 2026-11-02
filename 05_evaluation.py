from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

rf_model = joblib.load('models/rf_phishing_model.pkl')
X_train, X_test, y_train, y_test = joblib.load('processed_data.pkl')

# ── Generate predictions on the test set ─────────────────────────────
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]  # Confidence scores

# ── Compute all metrics from Section 3.10 ────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print('=' * 50)
print('       MODEL EVALUATION RESULTS')
print('=' * 50)
print(f'  Accuracy  (Eq 3.2): {accuracy*100:.2f}%')
print(f'  Precision (Eq 3.3): {precision*100:.2f}%')
print(f'  Recall    (Eq 3.4): {recall*100:.2f}%')
print(f'  F1-Score  (Eq 3.5): {f1*100:.2f}%')
print('=' * 50)

print('\n=== DETAILED CLASSIFICATION REPORT ===')
print(classification_report(y_test, y_pred,
      target_names=['Legitimate (0)', 'Phishing (1)']))


# ── Confusion Matrix (Table 3.6) ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print('=== CONFUSION MATRIX (Table 3.6) ===')
print(f'  True Positive  (TP): {tp}  ← Correctly detected phishing')
print(f'  True Negative  (TN): {tn}  ← Correctly detected legitimate')
print(f'  False Positive (FP): {fp}  ← Legitimate flagged as phishing')
print(f'  False Negative (FN): {fn}  ← Missed phishing attacks')
print(f'\n  False Positive Rate: {fp/(fp+tn)*100:.2f}%')
print(f'  False Negative Rate: {fn/(fn+tp)*100:.2f}%')
 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Plot confusion matrix as heatmap ─────────────────────────────────
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Legitimate', 'Phishing'],
    yticklabels=['Legitimate', 'Phishing'],
    linewidths=2
)
plt.title('Confusion Matrix — Random Forest Phishing Detection', fontsize=13)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual Label', fontsize=11)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print('Confusion matrix saved as confusion_matrix.png')

# ── Plot ROC Curve ───────────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='#1565C0', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], 'k--', lw=1)
plt.xlim([0,1]); plt.ylim([0,1.02])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest Phishing Detection')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print(f'AUC Score: {roc_auc:.4f}')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

classifiers = {
    'Random Forest (Proposed)': rf_model,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

results = []
for name, clf in classifiers.items():
    if name != 'Random Forest (Proposed)':
        clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results.append({
        'Classifier':  name,
        'Accuracy':    f'{accuracy_score(y_test,preds)*100:.2f}%',
        'Precision':   f'{precision_score(y_test,preds)*100:.2f}%',
        'Recall':      f'{recall_score(y_test,preds)*100:.2f}%',
        'F1-Score':    f'{f1_score(y_test,preds)*100:.2f}%',
    })
results_df = pd.DataFrame(results)
print('=== COMPARATIVE CLASSIFIER RESULTS ===')
print(results_df.to_string(index=False))
results_df.to_csv('rf_logistic_comparison.csv', index=False)
print('\nSaved to rf_logistic_comparison.csv — Random Forest and Logistic Regression only.')
 
