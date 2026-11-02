from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time

# Load preprocessed data from Step 9
X_train, X_test, y_train, y_test = joblib.load('processed_data.pkl')

print('=== RANDOM FOREST TRAINING ===')
print(f'Training samples: {X_train.shape[0]}')
print(f'Test samples:     {X_test.shape[0]}')
print(f'Features:         {X_train.shape[1]}')
print(f'Classes:          Phishing (1) | Legitimate (0)')
print()


# ── Configure Random Forest with methodology hyperparameters ─────────
# Exact values from Table 3.4 of your methodology:
rf_model = RandomForestClassifier(
    n_estimators     = 100,       # Table 3.4: 100 decision trees
    criterion        = 'gini',    # Table 3.4: Gini impurity
    max_depth        = None,      # Table 3.4: unrestricted depth
    min_samples_split= 2,         # Table 3.4
    min_samples_leaf = 1,         # Table 3.4
    max_features     = 'sqrt',    # Table 3.4: sqrt(n_features)
    bootstrap        = True,      # Table 3.4: bagging enabled
    random_state     = 42,        # Table 3.4: reproducibility seed
    n_jobs           = -1,        # Use all CPU cores for speed
    verbose          = 1          # Show training progress
)


# ── Train the model ───────────────────────────────────────────────────
print('Training model... (this may take 1-3 minutes)')
start = time.time()
rf_model.fit(X_train, y_train)
elapsed = time.time() - start
print(f'\nTraining completed in {elapsed:.2f} seconds')
print(f'Number of trees trained: {rf_model.n_estimators}')
print(f'Number of features used: {rf_model.n_features_in_}')


# ── Quick sanity check ───────────────────────────────────────────────
train_score = rf_model.score(X_train, y_train)
test_score  = rf_model.score(X_test, y_test)
print(f'Training accuracy: {train_score*100:.2f}%')
print(f'Testing accuracy:  {test_score*100:.2f}%')

print('\n=== LOGISTIC REGRESSION TRAINING ===')
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
print(f'Logistic Regression testing accuracy: {logistic_model.score(X_test, y_test)*100:.2f}%')


# Save the trained model
import os; os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/rf_phishing_model.pkl')
joblib.dump(logistic_model, 'models/logistic_regression_model.pkl')
joblib.dump({
    'random_forest': rf_model,
    'logistic_regression': logistic_model,
}, 'models/all_models.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
print('\nModel saved to models/rf_phishing_model.pkl')
print('Logistic Regression model saved to models/logistic_regression_model.pkl')
print('Feature names saved to models/feature_names.pkl')

from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

print('=== 5-FOLD CROSS VALIDATION ===')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run cross-validation on accuracy
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv,
                             scoring='accuracy', n_jobs=-1)

print(f'CV Accuracy per fold: {[f"{s*100:.2f}%" for s in cv_scores]}')
print(f'Mean CV Accuracy:     {cv_scores.mean()*100:.2f}%')
print(f'Standard Deviation:   ±{cv_scores.std()*100:.2f}%')

# Run cross-validation on F1-score
cv_f1 = cross_val_score(rf_model, X_train, y_train, cv=cv,
                         scoring='f1', n_jobs=-1)
print(f'\nMean CV F1-Score: {cv_f1.mean()*100:.2f}%')

if cv_scores.std() < 0.02:
    print('✓ Low variance: model is stable and generalizing well.')
else:
    print('⚠ High variance: consider tuning hyperparameters.')
