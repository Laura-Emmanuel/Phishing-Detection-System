"""Build the saved model artifacts required by the Flask app."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "phishing.csv"
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_CONFIG = {
    "version": 2,
    "algorithms": ["random_forest", "logistic_regression"],
}


def build_model_artifacts() -> None:
    """Train Random Forest and Logistic Regression model artifacts."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    target_col = df.columns[-1]

    df = df.dropna(thresh=len(df.columns) * 0.5)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df = df.drop_duplicates()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    unique_labels = set(df[target_col].unique())
    if -1 in unique_labels:
        df[target_col] = df[target_col].map({-1: 1, 1: 0})

    columns_to_drop = [target_col]
    if "Index" in df.columns:
        columns_to_drop.append("Index")

    X = df.drop(columns=columns_to_drop)
    y = df[target_col]

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    counts = Counter(y)
    majority = max(counts.values())
    minority = min(counts.values())
    if minority > 0 and majority / minority > 1.5:
        X_balanced, y_balanced = SMOTE(random_state=42).fit_resample(X_scaled, y)
    else:
        X_balanced, y_balanced = X_scaled, y

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    }

    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["random_forest"], MODELS_DIR / "rf_phishing_model.pkl")
    joblib.dump(models["logistic_regression"], MODELS_DIR / "logistic_regression_model.pkl")
    joblib.dump(models, MODELS_DIR / "all_models.pkl")
    joblib.dump(metrics, MODELS_DIR / "model_metrics.pkl")
    joblib.dump(MODEL_CONFIG, MODELS_DIR / "model_config.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(X.columns.tolist(), MODELS_DIR / "feature_names.pkl")


def ensure_model_artifacts() -> None:
    """Create model files when a fresh deployment does not have them yet."""
    required = [
        MODELS_DIR / "rf_phishing_model.pkl",
        MODELS_DIR / "logistic_regression_model.pkl",
        MODELS_DIR / "all_models.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "feature_names.pkl",
    ]
    if not all(path.exists() for path in required):
        build_model_artifacts()
        return

    config_path = MODELS_DIR / "model_config.pkl"
    if config_path.exists() and joblib.load(config_path) != MODEL_CONFIG:
        build_model_artifacts()


if __name__ == "__main__":
    build_model_artifacts()
    print(f"Model artifacts saved in: {MODELS_DIR}")
