"""Load saved model artifacts and expose a reusable prediction function."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from feature_extraction import analyze_url_risk, extract_model_url_features, score_email_risk


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts():
    allowed_algorithms = {"random_forest", "logistic_regression"}
    models = {
        name: model
        for name, model in joblib.load(MODELS_DIR / "all_models.pkl").items()
        if name in allowed_algorithms
    }
    if not models:
        models = {
            "random_forest": joblib.load(MODELS_DIR / "rf_phishing_model.pkl"),
            "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression_model.pkl"),
        }
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
    return models, scaler, feature_names


def predict_phishing(
    url: str | None = None,
    email_body: str | None = None,
    email_subject: str = "",
    sender: str = "",
    algorithm: str = "random_forest",
) -> dict[str, object]:
    """Predict whether a URL or email is phishing."""
    models, scaler, feature_names = load_artifacts()

    if url:
        raw_features = extract_model_url_features(url, feature_names)
        risk_analysis = analyze_url_risk(url)
    elif email_body:
        return score_email_risk(email_body, email_subject, sender)
    else:
        return {"error": "Provide either a URL or email body"}

    feature_vector = np.array([raw_features.get(name, 0) for name in feature_names]).reshape(1, -1)
    scaled_vector = scaler.transform(feature_vector)

    if algorithm not in models:
        algorithm = "random_forest"

    model = models[algorithm]
    prediction = int(model.predict(scaled_vector)[0])
    confidence = model.predict_proba(scaled_vector)[0][prediction]
    risk_score = int(risk_analysis["score"])

    if risk_score >= 30 or len(risk_analysis["reasons"]) >= 3:
        prediction = 1
        confidence = max(confidence, min(0.98, 0.64 + (risk_score * 0.0045)))

    return {
        "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence": f"{confidence * 100:.1f}%",
        "risk_level": "HIGH" if prediction == 1 else "LOW",
        "features": raw_features,
        "algorithm": algorithm.replace("_", " ").title(),
        "risk_score": risk_score,
        "reasons": risk_analysis["reasons"],
    }


if __name__ == "__main__":
    test = predict_phishing(url="http://paypal-login.update.net/verify")
    print(f"Prediction: {test['prediction']} (Confidence: {test['confidence']})")
