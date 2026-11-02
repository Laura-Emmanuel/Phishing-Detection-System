"""Flask web application for the phishing detection system."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_extraction import analyze_url_risk, extract_model_url_features, score_email_risk
from model_builder import ensure_model_artifacts


app = Flask(__name__, template_folder="templates")


def _load_artifact(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path}. Run the preprocessing and training "
            "steps first so the models folder is created."
        )
    return joblib.load(path)


ensure_model_artifacts()
ALLOWED_ALGORITHMS = {"random_forest", "logistic_regression"}
MODELS = {
    name: model
    for name, model in _load_artifact("all_models.pkl").items()
    if name in ALLOWED_ALGORITHMS
}
SCALER = _load_artifact("scaler.pkl")
FEATURES = _load_artifact("feature_names.pkl")

if not MODELS:
    MODELS = {
        "random_forest": _load_artifact("rf_phishing_model.pkl"),
        "logistic_regression": _load_artifact("logistic_regression_model.pkl"),
    }


def _model_label(name: str) -> str:
    labels = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }
    return labels.get(name, name.replace("_", " ").title())


def _predict_from_features(
    raw_features: dict[str, int],
    risk_analysis: dict[str, object],
    selected_algorithm: str = "random_forest",
) -> dict[str, object]:
    feature_vector = np.array([raw_features.get(name, 0) for name in FEATURES]).reshape(1, -1)
    scaled_vector = SCALER.transform(feature_vector)

    model_outputs = {}
    for name, model in MODELS.items():
        prediction = int(model.predict(scaled_vector)[0])
        probabilities = model.predict_proba(scaled_vector)[0]
        confidence = probabilities[prediction] * 100
        model_outputs[_model_label(name)] = {
            "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
            "confidence": f"{confidence:.1f}%",
        }

    if selected_algorithm not in MODELS:
        selected_algorithm = "random_forest"

    model = MODELS[selected_algorithm]
    model_prediction = int(model.predict(scaled_vector)[0])
    probabilities = model.predict_proba(scaled_vector)[0]
    model_confidence = probabilities[model_prediction] * 100
    risk_score = int(risk_analysis["score"])

    prediction = model_prediction
    confidence = model_confidence
    algorithm_name = _model_label(selected_algorithm)

    phishing_signals = len(risk_analysis["reasons"])

    if risk_score >= 30 or phishing_signals >= 3:
        prediction = 1
        confidence = max(model_confidence, min(98, 64 + (risk_score * 0.45)))
        algorithm_name = f"{algorithm_name} + URL risk rules"
    elif risk_score >= 20 and model_prediction == 1:
        confidence = max(model_confidence, min(95, 58 + risk_score))

    return {
        "prediction": "PHISHING" if prediction == 1 else "LEGITIMATE",
        "confidence": f"{confidence:.1f}%",
        "confidence_band": "HIGH" if confidence >= 90 else "MEDIUM" if confidence >= 70 else "LOW",
        "risk_level": "HIGH" if prediction == 1 else "LOW",
        "features": raw_features,
        "algorithm": algorithm_name,
        "model_outputs": model_outputs,
        "risk_score": risk_score,
        "reasons": risk_analysis["reasons"],
        "registered_domain": risk_analysis["registered_domain"],
        "signals_checked": len(FEATURES) + 14,
        "active_signals": phishing_signals,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict_url", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    algorithm = data.get("algorithm", "random_forest")
    if not url:
        return jsonify({"error": "Please enter a URL"}), 400

    risk_analysis = analyze_url_risk(url)
    result = _predict_from_features(
        extract_model_url_features(url, FEATURES),
        risk_analysis,
        algorithm,
    )
    result["url"] = url
    return jsonify(result)


@app.route("/predict_email", methods=["POST"])
def predict_email():
    data = request.get_json(silent=True) or {}
    body = data.get("body", "")
    subject = data.get("subject", "")
    sender = data.get("sender", "")
    algorithm = data.get("algorithm", "random_forest")
    if not body.strip():
        return jsonify({"error": "Please enter email body text"}), 400

    return jsonify(score_email_risk(body, subject, sender, algorithm))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting Phishing Detection System...")
    print(f"Open your browser at: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
