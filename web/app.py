"""
CustomerPulse Web App — Flask application for churn prediction.
Run: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")

app = Flask(__name__)

# Try to load the model at startup
model = None
preprocessor = None

def load_model():
    """Load the saved model and preprocessor."""
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"[*] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print("[!] No model found. Please run the notebook first to train and save a model.")


load_model()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main page with prediction form."""
    model_loaded = model is not None
    return render_template("index.html", model_loaded=model_loaded)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request from the form."""
    if model is None or preprocessor is None:
        return jsonify({"error": "No model loaded. Run the notebook first."}), 500

    try:
        # Collect form data
        data = {
            "gender": request.form.get("gender", "Male"),
            "SeniorCitizen": int(request.form.get("SeniorCitizen", 0)),
            "Partner": request.form.get("Partner", "No"),
            "Dependents": request.form.get("Dependents", "No"),
            "tenure": int(request.form.get("tenure", 1)),
            "PhoneService": request.form.get("PhoneService", "Yes"),
            "MultipleLines": request.form.get("MultipleLines", "No"),
            "InternetService": request.form.get("InternetService", "DSL"),
            "OnlineSecurity": request.form.get("OnlineSecurity", "No"),
            "OnlineBackup": request.form.get("OnlineBackup", "No"),
            "DeviceProtection": request.form.get("DeviceProtection", "No"),
            "TechSupport": request.form.get("TechSupport", "No"),
            "StreamingTV": request.form.get("StreamingTV", "No"),
            "StreamingMovies": request.form.get("StreamingMovies", "No"),
            "Contract": request.form.get("Contract", "Month-to-month"),
            "PaperlessBilling": request.form.get("PaperlessBilling", "Yes"),
            "PaymentMethod": request.form.get("PaymentMethod", "Electronic check"),
            "MonthlyCharges": float(request.form.get("MonthlyCharges", 70.0)),
            "TotalCharges": float(request.form.get("TotalCharges", 70.0)),
        }

        # Create DataFrame
        df = pd.DataFrame([data])

        # Preprocess and predict
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]

        churn_prob = round(float(probability[1]) * 100, 1)
        result = "Yes — Customer is likely to churn" if prediction == 1 else "No — Customer is likely to stay"

        # Risk level
        if churn_prob >= 70:
            risk = "HIGH"
            risk_color = "#e74c3c"
        elif churn_prob >= 40:
            risk = "MEDIUM"
            risk_color = "#f39c12"
        else:
            risk = "LOW"
            risk_color = "#27ae60"

        return jsonify({
            "prediction": result,
            "churn_probability": churn_prob,
            "risk_level": risk,
            "risk_color": risk_color,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
