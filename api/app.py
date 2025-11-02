import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import joblib
from mlflow.pyfunc import load_model
import pandas as pd

load_dotenv()
app = Flask(__name__)

MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "models/regression_model.pkl")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "")
_model = None
_model_source = None

def load_regression_model():
    global _model, _model_source
    if _model is not None:
        return _model
    try:
        if MLFLOW_MODEL_URI:
            app.logger.info(f"Loading model from MLflow URI: {MLFLOW_MODEL_URI}")
            _model = load_model(MLFLOW_MODEL_URI)
            _model_source = "mlflow"
            return _model
    except Exception as e:
        app.logger.warning(f"Failed to load MLflow model: {e}")
    app.logger.info(f"Loading local model: {MODEL_LOCAL_PATH}")
    _model = joblib.load(MODEL_LOCAL_PATH)
    _model_source = "local"
    return _model

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_source": _model_source})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    required = ["from", "to", "flightType", "time", "distance", "agency", "date"]
    missing = [r for r in required if r not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    df = pd.DataFrame([payload])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    df = df.assign(
        month=df["date"].dt.month,
        dayofweek=df["date"].dt.dayofweek,
        is_weekend=df["date"].dt.dayofweek.isin([5, 6]).astype(int),
    )
    model_features = ["from", "to", "flightType", "time", "distance", "agency", "month", "dayofweek", "is_weekend"]
    df = df[model_features]

    model = load_regression_model()
    pred = model.predict(df)[0]
    return jsonify({"predicted_price": float(pred)})

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    load_regression_model()
    app.run(host=host, port=port)