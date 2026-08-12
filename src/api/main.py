"""FastAPI application for SWaT anomaly detection.

Loads the production model and serves predictions with explainability.
The API maintains a rolling buffer of recent readings to compute
time-window features (rolling mean, frozen detection, etc.).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from tensorflow import keras

from src.api.database import Base, engine, get_db
from src.api.models_db import IncidentSensor, Prediction
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    DetectionDetail,
    HealthResponse,
    InvolvedSensor,
    PredictionResponse,
    SensorReading,
)
from src.explainability.attack_classifier import classify_attack
from src.explainability.explainer import explain_window
from src.monitoring.drift_detection import check_drift
from src.models.error_analysis_v2 import compute_raw_features
from src.preprocessing.feature_config import load_config
from src.preprocessing.feature_engineering import (
    compute_deviation,
    compute_pair_residuals,
    compute_rate_of_change,
    compute_rolling_features,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models" / "production"
BUFFER_SIZE = 300

app = FastAPI(
    title="SWaT Anomaly Detection API",
    description="Real-time anomaly detection for industrial control systems",
    version="1.0.0",
)

model_state = {
    "ae": None,
    "svm": None,
    "sensor_scaler": None,
    "feat_scaler": None,
    "feature_names": None,
    "continuous_sensors": None,
    "ae_threshold": None,
    "svm_threshold": None,
    "config": None,
    "buffer": None,
    "loaded": False,
}


@app.on_event("startup")
def load_model():
    """Load all model artifacts and create DB tables at startup."""
    Base.metadata.create_all(bind=engine)
    model_state["ae"] = keras.models.load_model(MODELS_DIR / "autoencoder.keras")
    model_state["svm"] = joblib.load(MODELS_DIR / "svm.joblib")
    model_state["sensor_scaler"] = joblib.load(MODELS_DIR / "sensor_scaler.joblib")
    model_state["feat_scaler"] = joblib.load(MODELS_DIR / "feature_scaler.joblib")

    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)

    model_state["feature_names"] = metadata["feature_names"]
    model_state["continuous_sensors"] = metadata["continuous_sensors"]
    model_state["ae_threshold"] = metadata["ae_threshold"]
    model_state["svm_threshold"] = metadata["svm_threshold"]
    model_state["config"] = load_config()
    model_state["buffer"] = pd.DataFrame()
    model_state["loaded"] = True


def build_features_from_buffer(buffer: pd.DataFrame) -> np.ndarray:
    """Build the full feature vector from the rolling buffer.

    Uses the last row as the current reading, but needs history
    for rolling/frozen/spike features.
    """
    config = model_state["config"]
    sensor_scaler = model_state["sensor_scaler"]
    feature_names = model_state["feature_names"]
    continuous = model_state["continuous_sensors"]

    raw_feat = compute_raw_features(buffer)

    df_scaled = buffer.copy()
    df_scaled[continuous] = sensor_scaler.transform(buffer[continuous])

    scaled_features = {}
    for sensor, info in config["continuous_sensors"].items():
        if sensor not in df_scaled.columns:
            continue
        series = df_scaled[sensor]
        features = info["features"]
        if "rolling_mean" in features or "rolling_std" in features:
            scaled_features.update(
                compute_rolling_features(series, sensor, config["window_sizes"], features)
            )
        if "rate_of_change" in features:
            scaled_features.update(compute_rate_of_change(series, sensor))
        if "deviation_from_baseline" in features:
            bl = config.get("baselines", {}).get(sensor, {})
            if bl:
                scaled_features.update(
                    compute_deviation(series, sensor, bl["mean"], bl["std"])
                )
    scaled_features.update(
        compute_pair_residuals(df_scaled, config.get("cross_sensor_pairs", []))
    )
    scaled_feat_df = pd.DataFrame(scaled_features, index=buffer.index)

    exclude = {"Timestamp", "label"}
    orig_cols = [c for c in df_scaled.columns if c not in exclude
                 and pd.api.types.is_numeric_dtype(df_scaled[c])]

    combined = pd.concat([df_scaled[orig_cols], scaled_feat_df, raw_feat], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    for col in feature_names:
        if col not in combined.columns:
            combined[col] = 0

    combined = combined[feature_names]

    last_row = combined.iloc[[-1]].values.astype(np.float32)
    last_row = np.nan_to_num(last_row, nan=0, posinf=100, neginf=-100)

    return model_state["feat_scaler"].transform(last_row)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the model is loaded and ready."""
    return HealthResponse(
        status="healthy" if model_state["loaded"] else "not_ready",
        model_loaded=model_state["loaded"],
        n_features=len(model_state["feature_names"]) if model_state["feature_names"] else 0,
        threshold=model_state["ae_threshold"] or 0.0,
        buffer_size=len(model_state["buffer"]) if model_state["buffer"] is not None else 0,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, db: Session = Depends(get_db)):
    """Predict anomaly for a single sensor reading.

    The reading is added to a rolling buffer. Features that require
    time history (rolling mean, frozen detection) use the buffer.
    Results are stored in the database.
    """
    if not model_state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([reading.model_dump()])

    buffer = model_state["buffer"]
    buffer = pd.concat([buffer, row], ignore_index=True)
    if len(buffer) > BUFFER_SIZE:
        buffer = buffer.iloc[-BUFFER_SIZE:]
    model_state["buffer"] = buffer

    X = build_features_from_buffer(buffer)

    ae = model_state["ae"]
    svm = model_state["svm"]

    recon = ae.predict(X, verbose=0)
    ae_score = float(np.mean((X - recon) ** 2))
    svm_score = float(-svm.decision_function(X)[0])

    ae_anomaly = ae_score > model_state["ae_threshold"]
    svm_anomaly = svm_score > model_state["svm_threshold"]
    is_anomaly = ae_anomaly or svm_anomaly

    attack_type = None
    description = None
    involved_sensors = []
    evidence = None

    if is_anomaly:
        explanation = explain_window(
            X, ae, model_state["feature_names"], top_n=8
        )
        classification = classify_attack(explanation)
        attack_type = classification["attack_type"]
        description = classification["description"]
        evidence = classification["evidence"]
        involved_sensors = [
            InvolvedSensor(
                sensor=s["sensor"],
                stage=s["stage"],
                contribution_pct=round(s["contribution"] * 100, 1),
            )
            for s in classification["involved_sensors"][:5]
        ]

    if is_anomaly:
        detected_parts = []
        if svm_anomaly:
            detected_parts.append("SVM")
        if ae_anomaly:
            detected_parts.append("AE")
        detected_by = " + ".join(detected_parts)
    else:
        detected_by = None

    db_prediction = Prediction(
        ae_score=round(ae_score, 6),
        svm_score=round(svm_score, 6),
        is_anomaly=is_anomaly,
        detected_by=detected_by,
        attack_type=attack_type,
        description=description,
        evidence=evidence,
    )
    for s in involved_sensors:
        db_prediction.sensors.append(
            IncidentSensor(
                sensor=s.sensor, stage=s.stage, contribution_pct=s.contribution_pct,
            )
        )
    db.add(db_prediction)
    db.commit()

    return PredictionResponse(
        anomaly_score=round(ae_score, 6),
        is_anomaly=is_anomaly,
        detected_by=detected_by,
        attack_type=attack_type,
        description=description,
        involved_sensors=involved_sensors,
        evidence=evidence,
        detection=DetectionDetail(
            ae_score=round(ae_score, 6),
            ae_anomaly=ae_anomaly,
            svm_score=round(svm_score, 6),
            svm_anomaly=svm_anomaly,
        ),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, db: Session = Depends(get_db)):
    """Predict anomalies for multiple readings at once."""
    if not model_state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for reading in batch.readings:
        result = predict(reading, db)
        results.append(result)

    anomaly_count = sum(1 for r in results if r.is_anomaly)

    return BatchResponse(
        total=len(results),
        anomalies=anomaly_count,
        results=results,
    )


@app.get("/health/model")
def model_health():
    """Check model health — run drift detection on the current buffer."""
    if not model_state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    buffer = model_state["buffer"]
    if buffer is None or len(buffer) < 30:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 30 rows, have {len(buffer) if buffer is not None else 0}",
        }

    continuous = model_state["continuous_sensors"]
    buffer_values = {
        sensor: buffer[sensor].values
        for sensor in continuous
        if sensor in buffer.columns
    }

    return check_drift(buffer_values)
