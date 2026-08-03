"""Feature attribution for detected anomalies.

For each anomaly window, ranks which features (and sensors) contributed
most to detection. Uses per-feature reconstruction error for Autoencoder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

STAGE_MAP = {
    "FIT101": 1, "LIT101": 1, "MV101": 1, "P101": 1, "P102": 1,
    "AIT201": 2, "AIT202": 2, "AIT203": 2, "FIT201": 2, "MV201": 2,
    "P201": 2, "P202": 2, "P203": 2, "P204": 2, "P205": 2, "P206": 2,
    "DPIT301": 3, "FIT301": 3, "LIT301": 3, "MV301": 3, "MV302": 3,
    "MV303": 3, "MV304": 3, "P301": 3, "P302": 3,
    "AIT401": 4, "AIT402": 4, "FIT401": 4, "LIT401": 4,
    "P401": 4, "P402": 4, "P403": 4, "P404": 4, "UV401": 4,
    "AIT501": 5, "AIT502": 5, "AIT503": 5, "AIT504": 5,
    "FIT501": 5, "FIT502": 5, "FIT503": 5, "FIT504": 5,
    "P501": 5, "P502": 5, "PIT501": 5, "PIT502": 5, "PIT503": 5,
    "FIT601": 6, "P601": 6, "P602": 6, "P603": 6,
}

FEATURE_CATEGORIES = {
    "rolling": ["_mean_", "_std_"],
    "rate_of_change": ["_roc"],
    "deviation": ["_dev"],
    "contradiction": ["_contra_"],
    "frozen": ["_frozen_"],
    "spike_jump": ["_spike_", "_max_jump_"],
    "constant_change": ["_not_normal", "_abnormal_duration", "_changed"],
    "switching": ["_switch_"],
    "pair_residual": ["_residual"],
    "low_variance": ["_low_var_"],
}


def classify_feature(feature_name: str) -> str:
    """Map a feature name to its category."""
    for category, patterns in FEATURE_CATEGORIES.items():
        if any(p in feature_name for p in patterns):
            return category
    return "raw_sensor"


def extract_sensor(feature_name: str) -> str:
    """Extract the base sensor name from a feature name."""
    for sensor in sorted(STAGE_MAP.keys(), key=len, reverse=True):
        if feature_name.startswith(sensor):
            return sensor
    parts = feature_name.split("_")
    if len(parts) >= 2 and parts[0] + parts[1] in STAGE_MAP:
        return parts[0] + parts[1]
    return feature_name.split("_")[0]


def get_stage(sensor: str) -> int:
    """Return the SWaT stage number for a sensor."""
    return STAGE_MAP.get(sensor, 0)


def compute_ae_feature_importance(
    X_window: np.ndarray,
    reconstructed: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Per-feature reconstruction error averaged over an attack window."""
    per_feature_error = np.mean((X_window - reconstructed) ** 2, axis=0)

    records = []
    for i, feat in enumerate(feature_names):
        sensor = extract_sensor(feat)
        records.append({
            "feature": feat,
            "reconstruction_error": per_feature_error[i],
            "category": classify_feature(feat),
            "sensor": sensor,
            "stage": get_stage(sensor),
        })

    df = pd.DataFrame(records)
    return df.sort_values("reconstruction_error", ascending=False)


def explain_window(
    X_window: np.ndarray,
    ae_model,
    feature_names: list[str],
    top_n: int = 10,
) -> dict:
    """Explain a single attack window — which features drove detection."""
    reconstructed = ae_model.predict(X_window, verbose=0)
    importance = compute_ae_feature_importance(
        X_window, reconstructed, feature_names,
    )

    top_features = importance.head(top_n)
    category_totals = (
        importance.groupby("category")["reconstruction_error"]
        .sum()
        .sort_values(ascending=False)
    )
    stage_totals = (
        importance[importance["stage"] > 0]
        .groupby("stage")["reconstruction_error"]
        .sum()
        .sort_values(ascending=False)
    )
    sensor_totals = (
        importance.groupby("sensor")["reconstruction_error"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return {
        "top_features": top_features,
        "category_totals": category_totals,
        "stage_totals": stage_totals,
        "sensor_totals": sensor_totals,
        "full_importance": importance,
    }


def explain_all_windows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    ae_model,
    feature_names: list[str],
    windows: list[tuple[int, int]],
) -> list[dict]:
    """Explain all attack windows in one pass."""
    reconstructed = ae_model.predict(X_test, verbose=0)
    results = []
    for i, (s, e) in enumerate(windows):
        importance = compute_ae_feature_importance(
            X_test[s:e + 1], reconstructed[s:e + 1], feature_names,
        )
        category_totals = (
            importance.groupby("category")["reconstruction_error"]
            .sum()
            .sort_values(ascending=False)
        )
        stage_totals = (
            importance[importance["stage"] > 0]
            .groupby("stage")["reconstruction_error"]
            .sum()
            .sort_values(ascending=False)
        )
        sensor_totals = (
            importance.groupby("sensor")["reconstruction_error"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        results.append({
            "window": f"P{i + 1}",
            "duration": e - s + 1,
            "top_features": importance.head(10),
            "category_totals": category_totals,
            "stage_totals": stage_totals,
            "sensor_totals": sensor_totals,
            "full_importance": importance,
        })
    return results
