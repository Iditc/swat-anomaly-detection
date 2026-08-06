"""Train the production model and save all artifacts for the API.

Replicates the error_analysis_v2 pipeline and saves:
- Autoencoder model (.keras)
- Sensor scaler (StandardScaler for raw sensors)
- Feature scaler (StandardScaler for combined feature matrix)
- Feature column names (ordered list)
- Anomaly threshold (float)
- Training baselines for constant sensors
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, f1_score
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing.load_data import clean_normal_data
from src.preprocessing.feature_config import load_config
from src.preprocessing.prepare_data import get_continuous_sensors
from src.preprocessing.feature_engineering import (
    compute_rolling_features, compute_rate_of_change,
    compute_deviation, compute_pair_residuals,
)
from src.models.error_analysis_v2 import (
    compute_raw_features, get_attack_windows, ALWAYS_CONSTANT,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
MODELS_DIR = ROOT / "models" / "production"
RANDOM_STATE = 42


def build_feature_matrix(df, sensor_scaler, config, fit_sensor_scaler=False):
    """Build the full 349-feature matrix from a dataframe.

    Returns (X, feature_names, sensor_scaler).
    """
    raw_feat = compute_raw_features(df)

    continuous = get_continuous_sensors(df)
    df_scaled = df.copy()
    if fit_sensor_scaler:
        sensor_scaler.fit(df[continuous])
    df_scaled[continuous] = sensor_scaler.transform(df[continuous])

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
                scaled_features.update(compute_deviation(series, sensor, bl["mean"], bl["std"]))
    scaled_features.update(
        compute_pair_residuals(df_scaled, config.get("cross_sensor_pairs", []))
    )
    scaled_feat_df = pd.DataFrame(scaled_features, index=df.index)

    exclude = {"Timestamp", "label"}
    orig_cols = [c for c in df_scaled.columns if c not in exclude
                 and pd.api.types.is_numeric_dtype(df_scaled[c])]

    combined = pd.concat([df_scaled[orig_cols], scaled_feat_df, raw_feat], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    feature_names = list(combined.columns)
    X = np.nan_to_num(combined.values.astype(np.float32), nan=0, posinf=100, neginf=-100)

    return X, feature_names, sensor_scaler


def train_autoencoder(X_train, epochs=100):
    """Train the autoencoder on normal data."""
    n_feat = X_train.shape[1]
    ae = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(n_feat, activation="linear"),
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_train, X_train, epochs=epochs, batch_size=256,
           validation_split=0.1, verbose=1)
    return ae


def find_threshold(scores, y_test, name="model"):
    """Find the optimal anomaly threshold using F1 on precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_test, scores)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    best_idx = np.argmax(f1)
    threshold = float(thr[min(best_idx, len(thr) - 1)])

    print(f"  {name} best F1: {f1[best_idx]:.4f} at threshold: {threshold:.6f}")
    return threshold


def train_svm(X_train):
    """Train One-Class SVM on a sample of normal data."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), 50_000, replace=False)
    svm = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm.fit(X_train[idx])
    return svm


def save_constant_baselines(train_df):
    """Save mode values for always-constant sensors."""
    baselines = {}
    for sensor in ALWAYS_CONSTANT:
        if sensor in train_df.columns:
            baselines[sensor] = int(train_df[sensor].mode().iloc[0])
    return baselines


def main():
    print("=" * 70)
    print("  SAVE PRODUCTION MODEL")
    print("=" * 70)

    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = clean_normal_data(normal)

    train_df = normal.copy()
    attack_start = attack["Timestamp"].min()
    attack_end = attack["Timestamp"].max()
    normal_in_test = normal[
        (normal["Timestamp"] >= attack_start) & (normal["Timestamp"] <= attack_end)
    ].copy()
    test_df = pd.concat([normal_in_test, attack], ignore_index=True)
    test_df = test_df.sort_values("Timestamp").reset_index(drop=True)
    y_test = test_df["label"].values

    print(f"\n  Train: {len(train_df):,} rows (normal only)")
    print(f"  Test:  {len(test_df):,} rows ({(y_test == 1).sum():,} attack)")

    config = load_config()
    sensor_scaler = StandardScaler()

    print("\nBuilding train feature matrix...")
    X_train, feature_names, sensor_scaler = build_feature_matrix(
        train_df, sensor_scaler, config, fit_sensor_scaler=True
    )

    print("Building test feature matrix...")
    X_test, _, _ = build_feature_matrix(test_df, sensor_scaler, config)

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    print(f"\n  Features: {len(feature_names)}")

    print("\nTraining Autoencoder...")
    ae = train_autoencoder(X_train)

    print("\nTraining SVM...")
    svm = train_svm(X_train)

    print("\nFinding optimal thresholds...")
    ae_recon = ae.predict(X_test, verbose=0)
    ae_scores = np.mean((X_test - ae_recon) ** 2, axis=1)
    ae_threshold = find_threshold(ae_scores, y_test, "AE")

    svm_scores = -svm.decision_function(X_test)
    svm_threshold = find_threshold(svm_scores, y_test, "SVM")

    constant_baselines = save_constant_baselines(train_df)

    # Save all artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ae.save(MODELS_DIR / "autoencoder.keras")
    joblib.dump(svm, MODELS_DIR / "svm.joblib")
    joblib.dump(sensor_scaler, MODELS_DIR / "sensor_scaler.joblib")
    joblib.dump(feat_scaler, MODELS_DIR / "feature_scaler.joblib")

    continuous = get_continuous_sensors(train_df)
    metadata = {
        "feature_names": feature_names,
        "continuous_sensors": continuous,
        "ae_threshold": ae_threshold,
        "svm_threshold": svm_threshold,
        "n_features": len(feature_names),
        "constant_baselines": constant_baselines,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to {MODELS_DIR}/:")
    print(f"    autoencoder.keras")
    print(f"    svm.joblib")
    print(f"    sensor_scaler.joblib")
    print(f"    feature_scaler.joblib")
    print(f"    model_metadata.json ({len(feature_names)} features)")
    print(f"    AE threshold: {ae_threshold:.6f}")
    print(f"    SVM threshold: {svm_threshold:.6f}")
    print("\n  Done! Ready for FastAPI.")


if __name__ == "__main__":
    main()
