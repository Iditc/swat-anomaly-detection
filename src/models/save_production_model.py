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
from tensorflow import keras
from tensorflow.keras import layers

from src.models.error_analysis_v2 import (
    ALWAYS_CONSTANT,
    compute_raw_features,
)
from src.preprocessing.feature_config import load_config
from src.preprocessing.feature_engineering import (
    compute_deviation,
    compute_pair_residuals,
    compute_rate_of_change,
    compute_rolling_features,
)
from src.preprocessing.load_data import clean_normal_data
from src.preprocessing.prepare_data import get_continuous_sensors

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


def find_threshold_percentile(train_scores, name="model", percentile=99):
    """Set anomaly threshold from a percentile of normal training errors.

    Uses only training data (no test labels) to avoid leakage.
    Percentile 99 means: flag anything above the worst 1% of normal behavior.
    """
    threshold = float(np.percentile(train_scores, percentile))
    print(f"  {name} threshold (P{percentile}): {threshold:.6f}")
    return threshold


def train_svm(X_train):
    """Train One-Class SVM on a sample of normal data."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_sample = min(50_000, len(X_train))
    idx = rng.choice(len(X_train), n_sample, replace=False)
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

    print("\nSetting thresholds from training data (no test labels used)...")
    ae_recon_train = ae.predict(X_train, verbose=0)
    ae_train_scores = np.mean((X_train - ae_recon_train) ** 2, axis=1)
    ae_threshold = find_threshold_percentile(ae_train_scores, "AE", percentile=99)

    svm_train_scores = -svm.decision_function(X_train)
    svm_threshold = find_threshold_percentile(svm_train_scores, "SVM", percentile=99)

    print("\nOverfitting check (train vs test error):")
    ae_recon_test = ae.predict(X_test, verbose=0)
    ae_test_scores = np.mean((X_test - ae_recon_test) ** 2, axis=1)
    svm_test_scores = -svm.decision_function(X_test)

    ae_train_mean = float(np.mean(ae_train_scores))
    ae_test_normal = ae_test_scores[y_test == 0]
    ae_test_attack = ae_test_scores[y_test == 1]
    ae_test_mean = float(np.mean(ae_test_normal))

    train_fp_rate = float(np.mean(ae_train_scores > ae_threshold))
    test_normal_fp_rate = float(np.mean(ae_test_normal > ae_threshold))
    test_attack_recall = float(np.mean(ae_test_attack > ae_threshold))

    print("  AE reconstruction error:")
    print(f"    Train normal:  mean={ae_train_mean:.4f}  "
          f"FP rate={train_fp_rate:.3f}")
    print(f"    Test normal:   mean={ae_test_mean:.4f}  "
          f"FP rate={test_normal_fp_rate:.3f}")
    print(f"    Test attack:   mean={float(np.mean(ae_test_attack)):.4f}  "
          f"recall={test_attack_recall:.3f}")

    if test_normal_fp_rate > train_fp_rate * 3:
        print("  ⚠️  Test FP rate much higher than train — possible overfitting")
    else:
        print("  ✓  No overfitting detected (train/test FP rates similar)")

    print("\nEvaluating on test set (for reference only — not used for threshold)...")

    ae_preds = (ae_test_scores > ae_threshold).astype(int)
    svm_preds = (svm_test_scores > svm_threshold).astype(int)
    ensemble_preds = (ae_preds | svm_preds).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score

    from src.models.error_analysis_v2 import point_adjust

    print("\n  Point-Wise (honest):")
    print(f"    AE      — P: {precision_score(y_test, ae_preds):.3f}  "
          f"R: {recall_score(y_test, ae_preds):.3f}  "
          f"F1: {f1_score(y_test, ae_preds):.3f}")
    print(f"    SVM     — P: {precision_score(y_test, svm_preds):.3f}  "
          f"R: {recall_score(y_test, svm_preds):.3f}  "
          f"F1: {f1_score(y_test, svm_preds):.3f}")
    print(f"    Ensemble— P: {precision_score(y_test, ensemble_preds):.3f}  "
          f"R: {recall_score(y_test, ensemble_preds):.3f}  "
          f"F1: {f1_score(y_test, ensemble_preds):.3f}")

    ae_pa = point_adjust(y_test, ae_preds)
    svm_pa = point_adjust(y_test, svm_preds)
    ensemble_pa = point_adjust(y_test, ensemble_preds)

    print("\n  Point-Adjusted (for comparison with published benchmarks):")
    print(f"    AE      — P: {precision_score(y_test, ae_pa):.3f}  "
          f"R: {recall_score(y_test, ae_pa):.3f}  "
          f"F1: {f1_score(y_test, ae_pa):.3f}")
    print(f"    SVM     — P: {precision_score(y_test, svm_pa):.3f}  "
          f"R: {recall_score(y_test, svm_pa):.3f}  "
          f"F1: {f1_score(y_test, svm_pa):.3f}")
    print(f"    Ensemble— P: {precision_score(y_test, ensemble_pa):.3f}  "
          f"R: {recall_score(y_test, ensemble_pa):.3f}  "
          f"F1: {f1_score(y_test, ensemble_pa):.3f}")

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
    print("    autoencoder.keras")
    print("    svm.joblib")
    print("    sensor_scaler.joblib")
    print("    feature_scaler.joblib")
    print(f"    model_metadata.json ({len(feature_names)} features)")
    print(f"    AE threshold: {ae_threshold:.6f}")
    print(f"    SVM threshold: {svm_threshold:.6f}")
    print("\n  Done! Ready for FastAPI.")


if __name__ == "__main__":
    main()
