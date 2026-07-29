"""Error analysis v2 — fix contradiction features + overfitting check.

Key fix: compute contradiction/frozen/discrete features on RAW data
before scaling, then combine with scaled rolling features.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_recall_curve
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing.load_data import clean_normal_data, get_sensor_columns
from src.preprocessing.feature_config import load_config
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42

# Physical rules: (pump/valve, flow_sensor, on_value, expected_flow_direction)
CONTRADICTION_RULES = [
    ("P101", "FIT101", 2, "positive"),
    ("P102", "FIT101", 2, "positive"),
    ("P301", "FIT301", 2, "positive"),
    ("P302", "FIT301", 2, "positive"),
    ("P401", "FIT401", 2, "positive"),
    ("P501", "FIT501", 2, "positive"),
    ("P602", "FIT601", 2, "positive"),
    ("MV101", "FIT101", 2, "positive"),
    ("MV301", "FIT301", 2, "positive"),
]

ALWAYS_CONSTANT = ["P102", "P204", "P206"]


def get_attack_windows(y_true):
    windows = []
    in_attack = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            start = i
            in_attack = True
        elif y_true[i] == 0 and in_attack:
            windows.append((start, i - 1))
            in_attack = False
    if in_attack:
        windows.append((start, len(y_true) - 1))
    return windows


def point_adjust(y_true, y_pred):
    y_adjusted = y_pred.copy()
    for s, e in get_attack_windows(y_true):
        if y_pred[s:e + 1].any():
            y_adjusted[s:e + 1] = 1
    return y_adjusted


def best_f1_predictions(y_true, scores):
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    best = np.argmax(f1)
    return (scores >= thr[min(best, len(thr) - 1)]).astype(int)


def compute_raw_features(df):
    """Compute features that need RAW (unscaled) values."""
    features = {}

    # Sustained contradiction: pump ON but no flow for 30+ seconds
    for actuator, sensor, on_val, direction in CONTRADICTION_RULES:
        if actuator not in df.columns or sensor not in df.columns:
            continue
        act_on = (df[actuator] == on_val).astype(int)
        if direction == "positive":
            no_effect = (df[sensor].abs() < 0.05).astype(int)
        else:
            no_effect = (df[sensor].abs() > 0.05).astype(int)

        contradiction = act_on * no_effect
        # Sustained: rolling sum over 30s window
        features[f"{actuator}_{sensor}_contra_30"] = (
            contradiction.rolling(30, min_periods=1).sum()
        )
        features[f"{actuator}_{sensor}_contra_instant"] = contradiction

    # Always-constant sensor changed
    for sensor in ALWAYS_CONSTANT:
        if sensor not in df.columns:
            continue
        mode_val = df[sensor].mode().iloc[0]
        features[f"{sensor}_not_normal"] = (df[sensor] != mode_val).astype(int)
        # Sustained: how long has it been abnormal
        abnormal = (df[sensor] != mode_val).astype(int)
        features[f"{sensor}_abnormal_duration"] = (
            abnormal.rolling(60, min_periods=1).sum()
        )

    # Frozen detection on raw values (more meaningful than scaled)
    continuous = [c for c in df.columns
                  if c not in ("Timestamp", "label") and df[c].dtype == "float64"]
    for sensor in continuous:
        rolling_std = df[sensor].rolling(10, min_periods=10).std().fillna(df[sensor].std())
        features[f"{sensor}_frozen_raw"] = (rolling_std == 0).astype(int)
        # Also: very low variance compared to normal
        features[f"{sensor}_low_var_60"] = df[sensor].rolling(60, min_periods=30).std().fillna(
            df[sensor].std()
        )

    # Spike detection on raw values
    for sensor in continuous:
        diff = df[sensor].diff().abs().fillna(0)
        rolling_max_diff = diff.rolling(10, min_periods=1).max()
        features[f"{sensor}_max_jump_10"] = rolling_max_diff

    # Discrete switching rate
    discrete = [c for c in df.columns
                if c not in ("Timestamp", "label") and df[c].dtype in ("int64", "int32")]
    for sensor in discrete:
        switches = df[sensor].diff().ne(0).astype(int)
        features[f"{sensor}_switch_30"] = switches.rolling(30, min_periods=1).sum()

    return pd.DataFrame(features, index=df.index)


def main():
    print("=" * 70)
    print("  ERROR ANALYSIS v2 — Fixed features + overfitting check")
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
    windows = get_attack_windows(y_test)

    print(f"\n  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows ({(y_test == 1).sum():,} attack)")
    print(f"  Attack windows: {len(windows)}")

    # Step 1: Compute raw features BEFORE scaling
    print("\nComputing raw features (contradiction, frozen, spike)...")
    train_raw_feat = compute_raw_features(train_df)
    test_raw_feat = compute_raw_features(test_df)
    print(f"  Raw features: {train_raw_feat.shape[1]}")

    # Check: do contradiction features fire in P21?
    p21_start, p21_end = windows[20]
    p21_raw = test_raw_feat.iloc[p21_start:p21_end + 1]
    contra_cols = [c for c in p21_raw.columns if "contra" in c]
    for c in contra_cols:
        if p21_raw[c].sum() > 0:
            print(f"  P21 signal: {c} = {p21_raw[c].sum():.0f} "
                  f"(max={p21_raw[c].max():.0f})")

    # Check P23/P25 frozen
    for pidx, pname in [(22, "P23"), (24, "P25")]:
        ps, pe = windows[pidx]
        pw = test_raw_feat.iloc[ps:pe + 1]
        frozen_cols = [c for c in pw.columns if "frozen" in c and pw[c].sum() > 0]
        if frozen_cols:
            print(f"  {pname} frozen sensors: {[c.replace('_frozen_raw','') for c in frozen_cols]}")

    # Check P28 constant sensor
    p28_start, p28_end = windows[27]
    p28_raw = test_raw_feat.iloc[p28_start:p28_end + 1]
    const_cols = [c for c in p28_raw.columns if "not_normal" in c or "abnormal" in c]
    for c in const_cols:
        if p28_raw[c].sum() > 0:
            print(f"  P28 signal: {c} = {p28_raw[c].sum():.0f}")

    # Step 2: Scale continuous sensors
    continuous = get_continuous_sensors(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous])
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[continuous] = scaler.transform(train_df[continuous])
    test_scaled[continuous] = scaler.transform(test_df[continuous])

    # Step 3: Generate standard features on scaled data (no filter)
    from src.preprocessing.feature_engineering import (
        compute_rolling_features, compute_rate_of_change, compute_deviation,
        compute_pair_residuals,
    )
    config = load_config()
    scaled_features = {}
    for sensor, info in config["continuous_sensors"].items():
        if sensor not in train_scaled.columns:
            continue
        series_train = train_scaled[sensor]
        features = info["features"]
        if "rolling_mean" in features or "rolling_std" in features:
            scaled_features.update(
                compute_rolling_features(series_train, sensor,
                                        config["window_sizes"], features)
            )
        if "rate_of_change" in features:
            scaled_features.update(compute_rate_of_change(series_train, sensor))
        if "deviation_from_baseline" in features:
            bl = config.get("baselines", {}).get(sensor, {})
            if bl:
                scaled_features.update(
                    compute_deviation(series_train, sensor, bl["mean"], bl["std"])
                )
    scaled_features.update(
        compute_pair_residuals(train_scaled, config.get("cross_sensor_pairs", []))
    )
    train_scaled_feat = pd.DataFrame(scaled_features, index=train_scaled.index)

    # Same for test
    scaled_features_test = {}
    for sensor, info in config["continuous_sensors"].items():
        if sensor not in test_scaled.columns:
            continue
        series_test = test_scaled[sensor]
        features = info["features"]
        if "rolling_mean" in features or "rolling_std" in features:
            scaled_features_test.update(
                compute_rolling_features(series_test, sensor,
                                        config["window_sizes"], features)
            )
        if "rate_of_change" in features:
            scaled_features_test.update(compute_rate_of_change(series_test, sensor))
        if "deviation_from_baseline" in features:
            bl = config.get("baselines", {}).get(sensor, {})
            if bl:
                scaled_features_test.update(
                    compute_deviation(series_test, sensor, bl["mean"], bl["std"])
                )
    scaled_features_test.update(
        compute_pair_residuals(test_scaled, config.get("cross_sensor_pairs", []))
    )
    test_scaled_feat = pd.DataFrame(scaled_features_test, index=test_scaled.index)

    # Step 4: Combine raw + scaled features + original sensors
    exclude = {"Timestamp", "label"}
    orig_cols = [c for c in train_scaled.columns if c not in exclude
                 and pd.api.types.is_numeric_dtype(train_scaled[c])]

    train_all = pd.concat([train_scaled[orig_cols], train_scaled_feat, train_raw_feat],
                          axis=1)
    test_all = pd.concat([test_scaled[orig_cols], test_scaled_feat, test_raw_feat],
                         axis=1)

    # Remove duplicate columns
    train_all = train_all.loc[:, ~train_all.columns.duplicated()]
    test_all = test_all.loc[:, ~test_all.columns.duplicated()]

    feature_cols = list(train_all.columns)
    print(f"\n  Total features: {len(feature_cols)}")

    X_train = np.nan_to_num(train_all.values.astype(np.float32),
                            nan=0, posinf=100, neginf=-100)
    X_test = np.nan_to_num(test_all.values.astype(np.float32),
                           nan=0, posinf=100, neginf=-100)

    # Normalize the combined features
    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    # --- Train SVM ---
    print("\nTraining SVM...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), 50_000, replace=False)
    svm = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm.fit(X_train[idx])
    svm_scores = -svm.decision_function(X_test)
    svm_pred = best_f1_predictions(y_test, svm_scores)

    # --- Train Autoencoder ---
    print("Training Autoencoder...")
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
    ae.fit(X_train, X_train, epochs=100, batch_size=256,
           validation_split=0.1, verbose=0)
    recon = ae.predict(X_test, verbose=0)
    ae_scores = np.mean((X_test - recon) ** 2, axis=1)
    ae_pred = best_f1_predictions(y_test, ae_scores)

    # --- OR Ensemble ---
    or_pred = ((svm_pred == 1) | (ae_pred == 1)).astype(int)
    or_pa = point_adjust(y_test, or_pred)

    f1_or = f1_score(y_test, or_pred, average="macro")
    f1_or_pa = f1_score(y_test, or_pa, average="macro")
    caught_or = sum(1 for s, e in windows if or_pred[s:e + 1].any())

    svm_pa = point_adjust(y_test, svm_pred)
    ae_pa = point_adjust(y_test, ae_pred)
    f1_svm = f1_score(y_test, svm_pred, average="macro")
    f1_svm_pa = f1_score(y_test, svm_pa, average="macro")
    f1_ae = f1_score(y_test, ae_pred, average="macro")
    f1_ae_pa = f1_score(y_test, ae_pa, average="macro")
    caught_svm = sum(1 for s, e in windows if svm_pred[s:e + 1].any())
    caught_ae = sum(1 for s, e in windows if ae_pred[s:e + 1].any())

    print(f"\n{'='*80}")
    print("  RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Model':<25} {'F1 Std':>8} {'F1 PA':>8} {'Caught':>10}")
    print("  " + "-" * 55)
    print(f"  {'SVM':<25} {f1_svm:>7.4f} {f1_svm_pa:>8.4f} {caught_svm:>5}/{len(windows)}")
    print(f"  {'Autoencoder':<25} {f1_ae:>7.4f} {f1_ae_pa:>8.4f} {caught_ae:>5}/{len(windows)}")
    print(f"  {'OR Ensemble':<25} {f1_or:>7.4f} {f1_or_pa:>8.4f} {caught_or:>5}/{len(windows)}")

    # --- Per-window ---
    print(f"\n  Per-window:")
    print(f"  {'Win':<6} {'Dur':>5} {'SVM':>8} {'AE':>8} {'OR':>8}")
    print("  " + "-" * 40)
    missed_indices = {10, 11, 14, 21, 23, 25, 28}
    for i, (s, e) in enumerate(windows):
        dur = e - s + 1
        sv = "CAUGHT" if svm_pred[s:e + 1].any() else "missed"
        av = "CAUGHT" if ae_pred[s:e + 1].any() else "missed"
        ov = "CAUGHT" if or_pred[s:e + 1].any() else "missed"
        marker = " <<<" if (i + 1) in missed_indices and ov == "CAUGHT" else ""
        marker2 = " ***" if (i + 1) in missed_indices and ov == "missed" else ""
        print(f"  P{i+1:<5} {dur:>5} {sv:>8} {av:>8} {ov:>8}{marker}{marker2}")

    # --- OVERFITTING CHECK ---
    print(f"\n{'='*80}")
    print("  OVERFITTING CHECK")
    print(f"{'='*80}")

    # SVM: score distribution on train-normal vs test-normal vs test-attack
    svm_train_scores = -svm.decision_function(X_train[idx])
    test_normal_mask = y_test == 0
    test_attack_mask = y_test == 1

    print(f"\n  SVM anomaly scores:")
    print(f"    Train normal:  mean={svm_train_scores.mean():.4f}  "
          f"std={svm_train_scores.std():.4f}  "
          f"p95={np.percentile(svm_train_scores, 95):.4f}")
    print(f"    Test normal:   mean={svm_scores[test_normal_mask].mean():.4f}  "
          f"std={svm_scores[test_normal_mask].std():.4f}  "
          f"p95={np.percentile(svm_scores[test_normal_mask], 95):.4f}")
    print(f"    Test attack:   mean={svm_scores[test_attack_mask].mean():.4f}  "
          f"std={svm_scores[test_attack_mask].std():.4f}  "
          f"p95={np.percentile(svm_scores[test_attack_mask], 95):.4f}")

    # AE: reconstruction error distribution
    recon_train = ae.predict(X_train[idx[:10000]], verbose=0)
    ae_train_scores = np.mean((X_train[idx[:10000]] - recon_train) ** 2, axis=1)

    print(f"\n  AE reconstruction error:")
    print(f"    Train normal:  mean={ae_train_scores.mean():.4f}  "
          f"std={ae_train_scores.std():.4f}  "
          f"p95={np.percentile(ae_train_scores, 95):.4f}")
    print(f"    Test normal:   mean={ae_scores[test_normal_mask].mean():.4f}  "
          f"std={ae_scores[test_normal_mask].std():.4f}  "
          f"p95={np.percentile(ae_scores[test_normal_mask], 95):.4f}")
    print(f"    Test attack:   mean={ae_scores[test_attack_mask].mean():.4f}  "
          f"std={ae_scores[test_attack_mask].std():.4f}  "
          f"p95={np.percentile(ae_scores[test_attack_mask], 95):.4f}")

    # FP rate on test-normal
    svm_fp = svm_pred[test_normal_mask].mean()
    ae_fp = ae_pred[test_normal_mask].mean()
    or_fp = or_pred[test_normal_mask].mean()
    print(f"\n  False positive rate (test-normal predicted as attack):")
    print(f"    SVM: {svm_fp:.4f} ({svm_pred[test_normal_mask].sum():,} / "
          f"{test_normal_mask.sum():,})")
    print(f"    AE:  {ae_fp:.4f} ({ae_pred[test_normal_mask].sum():,} / "
          f"{test_normal_mask.sum():,})")
    print(f"    OR:  {or_fp:.4f} ({or_pred[test_normal_mask].sum():,} / "
          f"{test_normal_mask.sum():,})")

    # Save
    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    results = {
        "model": ["SVM", "AE", "OR Ensemble"],
        "features": [n_feat] * 3,
        "f1_standard": [f1_svm, f1_ae, f1_or],
        "f1_point_adjust": [f1_svm_pa, f1_ae_pa, f1_or_pa],
        "windows_caught": [caught_svm, caught_ae, caught_or],
        "fp_rate": [svm_fp, ae_fp, or_fp],
    }
    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "metrics" / "error_analysis_v2.csv", index=False
    )
    print(f"\n  Saved: results/metrics/error_analysis_v2.csv")


if __name__ == "__main__":
    main()
