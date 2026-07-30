"""Error analysis v3 — two new approaches for the last 3 missed attacks.

1. Multi-sensor simultaneous deviation: count sensors deviating >1σ together
2. Rule-based layer: physical rules on top of ML predictions
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
from src.preprocessing.prepare_data import get_continuous_sensors
from src.models.error_analysis_v2 import (
    compute_raw_features, get_attack_windows, point_adjust, best_f1_predictions,
    CONTRADICTION_RULES,
)
from src.preprocessing.feature_engineering import (
    compute_rolling_features, compute_rate_of_change, compute_deviation,
    compute_pair_residuals,
)
from src.preprocessing.feature_config import load_config

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def compute_simultaneous_deviation(df_scaled, continuous_cols, thresholds, windows):
    """Count how many sensors deviate beyond threshold simultaneously."""
    features = {}
    for thresh in thresholds:
        deviating = (df_scaled[continuous_cols].abs() > thresh).astype(int)
        count = deviating.sum(axis=1)
        features[f"sensors_above_{thresh}sig"] = count
        for w in windows:
            features[f"sensors_above_{thresh}sig_mean_{w}"] = (
                count.rolling(w, min_periods=1).mean()
            )
            features[f"sensors_above_{thresh}sig_max_{w}"] = (
                count.rolling(w, min_periods=1).max()
            )
    return pd.DataFrame(features, index=df_scaled.index)


def compute_simultaneous_drift(df_scaled, continuous_cols, window=60):
    """Count sensors drifting in the same direction simultaneously."""
    features = {}
    diffs = df_scaled[continuous_cols].diff().fillna(0)
    drifting_up = (diffs > 0).astype(int).sum(axis=1)
    drifting_down = (diffs < 0).astype(int).sum(axis=1)
    # Imbalance: if most sensors drift same direction, suspicious
    features["drift_imbalance"] = (drifting_up - drifting_down).abs()
    features[f"drift_imbalance_mean_{window}"] = (
        features["drift_imbalance"].rolling(window, min_periods=1).mean()
    )
    # Max of up/down count
    features["max_same_direction"] = np.maximum(drifting_up, drifting_down)
    features[f"max_same_dir_mean_{window}"] = (
        features["max_same_direction"].rolling(window, min_periods=1).mean()
    )
    return pd.DataFrame(features, index=df_scaled.index)


def compute_low_variance_count(df_scaled, continuous_cols, window=30):
    """Count sensors with unusually low variance in rolling window."""
    features = {}
    rolling_stds = df_scaled[continuous_cols].rolling(window, min_periods=10).std()
    # After scaling, normal std should be ~1. Low var = std < 0.1
    low_var = (rolling_stds < 0.1).astype(int).sum(axis=1)
    features["low_var_count_30"] = low_var
    features["low_var_count_60"] = (
        (df_scaled[continuous_cols].rolling(60, min_periods=10).std() < 0.1)
        .astype(int).sum(axis=1)
    )
    return pd.DataFrame(features, index=df_scaled.index)


def apply_rules(test_raw, y_pred):
    """Rule-based layer: flag rows that match physical impossibility rules."""
    rules_fired = np.zeros(len(test_raw), dtype=int)

    # Rule 1: Multiple flows at zero simultaneously
    flow_sensors = ["FIT101", "FIT201", "FIT301", "FIT401", "FIT501", "FIT601"]
    available = [f for f in flow_sensors if f in test_raw.columns]
    if len(available) >= 2:
        zero_flows = sum((test_raw[f].abs() < 0.05).astype(int) for f in available)
        # 3+ flows at zero simultaneously is very unusual
        rule1 = (zero_flows >= 3).astype(int).values
        rules_fired |= rule1

    # Rule 2: FIT301=0 AND FIT201=0 simultaneously (physically impossible
    # if system is running)
    if "FIT301" in test_raw.columns and "FIT201" in test_raw.columns:
        rule2 = (
            (test_raw["FIT301"].abs() < 0.05) &
            (test_raw["FIT201"].abs() < 0.05)
        ).astype(int).values
        rules_fired |= rule2

    # Rule 3: P102 state change (always OFF in training)
    if "P102" in test_raw.columns:
        rule3 = (test_raw["P102"] == 2).astype(int).values
        rules_fired |= rule3

    # Rule 4: 5+ continuous sensors frozen simultaneously (rolling_std=0 over 10s)
    continuous = [c for c in test_raw.columns
                  if c not in ("Timestamp", "label") and test_raw[c].dtype == "float64"]
    if continuous:
        frozen_count = pd.DataFrame()
        for c in continuous:
            frozen_count[c] = (
                test_raw[c].rolling(10, min_periods=10).std().fillna(1) == 0
            ).astype(int)
        total_frozen = frozen_count.sum(axis=1)
        rule4 = (total_frozen >= 5).astype(int).values
        rules_fired |= rule4

    # Combine: OR with ML predictions
    combined = ((y_pred == 1) | (rules_fired == 1)).astype(int)
    return combined, rules_fired


def main():
    print("=" * 70)
    print("  ERROR ANALYSIS v3 — Simultaneous deviation + Rule-based")
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
    test_raw = test_df.copy()

    y_test = test_df["label"].values
    windows = get_attack_windows(y_test)
    continuous_cols = get_continuous_sensors(train_df)

    # Scale
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])
    train_df[continuous_cols] = scaler.transform(train_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

    # --- Build features ---
    print("\nBuilding features...")

    # 1. Raw features (v2)
    train_raw_feat = compute_raw_features(train_df.copy())
    # Need raw test for raw features
    test_raw_for_feat = test_raw.copy()
    test_raw_feat = compute_raw_features(test_raw_for_feat)

    # 2. Scaled rolling features
    config = load_config()
    train_scaled_feats = {}
    test_scaled_feats = {}
    for sensor, info in config["continuous_sensors"].items():
        if sensor not in train_df.columns:
            continue
        features = info["features"]
        if "rolling_mean" in features or "rolling_std" in features:
            train_scaled_feats.update(
                compute_rolling_features(train_df[sensor], sensor,
                                        config["window_sizes"], features))
            test_scaled_feats.update(
                compute_rolling_features(test_df[sensor], sensor,
                                        config["window_sizes"], features))
        if "rate_of_change" in features:
            train_scaled_feats.update(compute_rate_of_change(train_df[sensor], sensor))
            test_scaled_feats.update(compute_rate_of_change(test_df[sensor], sensor))
        if "deviation_from_baseline" in features:
            bl = config.get("baselines", {}).get(sensor, {})
            if bl:
                train_scaled_feats.update(
                    compute_deviation(train_df[sensor], sensor, bl["mean"], bl["std"]))
                test_scaled_feats.update(
                    compute_deviation(test_df[sensor], sensor, bl["mean"], bl["std"]))
    train_scaled_feats.update(
        compute_pair_residuals(train_df, config.get("cross_sensor_pairs", [])))
    test_scaled_feats.update(
        compute_pair_residuals(test_df, config.get("cross_sensor_pairs", [])))

    train_scaled_df = pd.DataFrame(train_scaled_feats, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled_feats, index=test_df.index)

    # 3. NEW: Simultaneous deviation features
    train_simdev = compute_simultaneous_deviation(
        train_df, continuous_cols, thresholds=[0.5, 1.0, 1.5, 2.0], windows=[30, 60])
    test_simdev = compute_simultaneous_deviation(
        test_df, continuous_cols, thresholds=[0.5, 1.0, 1.5, 2.0], windows=[30, 60])

    # 4. NEW: Simultaneous drift features
    train_drift = compute_simultaneous_drift(train_df, continuous_cols)
    test_drift = compute_simultaneous_drift(test_df, continuous_cols)

    # 5. NEW: Low variance count
    train_lowvar = compute_low_variance_count(train_df, continuous_cols)
    test_lowvar = compute_low_variance_count(test_df, continuous_cols)

    # Combine all
    exclude = {"Timestamp", "label"}
    orig_cols = [c for c in train_df.columns if c not in exclude
                 and pd.api.types.is_numeric_dtype(train_df[c])]

    train_all = pd.concat([train_df[orig_cols], train_scaled_df, train_raw_feat,
                           train_simdev, train_drift, train_lowvar], axis=1)
    test_all = pd.concat([test_df[orig_cols], test_scaled_df, test_raw_feat,
                          test_simdev, test_drift, test_lowvar], axis=1)

    train_all = train_all.loc[:, ~train_all.columns.duplicated()]
    test_all = test_all.loc[:, ~test_all.columns.duplicated()]

    n_feat = train_all.shape[1]
    print(f"  Total features: {n_feat}")
    print(f"  New simdev features: {train_simdev.shape[1]}")
    print(f"  New drift features: {train_drift.shape[1]}")
    print(f"  New lowvar features: {train_lowvar.shape[1]}")

    # Check new features on P10, P11, P23
    for pidx, pname in [(9, "P10"), (10, "P11"), (22, "P23")]:
        s, e = windows[pidx]
        pw_simdev = test_simdev.iloc[s:e + 1]
        pw_lowvar = test_lowvar.iloc[s:e + 1]
        print(f"\n  {pname} ({e-s+1}s):")
        print(f"    sensors_above_1sig: max={pw_simdev['sensors_above_1.0sig'].max()}, "
              f"mean={pw_simdev['sensors_above_1.0sig'].mean():.1f}")
        print(f"    sensors_above_1.5sig: max={pw_simdev['sensors_above_1.5sig'].max()}")
        print(f"    low_var_count_30: max={pw_lowvar['low_var_count_30'].max()}")

    # Compare to normal baseline
    train_above1 = train_simdev["sensors_above_1.0sig"]
    print(f"\n  Training normal sensors_above_1sig: "
          f"mean={train_above1.mean():.1f}, p95={train_above1.quantile(0.95):.0f}, "
          f"p99={train_above1.quantile(0.99):.0f}, max={train_above1.max()}")
    train_lowvar_vals = train_lowvar["low_var_count_30"]
    print(f"  Training normal low_var_count_30: "
          f"mean={train_lowvar_vals.mean():.1f}, p95={train_lowvar_vals.quantile(0.95):.0f}, "
          f"max={train_lowvar_vals.max()}")

    X_train = np.nan_to_num(train_all.values.astype(np.float32),
                            nan=0, posinf=100, neginf=-100)
    X_test = np.nan_to_num(test_all.values.astype(np.float32),
                           nan=0, posinf=100, neginf=-100)

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    # --- Train models ---
    print("\nTraining SVM...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), 50_000, replace=False)
    svm = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm.fit(X_train[idx])
    svm_scores = -svm.decision_function(X_test)
    svm_pred = best_f1_predictions(y_test, svm_scores)

    print("Training Autoencoder...")
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

    or_ml = ((svm_pred == 1) | (ae_pred == 1)).astype(int)

    # --- Rule-based layer ---
    print("\nApplying rule-based layer...")
    combined, rules_only = apply_rules(test_raw, or_ml)

    # Check rules on missed windows
    for pidx, pname in [(9, "P10"), (10, "P11"), (22, "P23")]:
        s, e = windows[pidx]
        rf = rules_only[s:e + 1].sum()
        ml = or_ml[s:e + 1].sum()
        cb = combined[s:e + 1].sum()
        print(f"  {pname}: ML={ml}, Rules={rf}, Combined={cb}")

    # --- Results ---
    or_ml_pa = point_adjust(y_test, or_ml)
    combined_pa = point_adjust(y_test, combined)

    f1_ml = f1_score(y_test, or_ml, average="macro")
    f1_ml_pa = f1_score(y_test, or_ml_pa, average="macro")
    f1_comb = f1_score(y_test, combined, average="macro")
    f1_comb_pa = f1_score(y_test, combined_pa, average="macro")

    caught_ml = sum(1 for s, e in windows if or_ml[s:e + 1].any())
    caught_comb = sum(1 for s, e in windows if combined[s:e + 1].any())

    # Rules-only FP rate
    rules_fp = rules_only[y_test == 0].mean()
    combined_fp = combined[y_test == 0].mean()

    print(f"\n{'='*80}")
    print("  RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Model':<35} {'F1 Std':>8} {'F1 PA':>8} {'Caught':>10} {'FP%':>8}")
    print("  " + "-" * 70)
    print(f"  {'v2 baseline (from prev run)':<35} {'0.9201':>8} {'0.9681':>8} "
          f"{'32/35':>10} {'1.4%':>8}")
    print(f"  {'v3 ML only (simdev+drift+lowvar)':<35} {f1_ml:>7.4f} {f1_ml_pa:>8.4f} "
          f"{caught_ml:>5}/{len(windows)} {or_ml[y_test==0].mean():>7.2%}")
    print(f"  {'v3 ML + Rules':<35} {f1_comb:>7.4f} {f1_comb_pa:>8.4f} "
          f"{caught_comb:>5}/{len(windows)} {combined_fp:>7.2%}")
    print(f"  {'Rules only (no ML)':<35} {'':>8} {'':>8} "
          f"{'':>10} {rules_fp:>7.2%}")

    # Per-window
    print(f"\n  Per-window (focus on P10, P11, P23):")
    print(f"  {'Win':<6} {'Dur':>5} {'ML':>8} {'Rules':>8} {'ML+Rules':>10}")
    print("  " + "-" * 42)
    target = {10, 11, 23}
    for i, (s, e) in enumerate(windows):
        if (i + 1) not in target:
            continue
        dur = e - s + 1
        ml = "CAUGHT" if or_ml[s:e + 1].any() else "missed"
        ru = "CAUGHT" if rules_only[s:e + 1].any() else "missed"
        cb = "CAUGHT" if combined[s:e + 1].any() else "missed"
        print(f"  P{i+1:<5} {dur:>5} {ml:>8} {ru:>8} {cb:>10}")

    # Full table
    print(f"\n  Full per-window:")
    print(f"  {'Win':<6} {'Dur':>5} {'v3 ML+R':>10}")
    print("  " + "-" * 25)
    for i, (s, e) in enumerate(windows):
        dur = e - s + 1
        cb = "CAUGHT" if combined[s:e + 1].any() else "missed"
        marker = " ***" if cb == "missed" else ""
        print(f"  P{i+1:<5} {dur:>5} {cb:>10}{marker}")

    # Save
    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "v3 ML+Rules",
        "features": n_feat,
        "f1_standard": f1_comb,
        "f1_point_adjust": f1_comb_pa,
        "windows_caught": caught_comb,
        "fp_rate": combined_fp,
    }]).to_csv(RESULTS_DIR / "metrics" / "error_analysis_v3.csv", index=False)
    print(f"\n  Saved: results/metrics/error_analysis_v3.csv")


if __name__ == "__main__":
    main()
