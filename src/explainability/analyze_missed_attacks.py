"""Analyze missed attack windows — what happens physically in each one.

Uses RAW (unscaled) sensor values and train-period statistics for
interpretable deviation measurements.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve

from src.preprocessing.load_data import clean_normal_data, get_sensor_columns
from src.preprocessing.feature_engineering import generate_features
from src.preprocessing.feature_config import load_config
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


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


def analyze_window(raw_window, train_stats, window_idx, start, end, y_pred,
                   continuous_cols, discrete_cols):
    """Analyze a single attack window using RAW sensor values."""
    duration = end - start + 1
    detected = y_pred[start:end + 1].sum()
    detection_rate = detected / duration

    result = {
        "window": window_idx + 1,
        "duration_sec": duration,
        "detection_rate": round(detection_rate, 3),
        "detected_points": int(detected),
        "status": "CAUGHT" if detected > 0 else "MISSED",
    }

    sensor_deviations = {}
    for col in continuous_cols:
        if col not in raw_window.columns:
            continue
        normal_mean = train_stats[col]["mean"]
        normal_std = train_stats[col]["std"]
        if normal_std > 0:
            vals = raw_window[col].values
            max_dev = np.max(np.abs(vals - normal_mean)) / normal_std
            mean_dev = np.mean(np.abs(vals - normal_mean)) / normal_std
            sensor_deviations[col] = {
                "max_sigma": round(float(max_dev), 2),
                "mean_sigma": round(float(mean_dev), 2),
            }

    top_deviating = sorted(sensor_deviations.items(),
                           key=lambda x: x[1]["max_sigma"], reverse=True)[:5]
    result["top_sensors"] = top_deviating

    discrete_changes = {}
    for col in discrete_cols:
        if col not in raw_window.columns:
            continue
        vals = raw_window[col].values
        unique = np.unique(vals)
        if len(unique) > 1:
            changes = int(np.sum(np.diff(vals) != 0))
            discrete_changes[col] = {"states": unique.tolist(), "changes": changes}
    result["discrete_changes"] = discrete_changes

    frozen = []
    for col in continuous_cols:
        if col not in raw_window.columns:
            continue
        if raw_window[col].std() == 0 and duration > 10:
            frozen.append(col)
    result["frozen_sensors"] = frozen

    return result


def main():
    print("Loading data (paper-style split)...")
    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = clean_normal_data(normal)

    train_df = normal.copy()
    attack_start = attack["Timestamp"].min()
    attack_end = attack["Timestamp"].max()
    normal_in_test = normal[
        (normal["Timestamp"] >= attack_start) & (normal["Timestamp"] <= attack_end)
    ].copy()
    test_raw = pd.concat([normal_in_test, attack], ignore_index=True)
    test_raw = test_raw.sort_values("Timestamp").reset_index(drop=True)

    continuous_cols = get_continuous_sensors(train_df)
    discrete_cols = [c for c in get_sensor_columns(train_df)
                     if c not in continuous_cols]

    # Stats on RAW training data — for interpretable deviations
    train_stats = {col: {"mean": train_df[col].mean(), "std": train_df[col].std()}
                   for col in continuous_cols}

    # Scale for model training
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])
    train_scaled = train_df.copy()
    test_scaled = test_raw.copy()
    train_scaled[continuous_cols] = scaler.transform(train_df[continuous_cols])
    test_scaled[continuous_cols] = scaler.transform(test_raw[continuous_cols])

    print("Generating features...")
    config = load_config()
    train_feat = generate_features(train_scaled, config)
    test_feat = generate_features(test_scaled, config)

    exclude = {"Timestamp", "label"}
    feature_cols = [c for c in train_feat.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(train_feat[c])]

    X_train = train_feat[feature_cols].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat["label"].values

    print("Training SVM...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), 50_000, replace=False)
    svm = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm.fit(X_train[idx])
    svm_scores = -svm.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, svm_scores)
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1)
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    y_pred = (svm_scores >= threshold).astype(int)

    windows = get_attack_windows(y_test)
    print(f"\nTotal attack windows: {len(windows)}")

    missed = []
    caught = []
    for i, (start, end) in enumerate(windows):
        analysis = analyze_window(
            test_raw.iloc[start:end + 1], train_stats, i, start, end,
            y_pred, continuous_cols, discrete_cols,
        )
        if analysis["status"] == "MISSED":
            missed.append(analysis)
        else:
            caught.append(analysis)

    print(f"Caught: {len(caught)}, Missed: {len(missed)}")

    # --- Print missed attacks ---
    print(f"\n{'='*80}")
    print("  MISSED ATTACK WINDOWS — DETAILED ANALYSIS")
    print(f"{'='*80}")

    for a in missed:
        print(f"\n  Window P{a['window']} — {a['duration_sec']}s — MISSED")
        print(f"  {'─'*60}")

        print(f"  Top deviating sensors (σ from training normal):")
        for sensor, dev in a["top_sensors"]:
            print(f"    {sensor:<12} max={dev['max_sigma']:.1f}σ  "
                  f"mean={dev['mean_sigma']:.1f}σ")

        if a["discrete_changes"]:
            print(f"  Discrete sensor changes:")
            for sensor, info in a["discrete_changes"].items():
                print(f"    {sensor}: states={info['states']}, "
                      f"changes={info['changes']}")

        if a["frozen_sensors"]:
            print(f"  Frozen sensors: {', '.join(a['frozen_sensors'])}")

    # --- Comparison ---
    print(f"\n{'='*80}")
    print("  MISSED vs CAUGHT — STATISTICAL COMPARISON")
    print(f"{'='*80}")

    missed_max = [a["top_sensors"][0][1]["max_sigma"] for a in missed if a["top_sensors"]]
    caught_max = [a["top_sensors"][0][1]["max_sigma"] for a in caught if a["top_sensors"]]

    print(f"\n  Top sensor max deviation (σ):")
    print(f"    Missed ({len(missed)}): median={np.median(missed_max):.1f}, "
          f"range=[{min(missed_max):.1f}, {max(missed_max):.1f}]")
    print(f"    Caught ({len(caught)}): median={np.median(caught_max):.1f}, "
          f"range=[{min(caught_max):.1f}, {max(caught_max):.1f}]")

    missed_disc = sum(1 for a in missed if a["discrete_changes"])
    caught_disc = sum(1 for a in caught if a["discrete_changes"])
    missed_froz = sum(1 for a in missed if a["frozen_sensors"])
    caught_froz = sum(1 for a in caught if a["frozen_sensors"])

    print(f"\n  With discrete changes: missed={missed_disc}/{len(missed)}, "
          f"caught={caught_disc}/{len(caught)}")
    print(f"  With frozen sensors: missed={missed_froz}/{len(missed)}, "
          f"caught={caught_froz}/{len(caught)}")

    # Duration comparison
    missed_dur = [a["duration_sec"] for a in missed]
    caught_dur = [a["duration_sec"] for a in caught]
    print(f"\n  Duration (seconds):")
    print(f"    Missed: median={int(np.median(missed_dur))}, "
          f"range=[{min(missed_dur)}, {max(missed_dur)}]")
    print(f"    Caught: median={int(np.median(caught_dur))}, "
          f"range=[{min(caught_dur)}, {max(caught_dur)}]")

    # --- All sensors that appear in missed but not caught ---
    missed_sensors = set()
    for a in missed:
        for s, _ in a["top_sensors"]:
            missed_sensors.add(s)
    caught_sensors = set()
    for a in caught:
        for s, _ in a["top_sensors"]:
            caught_sensors.add(s)
    print(f"\n  Sensors unique to missed top-5: {missed_sensors - caught_sensors}")
    print(f"  Sensors unique to caught top-5: {caught_sensors - missed_sensors}")

    # --- Feature recommendations ---
    print(f"\n{'='*80}")
    print("  FEATURE RECOMMENDATIONS BASED ON ANALYSIS")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
