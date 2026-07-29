"""Error analysis — test new features (frozen detection, spike detection)
on missed attack windows. Uses paper-style split."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_recall_curve,
)
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.load_data import clean_normal_data, get_sensor_columns
from src.preprocessing.feature_engineering import generate_features
from src.preprocessing.feature_config import load_config
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
SELECTED_PATH = ROOT / "config" / "selected_features.json"
RANDOM_STATE = 42


def best_f1_predictions(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    best_idx = np.argmax(f1)
    threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return (scores >= threshold).astype(int), threshold


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
    windows = get_attack_windows(y_true)
    for start, end in windows:
        if y_pred[start:end + 1].any():
            y_adjusted[start:end + 1] = 1
    return y_adjusted


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    out = RESULTS_DIR / "figures" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def load_and_prepare_data():
    """Paper-style split with all features (no feature selection filter)."""
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

    continuous = get_continuous_sensors(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous])
    train_df[continuous] = scaler.transform(train_df[continuous])
    test_df[continuous] = scaler.transform(test_df[continuous])

    return train_df, test_df


def generate_features_no_filter(df, config):
    """Generate features without applying selected_features.json filter."""
    # Temporarily rename the file so it's not applied
    backup = None
    if SELECTED_PATH.exists():
        backup = SELECTED_PATH.read_text()
        SELECTED_PATH.unlink()

    result = generate_features(df, config)

    if backup is not None:
        SELECTED_PATH.write_text(backup)

    return result


def main():
    print("=" * 70)
    print("  ERROR ANALYSIS — Testing new features on missed attacks")
    print("=" * 70)

    train_df, test_df = load_and_prepare_data()
    y_test = test_df["label"].values
    windows = get_attack_windows(y_test)

    print(f"\n  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows ({(y_test == 1).sum():,} attack)")
    print(f"  Attack windows: {len(windows)}")

    config = load_config()

    # --- Baseline: original 142 features ---
    print("\n--- BASELINE: Original 142 features ---")
    train_base = generate_features(train_df.copy(), config)
    test_base = generate_features(test_df.copy(), config)

    exclude = {"Timestamp", "label"}
    base_features = [c for c in train_base.columns if c not in exclude
                     and pd.api.types.is_numeric_dtype(train_base[c])]
    print(f"  Features: {len(base_features)}")

    X_train_base = train_base[base_features].values
    X_test_base = test_base[base_features].values

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train_base), 50_000, replace=False)

    svm_base = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm_base.fit(X_train_base[idx])
    scores_base = -svm_base.decision_function(X_test_base)
    pred_base, _ = best_f1_predictions(y_test, scores_base)
    pred_base_pa = point_adjust(y_test, pred_base)

    f1_base = f1_score(y_test, pred_base, average="macro")
    f1_base_pa = f1_score(y_test, pred_base_pa, average="macro")
    caught_base = sum(1 for s, e in windows if pred_base[s:e + 1].any())

    print(f"  F1 Macro: {f1_base:.4f} (standard), {f1_base_pa:.4f} (point-adjust)")
    print(f"  Windows caught: {caught_base}/{len(windows)}")

    # --- New: all features including frozen + spike ---
    print("\n--- NEW: Original + frozen detection + spike detection ---")
    train_new = generate_features_no_filter(train_df.copy(), config)
    test_new = generate_features_no_filter(test_df.copy(), config)

    new_features = [c for c in train_new.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(train_new[c])]

    # Show what's new
    new_only = [f for f in new_features if f not in base_features]
    frozen_feats = [f for f in new_only if "frozen" in f]
    spike_feats = [f for f in new_only if "spike" in f]
    other_new = [f for f in new_only if f not in frozen_feats and f not in spike_feats]
    print(f"  Total features: {len(new_features)}")
    print(f"  New frozen features: {len(frozen_feats)}")
    print(f"  New spike features: {len(spike_feats)}")
    if other_new:
        print(f"  Other new features: {len(other_new)}")

    # Handle inf/nan from spike ratios
    X_train_new = train_new[new_features].values
    X_test_new = test_new[new_features].values
    X_train_new = np.nan_to_num(X_train_new, nan=0, posinf=100, neginf=-100)
    X_test_new = np.nan_to_num(X_test_new, nan=0, posinf=100, neginf=-100)

    svm_new = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm_new.fit(X_train_new[idx])
    scores_new = -svm_new.decision_function(X_test_new)
    pred_new, _ = best_f1_predictions(y_test, scores_new)
    pred_new_pa = point_adjust(y_test, pred_new)

    f1_new = f1_score(y_test, pred_new, average="macro")
    f1_new_pa = f1_score(y_test, pred_new_pa, average="macro")
    caught_new = sum(1 for s, e in windows if pred_new[s:e + 1].any())

    print(f"  F1 Macro: {f1_new:.4f} (standard), {f1_new_pa:.4f} (point-adjust)")
    print(f"  Windows caught: {caught_new}/{len(windows)}")

    # --- Also try Autoencoder with new features ---
    print("\n--- NEW: Autoencoder with all features ---")
    ae = keras.Sequential([
        layers.Input(shape=(len(new_features),)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(len(new_features), activation="linear"),
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_train_new, X_train_new, epochs=100, batch_size=256,
           validation_split=0.1, verbose=0)
    recon = ae.predict(X_test_new, verbose=0)
    ae_scores = np.mean((X_test_new - recon) ** 2, axis=1)
    ae_pred, _ = best_f1_predictions(y_test, ae_scores)
    ae_pred_pa = point_adjust(y_test, ae_pred)

    f1_ae = f1_score(y_test, ae_pred, average="macro")
    f1_ae_pa = f1_score(y_test, ae_pred_pa, average="macro")
    caught_ae = sum(1 for s, e in windows if ae_pred[s:e + 1].any())

    print(f"  F1 Macro: {f1_ae:.4f} (standard), {f1_ae_pa:.4f} (point-adjust)")
    print(f"  Windows caught: {caught_ae}/{len(windows)}")

    # --- OR Ensemble with new features ---
    or_pred = ((pred_new == 1) | (ae_pred == 1)).astype(int)
    or_pred_pa = point_adjust(y_test, or_pred)
    f1_or = f1_score(y_test, or_pred, average="macro")
    f1_or_pa = f1_score(y_test, or_pred_pa, average="macro")
    caught_or = sum(1 for s, e in windows if or_pred[s:e + 1].any())

    # --- Summary ---
    print(f"\n{'='*80}")
    print("  SUMMARY — BASELINE vs NEW FEATURES")
    print(f"{'='*80}")
    print(f"\n  {'Model':<30} {'F1 Std':>8} {'F1 PA':>8} {'Caught':>10}")
    print("  " + "-" * 60)
    print(f"  {'Baseline SVM (142 feat)':<30} {f1_base:>7.4f} {f1_base_pa:>8.4f} "
          f"{caught_base:>5}/{len(windows)}")
    print(f"  {'New SVM (all feat)':<30} {f1_new:>7.4f} {f1_new_pa:>8.4f} "
          f"{caught_new:>5}/{len(windows)}")
    print(f"  {'New AE (all feat)':<30} {f1_ae:>7.4f} {f1_ae_pa:>8.4f} "
          f"{caught_ae:>5}/{len(windows)}")
    print(f"  {'New OR Ensemble':<30} {f1_or:>7.4f} {f1_or_pa:>8.4f} "
          f"{caught_or:>5}/{len(windows)}")

    # --- Per-window comparison ---
    print(f"\n  Per-window detection (NEW vs BASELINE):")
    print(f"  {'Window':<8} {'Dur':>6} {'Base':>8} {'SVM-new':>8} {'AE-new':>8} {'OR-new':>8}")
    print("  " + "-" * 48)
    for i, (s, e) in enumerate(windows):
        dur = e - s + 1
        b = "CAUGHT" if pred_base[s:e + 1].any() else "missed"
        sn = "CAUGHT" if pred_new[s:e + 1].any() else "missed"
        an = "CAUGHT" if ae_pred[s:e + 1].any() else "missed"
        on = "CAUGHT" if or_pred[s:e + 1].any() else "missed"
        marker = " <<<" if b == "missed" and on == "CAUGHT" else ""
        print(f"  P{i+1:<6} {dur:>6} {b:>8} {sn:>8} {an:>8} {on:>8}{marker}")

    # --- Save confusion matrix ---
    save_confusion_matrix(y_test, or_pred_pa,
                          "OR Ensemble + Error Analysis Features (Point-Adjust)",
                          "error_analysis_ensemble_cm.png")

    # --- Save results ---
    results = [
        {"model": "Baseline SVM", "features": len(base_features),
         "f1_standard": f1_base, "f1_point_adjust": f1_base_pa,
         "windows_caught": caught_base},
        {"model": "New SVM", "features": len(new_features),
         "f1_standard": f1_new, "f1_point_adjust": f1_new_pa,
         "windows_caught": caught_new},
        {"model": "New AE", "features": len(new_features),
         "f1_standard": f1_ae, "f1_point_adjust": f1_ae_pa,
         "windows_caught": caught_ae},
        {"model": "New OR Ensemble", "features": len(new_features),
         "f1_standard": f1_or, "f1_point_adjust": f1_or_pa,
         "windows_caught": caught_or},
    ]
    out = RESULTS_DIR / "metrics" / "error_analysis_comparison.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
