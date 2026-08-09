"""Evaluate with paper-style split: train on normal before attacks, test on attack period.

Most published SWaT results use this split — train on pure normal data,
test on the full attack period (which includes both normal and attack rows).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tensorflow import keras
from tensorflow.keras import layers

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing.feature_config import load_config
from src.preprocessing.feature_engineering import generate_features
from src.preprocessing.load_data import clean_normal_data
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
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


def calc_metrics(y_true, y_pred, label=""):
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=["Normal", "Attack"],
                                   labels=[0, 1], output_dict=True, zero_division=0)
    return {
        "method": label,
        "f1_macro": round(f1_macro, 4),
        "precision": round(report["Attack"]["precision"], 4),
        "recall": round(report["Attack"]["recall"], 4),
        "f1_attack": round(report["Attack"]["f1-score"], 4),
    }


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


def main():
    print("=" * 70)
    print("  PAPER-STYLE SPLIT EVALUATION")
    print("  Train: normal data before attack period")
    print("  Test: full attack period (normal + attack rows)")
    print("=" * 70)

    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = clean_normal_data(normal)

    # Paper split: train on ALL normal data, test on the attack period
    # (attack period = normal rows during overlap + attack rows)
    train_df = normal.copy()

    attack_start = attack["Timestamp"].min()
    attack_end = attack["Timestamp"].max()
    normal_in_test = normal[
        (normal["Timestamp"] >= attack_start) & (normal["Timestamp"] <= attack_end)
    ].copy()
    test_df = pd.concat([normal_in_test, attack], ignore_index=True)
    test_df = test_df.sort_values("Timestamp").reset_index(drop=True)

    print(f"\n  Train: {len(train_df):,} rows (all normal)")
    print(f"  Test:  {len(test_df):,} rows "
          f"({(test_df['label'] == 0).sum():,} normal, "
          f"{(test_df['label'] == 1).sum():,} attack)")
    print(f"  Attack windows: {len(get_attack_windows(test_df['label'].values))}")

    continuous = get_continuous_sensors(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous])
    train_df[continuous] = scaler.transform(train_df[continuous])
    test_df[continuous] = scaler.transform(test_df[continuous])

    print("\nGenerating features...")
    config = load_config()
    train_feat = generate_features(train_df, config)
    test_feat = generate_features(test_df, config)

    exclude = {"Timestamp", "label"}
    feature_cols = [c for c in train_feat.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(train_feat[c])]
    print(f"  Features: {len(feature_cols)}")

    X_train = train_feat[feature_cols].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat["label"].values

    # --- SVM ---
    print("\nTraining SVM (nu=0.001, gamma=0.005, 50K samples)...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), min(50_000, len(X_train)), replace=False)
    svm = OneClassSVM(kernel="rbf", gamma=0.005, nu=0.001)
    svm.fit(X_train[idx])
    svm_scores = -svm.decision_function(X_test)
    svm_pred, _ = best_f1_predictions(y_test, svm_scores)

    # --- Autoencoder ---
    print("Training Autoencoder (bottleneck=8, 100 epochs)...")
    n_features = len(feature_cols)
    ae = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation="linear"),
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_train, X_train, epochs=100, batch_size=256,
           validation_split=0.1, verbose=0)
    recon = ae.predict(X_test, verbose=0)
    ae_scores = np.mean((X_test - recon) ** 2, axis=1)
    ae_pred, _ = best_f1_predictions(y_test, ae_scores)

    # --- OR Ensemble ---
    or_pred = ((svm_pred == 1) | (ae_pred == 1)).astype(int)

    # --- Point-adjust ---
    svm_pa = point_adjust(y_test, svm_pred)
    ae_pa = point_adjust(y_test, ae_pred)
    or_pa = point_adjust(y_test, or_pred)

    # --- Results ---
    results = [
        calc_metrics(y_test, svm_pred, "SVM (standard)"),
        calc_metrics(y_test, svm_pa, "SVM (point-adjust)"),
        calc_metrics(y_test, ae_pred, "AE (standard)"),
        calc_metrics(y_test, ae_pa, "AE (point-adjust)"),
        calc_metrics(y_test, or_pred, "OR Ensemble (standard)"),
        calc_metrics(y_test, or_pa, "OR Ensemble (point-adjust)"),
    ]

    print(f"\n{'='*80}")
    print("  RESULTS — Paper-Style Split")
    print(f"{'='*80}")
    print(f"\n{'Method':<30} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10} {'F1 Attack':>10}")
    print("-" * 72)
    for r in results:
        print(f"  {r['method']:<28} {r['f1_macro']:>9.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1_attack']:>10.4f}")

    # --- Per-attack window analysis ---
    windows = get_attack_windows(y_test)
    caught_svm = sum(1 for s, e in windows if svm_pred[s:e + 1].any())
    caught_ae = sum(1 for s, e in windows if ae_pred[s:e + 1].any())
    caught_or = sum(1 for s, e in windows if or_pred[s:e + 1].any())
    print(f"\n  Attack windows caught: SVM {caught_svm}/{len(windows)}, "
          f"AE {caught_ae}/{len(windows)}, OR {caught_or}/{len(windows)}")

    # --- Comparison with our split ---
    print(f"\n{'='*80}")
    print("  COMPARISON: Our Split vs Paper Split (Point-Adjust)")
    print(f"{'='*80}")
    print(f"\n  {'Model':<20} {'Our Split':>12} {'Paper Split':>12} {'Diff':>10}")
    print("  " + "-" * 56)
    our_results = {"SVM": 0.9000, "AE": 0.8703, "OR Ensemble": 0.8985}
    pa_results = {r["method"].replace(" (point-adjust)", ""): r["f1_macro"]
                  for r in results if "point-adjust" in r["method"]}
    for model in ["SVM", "AE", "OR Ensemble"]:
        our = our_results.get(model, 0)
        paper = pa_results.get(model, 0)
        diff = paper - our
        print(f"  {model:<20} {our:>11.4f} {paper:>12.4f} {diff:>+10.4f}")

    # --- Confusion matrices ---
    save_confusion_matrix(y_test, svm_pa, "SVM — Paper Split (Point-Adjust)",
                          "paper_split_svm_cm.png")
    save_confusion_matrix(y_test, or_pa, "OR Ensemble — Paper Split (Point-Adjust)",
                          "paper_split_ensemble_cm.png")

    # --- Save ---
    RESULTS_DIR.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "metrics" / "paper_split_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
