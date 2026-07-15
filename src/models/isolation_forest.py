"""Isolation Forest anomaly detector for SWaT dataset.

Trains on normal data only. Flags deviations as potential attacks.
Anomaly score = average isolation depth across random trees.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

N_ESTIMATORS = 200
CONTAMINATION = 0.01
RANDOM_STATE = 42


def load_featured_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature-engineered train and test datasets."""
    train = pd.read_parquet(PROCESSED_DIR / "train_featured.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_featured.parquet")
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude Timestamp and label)."""
    exclude = {"Timestamp", "label"}
    return [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def train_model(X_train: pd.DataFrame) -> IsolationForest:
    """Train Isolation Forest on normal data."""
    print(f"Training Isolation Forest...")
    print(f"  Samples: {len(X_train):,}, Features: {X_train.shape[1]}")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train)
    print("  Done.")
    return model


def predict_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Return anomaly scores (higher = more anomalous)."""
    return -model.score_samples(X)


def find_best_threshold(
    y_true: np.ndarray, scores: np.ndarray,
) -> tuple[float, dict]:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    return best_threshold, {
        "precision": precision[best_idx],
        "recall": recall[best_idx],
        "f1": f1_scores[best_idx],
    }


def evaluate(
    y_true: np.ndarray, scores: np.ndarray, threshold: float,
) -> dict:
    """Compute evaluation metrics at given threshold."""
    y_pred = (scores >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"],
        labels=[0, 1], output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": round(threshold, 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_attack": round(report["Attack"]["f1-score"], 4),
        "precision_attack": round(report["Attack"]["precision"], 4),
        "recall_attack": round(report["Attack"]["recall"], 4),
        "f1_normal": round(report["Normal"]["f1-score"], 4),
        "ap": round(average_precision_score(y_true, scores), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics."""
    print(f"\n{'=' * 55}")
    print("  ISOLATION FOREST — Evaluation Results")
    print(f"{'=' * 55}")
    print(f"  Threshold:          {metrics['threshold']}")
    print(f"  F1 Macro:           {metrics['f1_macro']}")
    print(f"  F1 Attack:          {metrics['f1_attack']}")
    print(f"  Precision (Attack): {metrics['precision_attack']}")
    print(f"  Recall (Attack):    {metrics['recall_attack']}")
    print(f"  Average Precision:  {metrics['ap']}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']:,}  FP={metrics['fp']:,}")
    print(f"    FN={metrics['fn']:,}  TN={metrics['tn']:,}")


def plot_scores_timeline(
    test_df: pd.DataFrame, scores: np.ndarray, threshold: float,
) -> Path:
    """Plot anomaly scores over time with attack periods highlighted."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    timestamps = test_df["Timestamp"]
    y_true = test_df["label"].values

    axes[0].plot(timestamps, scores, linewidth=0.3, color="steelblue", alpha=0.7)
    axes[0].axhline(y=threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
    axes[0].set_ylabel("Anomaly Score")
    axes[0].set_title("Isolation Forest — Anomaly Scores Over Time")
    axes[0].legend()

    attack_mask = y_true == 1
    for start, end in _get_contiguous_ranges(attack_mask):
        axes[0].axvspan(timestamps.iloc[start], timestamps.iloc[end],
                        alpha=0.15, color="red")

    pred_anomaly = scores >= threshold
    axes[1].fill_between(timestamps, 0, y_true, alpha=0.3, color="red", label="Actual Attack")
    axes[1].fill_between(timestamps, 0, pred_anomaly.astype(int),
                         alpha=0.3, color="blue", label="Predicted Anomaly")
    axes[1].set_ylabel("Attack / Anomaly")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].set_title("Actual Attacks vs Predictions")

    plt.tight_layout()
    out_path = RESULTS_DIR / "figures" / "if_scores_timeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrix(metrics: dict) -> Path:
    """Plot confusion matrix heatmap."""
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Normal", "Attack"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Isolation Forest — Confusion Matrix\nF1 Macro = {metrics['f1_macro']}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()

    out_path = RESULTS_DIR / "figures" / "if_confusion_matrix.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_precision_recall(y_true: np.ndarray, scores: np.ndarray) -> Path:
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"Isolation Forest (AP={ap:.4f})")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--",
               label=f"Random baseline ({y_true.mean():.3f})")
    ax.set_xlabel("Recall (fraction of attacks caught)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Isolation Forest")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    out_path = RESULTS_DIR / "figures" / "if_precision_recall.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _get_contiguous_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find start/end indices of contiguous True regions."""
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    return list(zip(starts, ends - 1))


def save_metrics(metrics: dict) -> Path:
    """Save metrics to CSV."""
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isolation_forest.csv"
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    train_df, test_df = load_featured_data()
    feature_cols = get_feature_columns(train_df)

    print(f"Train: {train_df.shape[0]:,} rows, {len(feature_cols)} features")
    print(f"Test:  {test_df.shape[0]:,} rows")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["label"].values

    model = train_model(X_train)

    print("\nScoring test data...")
    scores = predict_scores(model, X_test)

    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Score mean: {scores.mean():.4f}, median: {np.median(scores):.4f}")
    print(f"  Labels: {np.bincount(y_test)} (0=Normal, 1=Attack)")

    threshold, best_info = find_best_threshold(y_test, scores)
    print(f"\nBest threshold: {threshold:.4f} (F1={best_info['f1']:.4f})")

    metrics = evaluate(y_test, scores, threshold)
    print_metrics(metrics)

    print("\nSaving plots...")
    print(f"  {plot_scores_timeline(test_df, scores, threshold)}")
    print(f"  {plot_confusion_matrix(metrics)}")
    print(f"  {plot_precision_recall(y_test, scores)}")

    metrics_path = save_metrics(metrics)
    print(f"\nMetrics saved: {metrics_path}")

    model_path = MODELS_DIR / "isolation_forest.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
