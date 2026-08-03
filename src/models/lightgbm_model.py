"""LightGBM supervised anomaly detector for SWaT dataset.

Trained on labeled data (normal + attack). Provides feature importance
for feature selection across all models.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

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


def train_model(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_val: pd.DataFrame, y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Train LightGBM with early stopping."""
    attack_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=attack_ratio,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    print(f"  Best iteration: {model.best_iteration_}")
    return model


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
) -> dict:
    """Compute evaluation metrics."""
    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"],
        labels=[0, 1], output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_attack": round(report["Attack"]["f1-score"], 4),
        "precision_attack": round(report["Attack"]["precision"], 4),
        "recall_attack": round(report["Attack"]["recall"], 4),
        "f1_normal": round(report["Normal"]["f1-score"], 4),
        "ap": round(average_precision_score(y_true, y_proba), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics."""
    print(f"\n{'=' * 55}")
    print("  LIGHTGBM — Evaluation Results")
    print(f"{'=' * 55}")
    print(f"  F1 Macro:           {metrics['f1_macro']}")
    print(f"  F1 Attack:          {metrics['f1_attack']}")
    print(f"  Precision (Attack): {metrics['precision_attack']}")
    print(f"  Recall (Attack):    {metrics['recall_attack']}")
    print(f"  Average Precision:  {metrics['ap']}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']:,}  FP={metrics['fp']:,}")
    print(f"    FN={metrics['fn']:,}  TN={metrics['tn']:,}")


def plot_feature_importance(
    model: lgb.LGBMClassifier, f1: float, top_n: int = 30,
) -> Path:
    """Plot top feature importances."""
    importance = pd.Series(
        model.feature_importances_, index=model.feature_name_,
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    top = importance.head(top_n)
    ax.barh(range(len(top)), top.values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (split count)")
    ax.set_title(f"LightGBM — top {top_n} features (F1 macro = {f1:.4f})")
    plt.tight_layout()

    out_path = RESULTS_DIR / "figures" / "lgbm_feature_importance.png"
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
    ax.set_title(f"LightGBM — Confusion Matrix\nF1 Macro = {metrics['f1_macro']}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=color, fontsize=14)

    plt.colorbar(im)
    plt.tight_layout()

    out_path = RESULTS_DIR / "figures" / "lgbm_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def get_feature_importance_df(model: lgb.LGBMClassifier) -> pd.DataFrame:
    """Return feature importance as sorted DataFrame."""
    return pd.DataFrame({
        "feature": model.feature_name_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def save_metrics(metrics: dict) -> Path:
    """Save metrics to CSV."""
    out_dir = RESULTS_DIR / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lightgbm.csv"
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    train_df, test_df = load_featured_data()
    feature_cols = get_feature_columns(train_df)

    print(f"Train: {len(train_df):,} rows  "
          f"(Normal: {(train_df['label']==0).sum():,}, "
          f"Attack: {(train_df['label']==1).sum():,})")
    print(f"Test:  {len(test_df):,} rows  "
          f"(Normal: {(test_df['label']==0).sum():,}, "
          f"Attack: {(test_df['label']==1).sum():,})")
    print(f"Features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df["label"].values
    X_test = test_df[feature_cols]
    y_test = test_df["label"].values

    print("\nTraining LightGBM...")
    model = train_model(X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba)
    print_metrics(metrics)

    print("\nSaving...")
    print(f"  {plot_feature_importance(model, metrics['f1_macro'])}")
    print(f"  {plot_confusion_matrix(metrics)}")

    imp_df = get_feature_importance_df(model)
    imp_path = RESULTS_DIR / "metrics" / "lgbm_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  {imp_path}")

    zero_imp = (imp_df["importance"] == 0).sum()
    print(f"\nFeatures with importance > 0: {len(imp_df) - zero_imp}")
    print(f"Features with importance = 0: {zero_imp}")

    save_metrics(metrics)

    model_path = MODELS_DIR / "lightgbm.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
