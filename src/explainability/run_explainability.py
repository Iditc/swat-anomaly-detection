"""Run full explainability pipeline on all attack windows.

Produces:
  - Per-window attack classification + evidence
  - Stacked bar chart: feature category contribution per attack window
  - Summary CSV with all results
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from src.explainability.attack_classifier import classify_attack
from src.explainability.explainer import (
    explain_all_windows,
)
from src.models.error_analysis_v2 import (
    compute_raw_features,
    get_attack_windows,
)
from src.preprocessing.feature_config import load_config
from src.preprocessing.load_data import clean_normal_data
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42

CATEGORY_COLORS = {
    "rolling": "#4C72B0",
    "rate_of_change": "#55A868",
    "deviation": "#C44E52",
    "contradiction": "#8172B2",
    "frozen": "#CCB974",
    "spike_jump": "#64B5CD",
    "constant_change": "#E78AC3",
    "switching": "#A6D854",
    "pair_residual": "#FFD92F",
    "low_variance": "#B3B3B3",
    "raw_sensor": "#FC8D62",
}

CATEGORY_LABELS = {
    "rolling": "Rolling (mean/std)",
    "rate_of_change": "Rate of Change",
    "deviation": "Deviation",
    "contradiction": "Contradiction",
    "frozen": "Frozen",
    "spike_jump": "Spike / Jump",
    "constant_change": "Constant Change",
    "switching": "Switching Rate",
    "pair_residual": "Pair Residual",
    "low_variance": "Low Variance",
    "raw_sensor": "Raw Sensor",
}


def build_data_and_model():
    """Load data, build features, train AE — same as error_analysis_v2."""
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

    train_raw_feat = compute_raw_features(train_df)
    test_raw_feat = compute_raw_features(test_df)

    continuous = get_continuous_sensors(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous])
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[continuous] = scaler.transform(train_df[continuous])
    test_scaled[continuous] = scaler.transform(test_df[continuous])

    from src.preprocessing.feature_engineering import (
        compute_deviation,
        compute_pair_residuals,
        compute_rate_of_change,
        compute_rolling_features,
    )
    config = load_config()

    def compute_scaled_features(df_scaled):
        feats = {}
        for sensor, info in config["continuous_sensors"].items():
            if sensor not in df_scaled.columns:
                continue
            series = df_scaled[sensor]
            features = info["features"]
            if "rolling_mean" in features or "rolling_std" in features:
                feats.update(compute_rolling_features(
                    series, sensor, config["window_sizes"], features,
                ))
            if "rate_of_change" in features:
                feats.update(compute_rate_of_change(series, sensor))
            if "deviation_from_baseline" in features:
                bl = config.get("baselines", {}).get(sensor, {})
                if bl:
                    feats.update(compute_deviation(
                        series, sensor, bl["mean"], bl["std"],
                    ))
        feats.update(compute_pair_residuals(
            df_scaled, config.get("cross_sensor_pairs", []),
        ))
        return pd.DataFrame(feats, index=df_scaled.index)

    train_scaled_feat = compute_scaled_features(train_scaled)
    test_scaled_feat = compute_scaled_features(test_scaled)

    exclude = {"Timestamp", "label"}
    orig_cols = [
        c for c in train_scaled.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(train_scaled[c])
    ]

    train_all = pd.concat(
        [train_scaled[orig_cols], train_scaled_feat, train_raw_feat], axis=1,
    )
    test_all = pd.concat(
        [test_scaled[orig_cols], test_scaled_feat, test_raw_feat], axis=1,
    )
    train_all = train_all.loc[:, ~train_all.columns.duplicated()]
    test_all = test_all.loc[:, ~test_all.columns.duplicated()]

    feature_names = list(train_all.columns)

    X_train = np.nan_to_num(
        train_all.values.astype(np.float32), nan=0, posinf=100, neginf=-100,
    )
    X_test = np.nan_to_num(
        test_all.values.astype(np.float32), nan=0, posinf=100, neginf=-100,
    )

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    print("Training Autoencoder for explainability...")
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

    return X_test, y_test, ae, feature_names, windows


def plot_stacked_bar(explanations: list[dict], save_path: Path) -> None:
    """Stacked bar chart — each attack window is a column, colors = categories."""
    categories = list(CATEGORY_COLORS.keys())
    window_names = [exp["window"] for exp in explanations]
    n_windows = len(window_names)

    data = np.zeros((len(categories), n_windows))
    for j, exp in enumerate(explanations):
        cat_totals = exp["category_totals"]
        total = cat_totals.sum()
        if total == 0:
            continue
        for i, cat in enumerate(categories):
            data[i, j] = cat_totals.get(cat, 0) / total * 100

    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(n_windows)
    bottom = np.zeros(n_windows)

    for i, cat in enumerate(categories):
        if data[i].sum() == 0:
            continue
        ax.bar(
            x, data[i], bottom=bottom,
            color=CATEGORY_COLORS[cat],
            label=CATEGORY_LABELS[cat],
            edgecolor="white", linewidth=0.3,
        )
        bottom += data[i]

    ax.set_xlabel("Attack Window", fontsize=12)
    ax.set_ylabel("Feature Category Contribution (%)", fontsize=12)
    ax.set_title(
        "Feature Attribution per Attack Window (Autoencoder Reconstruction Error)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(window_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, frameon=True,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.relative_to(ROOT)}")


def main():
    print("=" * 70)
    print("  EXPLAINABILITY — Feature Attribution + Attack Classification")
    print("=" * 70)

    X_test, y_test, ae, feature_names, windows = build_data_and_model()

    print(f"\n  Features: {len(feature_names)}")
    print(f"  Attack windows: {len(windows)}")

    print("\nComputing feature attribution for all windows...")
    explanations = explain_all_windows(X_test, y_test, ae, feature_names, windows)

    print("\nClassifying attack types...")
    classifications = []
    for exp in explanations:
        result = classify_attack(exp)
        exp["classification"] = result
        classifications.append(result)

    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Win':<6} {'Dur':>5} {'Type':<22} {'Top Sensor':<12} "
          f"{'Stage':>5}  Evidence")
    print("  " + "-" * 90)

    summary_rows = []
    for exp in explanations:
        cls = exp["classification"]
        top_sensor = (
            cls["involved_sensors"][0]["sensor"]
            if cls["involved_sensors"] else "—"
        )
        top_stage = (
            cls["involved_sensors"][0]["stage"]
            if cls["involved_sensors"] else 0
        )
        evidence_short = cls["evidence"][:60]
        print(f"  {exp['window']:<6} {exp['duration']:>5} "
              f"{cls['attack_type']:<22} {top_sensor:<12} "
              f"{'P'+str(top_stage):>5}  {evidence_short}")

        top_cats = exp["category_totals"]
        summary_rows.append({
            "window": exp["window"],
            "duration": exp["duration"],
            "attack_type": cls["attack_type"],
            "top_sensor": top_sensor,
            "stage": top_stage,
            "evidence": cls["evidence"],
            "top_category": top_cats.index[0] if len(top_cats) > 0 else "",
            "top_cat_pct": (
                top_cats.iloc[0] / top_cats.sum() * 100
                if len(top_cats) > 0 and top_cats.sum() > 0 else 0
            ),
        })

    type_counts = pd.Series(
        [c["attack_type"] for c in classifications]
    ).value_counts()
    print("\n  Attack type distribution:")
    for atype, count in type_counts.items():
        print(f"    {atype}: {count}")

    print("\nGenerating stacked bar chart...")
    plot_stacked_bar(
        explanations,
        RESULTS_DIR / "figures" / "explainability_stacked_bar.png",
    )

    summary_df = pd.DataFrame(summary_rows)
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(metrics_dir / "explainability_summary.csv", index=False)
    print("  Saved: results/metrics/explainability_summary.csv")


if __name__ == "__main__":
    main()
