"""EDA: Dataset overview and exploration of SWaT data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing.load_data import (
    clean_normal_data,
    get_sensor_columns,
    load_or_create_processed,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def part1_basic_statistics(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Print basic statistics about both datasets."""
    print("=" * 60)
    print("PART 1 — BASIC STATISTICS")
    print("=" * 60)

    for name, df in [("Normal", normal), ("Attack", attack)]:
        print(f"\n--- {name} ---")
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"Period: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

        sensors = get_sensor_columns(df)
        float_cols = df[sensors].select_dtypes(include="float64").columns.tolist()
        int_cols = df[sensors].select_dtypes(include="int64").columns.tolist()
        print(f"Continuous sensors (float): {len(float_cols)}")
        print(f"Discrete sensors (int):     {len(int_cols)}")

    print("\n--- Data Quality ---")
    for name, df in [("Normal", normal), ("Attack", attack)]:
        sensors = get_sensor_columns(df)
        nulls = df[sensors].isnull().sum().sum()
        inf_count = df[sensors].select_dtypes(include="float64").apply(
            lambda x: x.isin([float("inf"), float("-inf")]).sum()
        ).sum()
        dup_ts = df.duplicated(subset=["Timestamp"]).sum()
        print(f"\n{name}:")
        print(f"  Missing values:       {nulls}")
        print(f"  Infinite values:      {inf_count}")
        print(f"  Duplicate timestamps: {dup_ts}")

    print("\n--- Summary Statistics (Normal, continuous sensors) ---")
    sensors = get_sensor_columns(normal)
    float_cols = normal[sensors].select_dtypes(include="float64").columns.tolist()
    print(normal[float_cols].describe().round(2).to_string())


def part2_class_distribution(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Analyze and visualize class distribution after cleanup."""
    print("\n" + "=" * 60)
    print("PART 2 — CLASS DISTRIBUTION (after cleanup)")
    print("=" * 60)

    total = len(normal) + len(attack)
    print(f"\nNormal: {len(normal):>10,} ({len(normal)/total*100:.1f}%)")
    print(f"Attack: {len(attack):>10,} ({len(attack)/total*100:.1f}%)")
    print(f"Total:  {total:>10,}")
    print(f"Imbalance ratio: {len(normal)/len(attack):.1f}:1")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"Normal": len(normal), "Attack": len(attack)}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=["#2a78d6", "#e34948"])
    ax.set_ylabel("Number of Samples")
    ax.set_title("SWaT Dataset — Class Distribution (after cleanup)")

    for bar, count in zip(bars, counts.values()):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'class_distribution.png'}")


if __name__ == "__main__":
    print("Loading data...")
    normal, attack = load_or_create_processed()
    normal = clean_normal_data(normal)
    part2_class_distribution(normal, attack)
