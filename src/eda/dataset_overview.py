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


def _build_combined_timeline(
    normal: pd.DataFrame, attack: pd.DataFrame,
) -> pd.DataFrame:
    """Merge normal and attack into one sorted timeline."""
    combined = pd.concat([normal, attack]).sort_values("Timestamp")
    return combined.drop_duplicates(subset=["Timestamp"]).reset_index(drop=True)


def part3_timeseries(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Plot key sensors over time with attack periods highlighted."""
    print("\n" + "=" * 60)
    print("PART 3 — TIME-SERIES BEHAVIOR")
    print("=" * 60)

    combined = _build_combined_timeline(normal, attack)
    print(f"\nCombined timeline: {len(combined):,} rows")
    print(f"Period: {combined['Timestamp'].min()} to {combined['Timestamp'].max()}")

    attack_timestamps = set(attack["Timestamp"])

    key_sensors = [
        ("LIT101", "P1 — Water level"),
        ("FIT101", "P1 — Flow rate"),
        ("AIT201", "P2 — Chemical analysis"),
        ("LIT301", "P3 — UF tank level"),
        ("FIT401", "P4 — Dechlorination flow"),
        ("AIT501", "P5 — RO analysis"),
    ]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(key_sensors), 1, figsize=(16, 3.5 * len(key_sensors)))

    sample = combined.iloc[::30].copy()
    sample["is_attack"] = sample["Timestamp"].isin(attack_timestamps)

    for ax, (sensor, label) in zip(axes, key_sensors):
        normal_mask = ~sample["is_attack"]
        attack_mask = sample["is_attack"]

        ax.plot(
            sample.loc[normal_mask, "Timestamp"],
            sample.loc[normal_mask, sensor],
            color="#2a78d6", linewidth=0.4, alpha=0.7, label="Normal",
        )
        ax.scatter(
            sample.loc[attack_mask, "Timestamp"],
            sample.loc[attack_mask, sensor],
            color="#e34948", s=1, alpha=0.8, label="Attack",
        )
        ax.set_ylabel(sensor, fontsize=10)
        ax.set_title(label, fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=8, markerscale=5)
        ax.tick_params(axis="x", labelsize=8)

    axes[-1].set_xlabel("Time")
    plt.suptitle("SWaT — Key Sensor Readings Over Time", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'sensor_timeseries.png'}")

    print("\n--- Attack periods ---")
    attack_sorted = attack.sort_values("Timestamp")
    time_diff = attack_sorted["Timestamp"].diff().dt.total_seconds()
    gap_indices = time_diff[time_diff > 60].index.tolist()

    periods = []
    start_idx = attack_sorted.index[0]
    for gap_idx in gap_indices:
        prev_idx = attack_sorted.index[attack_sorted.index.get_loc(gap_idx) - 1]
        periods.append((attack_sorted.loc[start_idx, "Timestamp"],
                        attack_sorted.loc[prev_idx, "Timestamp"]))
        start_idx = gap_idx
    periods.append((attack_sorted.loc[start_idx, "Timestamp"],
                    attack_sorted.loc[attack_sorted.index[-1], "Timestamp"]))

    print(f"\nFound {len(periods)} distinct attack periods:")
    for i, (start, end) in enumerate(periods, 1):
        duration = (end - start).total_seconds()
        rows = len(attack[(attack["Timestamp"] >= start) & (attack["Timestamp"] <= end)])
        print(f"  {i:2d}. {start} to {end} ({duration:.0f}s, {rows} rows)")


if __name__ == "__main__":
    print("Loading data...")
    normal, attack = load_or_create_processed()
    normal = clean_normal_data(normal)
    part3_timeseries(normal, attack)
