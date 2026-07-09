"""EDA: Dataset overview and initial exploration of SWaT data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing.load_data import get_sensor_columns, load_or_create_processed

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def print_dataset_summary(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Print basic statistics about both datasets."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nNormal data: {normal.shape[0]:,} rows x {normal.shape[1]} columns")
    print(f"Attack data: {attack.shape[0]:,} rows x {attack.shape[1]} columns")

    print(f"\nNormal period: {normal['Timestamp'].min()} to {normal['Timestamp'].max()}")
    print(f"Attack period: {attack['Timestamp'].min()} to {attack['Timestamp'].max()}")

    sensors = get_sensor_columns(normal)
    float_cols = normal[sensors].select_dtypes(include="float64").columns.tolist()
    int_cols = normal[sensors].select_dtypes(include="int64").columns.tolist()
    print(f"\nSensor columns: {len(sensors)}")
    print(f"  Continuous (float): {len(float_cols)} — {float_cols[:5]}...")
    print(f"  Discrete (int):     {len(int_cols)} — {int_cols[:5]}...")


def print_data_quality(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Check for missing values, infinities, and duplicates."""
    print("\n" + "=" * 60)
    print("DATA QUALITY")
    print("=" * 60)

    for name, df in [("Normal", normal), ("Attack", attack)]:
        sensors = get_sensor_columns(df)
        nulls = df[sensors].isnull().sum()
        inf_count = df[sensors].select_dtypes(include="float64").apply(
            lambda x: x.isin([float("inf"), float("-inf")]).sum()
        )
        duplicates = df.duplicated(subset=["Timestamp"]).sum()

        print(f"\n{name}:")
        print(f"  Missing values: {nulls.sum()}")
        if nulls.sum() > 0:
            print(f"  Columns with nulls: {nulls[nulls > 0].to_dict()}")
        print(f"  Infinite values: {inf_count.sum()}")
        print(f"  Duplicate timestamps: {duplicates}")


def print_class_distribution(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Show label distribution."""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    total = len(normal) + len(attack)
    print(f"\nNormal: {len(normal):>10,} ({len(normal)/total*100:.1f}%)")
    print(f"Attack: {len(attack):>10,} ({len(attack)/total*100:.1f}%)")
    print(f"Total:  {total:>10,}")
    print(f"Imbalance ratio: {len(normal)/len(attack):.1f}:1")


def plot_class_distribution(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Bar chart of class distribution."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = {"Normal": len(normal), "Attack": len(attack)}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=["steelblue", "crimson"])
    ax.set_ylabel("Number of Samples")
    ax.set_title("SWaT Dataset — Class Distribution")

    for bar, count in zip(bars, counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'class_distribution.png'}")


def plot_sensor_timeseries(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Plot key sensors over time for normal vs attack periods."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    key_sensors = ["LIT101", "FIT101", "AIT201", "LIT301", "FIT401", "AIT501"]
    available = [s for s in key_sensors if s in normal.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(14, 3 * len(available)), sharex=False)
    if len(available) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, available):
        sample_n = normal.iloc[::60]
        sample_a = attack.iloc[::10]

        ax.plot(sample_n["Timestamp"], sample_n[sensor], color="steelblue",
                alpha=0.6, linewidth=0.5, label="Normal")
        ax.plot(sample_a["Timestamp"], sample_a[sensor], color="crimson",
                alpha=0.6, linewidth=0.5, label="Attack")
        ax.set_ylabel(sensor)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title("SWaT — Key Sensor Readings Over Time")
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_timeseries.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'sensor_timeseries.png'}")


def plot_sensor_distributions(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Distribution comparison of continuous sensors: normal vs attack."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sensors = get_sensor_columns(normal)
    float_sensors = normal[sensors].select_dtypes(include="float64").columns.tolist()
    plot_sensors = float_sensors[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for ax, sensor in zip(axes, plot_sensors):
        ax.hist(normal[sensor].dropna(), bins=50, alpha=0.6, color="steelblue",
                label="Normal", density=True)
        ax.hist(attack[sensor].dropna(), bins=50, alpha=0.6, color="crimson",
                label="Attack", density=True)
        ax.set_title(sensor, fontsize=10)
        ax.legend(fontsize=7)

    plt.suptitle("SWaT — Sensor Distributions: Normal vs Attack", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_distributions.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'sensor_distributions.png'}")


def plot_correlation_heatmap(normal: pd.DataFrame) -> None:
    """Correlation heatmap of sensor features (normal data only)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sensors = get_sensor_columns(normal)
    corr = normal[sensors].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("SWaT — Sensor Correlation Matrix (Normal Data)")
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'correlation_heatmap.png'}")


def run_full_eda() -> None:
    """Run all EDA steps."""
    print("Loading data...")
    normal, attack = load_or_create_processed()

    print_dataset_summary(normal, attack)
    print_data_quality(normal, attack)
    print_class_distribution(normal, attack)

    print("\nGenerating plots...")
    plot_class_distribution(normal, attack)
    plot_sensor_timeseries(normal, attack)
    plot_sensor_distributions(normal, attack)
    plot_correlation_heatmap(normal)

    print("\nEDA complete!")


if __name__ == "__main__":
    run_full_eda()
