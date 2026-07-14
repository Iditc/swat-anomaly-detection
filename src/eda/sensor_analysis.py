"""EDA: Sensor distribution and correlation analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.preprocessing.load_data import (
    clean_normal_data,
    get_sensor_columns,
    load_or_create_processed,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def part4_sensor_distributions(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Compare sensor distributions between normal and attack data."""
    print("\n" + "=" * 60)
    print("PART 4 — SENSOR DISTRIBUTIONS (normal vs attack)")
    print("=" * 60)

    sensors = get_sensor_columns(normal)
    float_sensors = normal[sensors].select_dtypes(include="float64").columns.tolist()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nContinuous sensors: {len(float_sensors)}")
    print("\nKolmogorov-Smirnov test (normal vs attack):")
    print(f"{'Sensor':<12} {'KS stat':>10} {'p-value':>12} {'Different?':>12}")
    print("-" * 48)

    ks_results = []
    for sensor in float_sensors:
        n_vals = normal[sensor].dropna()
        a_vals = attack[sensor].dropna()
        ks_stat, p_val = stats.ks_2samp(n_vals, a_vals)
        is_diff = "YES" if p_val < 0.001 else "no"
        print(f"{sensor:<12} {ks_stat:>10.4f} {p_val:>12.2e} {is_diff:>12}")
        ks_results.append((sensor, ks_stat, p_val))

    ks_results.sort(key=lambda x: x[1], reverse=True)
    top_sensors = [s for s, _, _ in ks_results[:12]]

    print(f"\nTop 12 most different sensors (highest KS statistic):")
    for i, (sensor, ks_stat, _) in enumerate(ks_results[:12], 1):
        print(f"  {i:2d}. {sensor:<12} KS={ks_stat:.4f}")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for ax, sensor in zip(axes, top_sensors):
        ax.hist(
            normal[sensor].dropna(), bins=50, alpha=0.6, color="#2a78d6",
            label="Normal", density=True,
        )
        ax.hist(
            attack[sensor].dropna(), bins=50, alpha=0.6, color="#e34948",
            label="Attack", density=True,
        )
        ks_val = next(ks for s, ks, _ in ks_results if s == sensor)
        ax.set_title(f"{sensor} (KS={ks_val:.3f})", fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    plt.suptitle(
        "SWaT — Top 12 Most Different Sensors: Normal vs Attack",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sensor_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'sensor_distributions.png'}")


def part5_correlations(normal: pd.DataFrame, attack: pd.DataFrame) -> None:
    """Analyze sensor correlations and how they change during attacks."""
    print("\n" + "=" * 60)
    print("PART 5 — SENSOR CORRELATIONS")
    print("=" * 60)

    sensors = get_sensor_columns(normal)
    float_sensors = normal[sensors].select_dtypes(include="float64").columns.tolist()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corr_normal = normal[float_sensors].corr()
    corr_attack = attack[float_sensors].corr()

    pairs = []
    for i in range(len(float_sensors)):
        for j in range(i + 1, len(float_sensors)):
            s1, s2 = float_sensors[i], float_sensors[j]
            rn = corr_normal.loc[s1, s2]
            ra = corr_attack.loc[s1, s2]
            pairs.append((s1, s2, rn, ra, abs(ra - rn)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\nTop 15 correlated pairs (normal data):")
    print(f"{'Sensor 1':<12} {'Sensor 2':<12} {'Correlation':>12}")
    print("-" * 38)
    for s1, s2, rn, _, _ in pairs[:15]:
        print(f"{s1:<12} {s2:<12} {rn:>12.4f}")

    changes = sorted(pairs, key=lambda x: x[4], reverse=True)
    print("\nTop 15 biggest correlation changes (normal vs attack):")
    print(f"{'Sensor 1':<12} {'Sensor 2':<12} {'Normal':>10} {'Attack':>10} {'Change':>10}")
    print("-" * 56)
    for s1, s2, rn, ra, delta in changes[:15]:
        print(f"{s1:<12} {s2:<12} {rn:>10.4f} {ra:>10.4f} {delta:>10.4f}")

    mask = np.triu(np.ones_like(corr_normal, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_normal, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, annot=False,
        cbar_kws={"label": "Correlation", "shrink": 0.8}, ax=ax,
    )
    ax.set_title("SWaT — Sensor Correlations (Normal Data)", fontsize=14, pad=15)
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "correlation_heatmap_normal.png", dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'correlation_heatmap_normal.png'}")

    corr_diff = corr_attack - corr_normal
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_diff, mask=mask, cmap="RdBu_r", center=0, vmin=-1.2, vmax=1.2,
        square=True, linewidths=0.5, annot=False,
        cbar_kws={
            "label": "Correlation Change (Attack - Normal)", "shrink": 0.8,
        },
        ax=ax,
    )
    ax.set_title("SWaT — Correlation Changes During Attacks", fontsize=14, pad=15)
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "correlation_change_heatmap.png", dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'correlation_change_heatmap.png'}")

    top10 = changes[:10]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{s1}-{s2}" for s1, s2, _, _, _ in top10]
    normal_vals = [rn for _, _, rn, _, _ in top10]
    attack_vals = [ra for _, _, _, ra, _ in top10]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, normal_vals, width, label="Normal", color="#2a78d6", alpha=0.8)
    ax.bar(x + width / 2, attack_vals, width, label="Attack", color="#e34948", alpha=0.8)
    ax.set_ylabel("Correlation")
    ax.set_title("SWaT — Top 10 Biggest Correlation Changes", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "correlation_changes_top10.png", dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'correlation_changes_top10.png'}")


if __name__ == "__main__":
    print("Loading data...")
    normal, attack = load_or_create_processed()
    normal = clean_normal_data(normal)
    part5_correlations(normal, attack)
