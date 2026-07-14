"""EDA: Discrete sensor (actuator) analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.load_data import (
    clean_normal_data,
    get_sensor_columns,
    load_or_create_processed,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"

ACTUATOR_FLOW_PAIRS = [
    ("MV101", 2, "FIT101", ">", 0.5, "MV101 open + no flow"),
    ("MV101", 0, "FIT101", ">", 0.5, "MV101 closed + flow"),
    ("P101", 2, "FIT101", "<", 0.5, "P101 on + no flow"),
    ("P501", 2, "FIT501", "<", 0.5, "P501 on + no RO flow"),
    ("P501", 1, "FIT501", ">", 0.5, "P501 off + RO flow"),
]


def _get_discrete_sensors(df: pd.DataFrame) -> list[str]:
    """Return discrete sensor column names."""
    sensors = get_sensor_columns(df)
    return df[sensors].select_dtypes(include="int64").columns.tolist()


def _state_on_pct(series: pd.Series) -> float:
    """Calculate percentage of time a sensor is ON/OPEN."""
    if series.max() == 2:
        return (series == 2).mean()
    return (series == 1).mean()


def part6_discrete_features(
    normal: pd.DataFrame, attack: pd.DataFrame,
) -> None:
    """Analyze discrete sensor states, switching, and contradictions."""
    print("\n" + "=" * 60)
    print("PART 6 — DISCRETE FEATURES (actuators)")
    print("=" * 60)

    discrete = _get_discrete_sensors(normal)
    print(f"\nDiscrete sensors: {len(discrete)}")

    print(f"\n{'Sensor':<12} {'Type':<8} {'Normal %ON':>12} {'Attack %ON':>12} {'Diff':>8}")
    print("-" * 54)

    state_diffs = []
    for sensor in discrete:
        n_on = _state_on_pct(normal[sensor])
        a_on = _state_on_pct(attack[sensor])
        stype = _sensor_type(sensor)
        diff = abs(a_on - n_on)
        state_diffs.append((sensor, stype, n_on, a_on, diff))
        print(f"{sensor:<12} {stype:<8} {n_on:>11.1%} {a_on:>11.1%} {diff:>7.1%}")

    _print_switching_frequency(normal, attack, discrete)
    _print_contradictions(normal, attack)
    _print_always_on_changes(normal, attack, discrete)
    _plot_state_changes(state_diffs)


def _sensor_type(sensor: str) -> str:
    """Determine actuator type from name."""
    if sensor.startswith("MV"):
        return "Valve"
    if sensor.startswith("UV"):
        return "UV"
    return "Pump"


def _print_switching_frequency(
    normal: pd.DataFrame, attack: pd.DataFrame, discrete: list[str],
) -> None:
    """Print how often each actuator changes state."""
    print(f"\n{'Sensor':<12} {'Normal/hr':>12} {'Attack/hr':>12} {'Ratio':>8}")
    print("-" * 46)

    n_hours = len(normal) / 3600
    a_hours = len(attack) / 3600

    for sensor in discrete:
        n_changes = (normal[sensor].diff().abs() > 0).sum()
        a_changes = (attack[sensor].diff().abs() > 0).sum()
        n_rate = n_changes / n_hours
        a_rate = a_changes / a_hours
        ratio = a_rate / n_rate if n_rate > 0 else float("inf")
        flag = " ***" if ratio > 1.5 or ratio < 0.5 else ""
        print(f"{sensor:<12} {n_rate:>11.1f} {a_rate:>11.1f} {ratio:>7.2f}{flag}")


def _print_contradictions(
    normal: pd.DataFrame, attack: pd.DataFrame,
) -> None:
    """Check for physically impossible actuator-sensor combinations."""
    print("\n--- Physical contradictions ---")
    print(f"{'Contradiction':<30} {'Normal':>10} {'Attack':>10} "
          f"{'Normal%':>10} {'Attack%':>10}")
    print("-" * 72)

    for actuator, act_val, sensor, op, threshold, desc in ACTUATOR_FLOW_PAIRS:
        n_act = normal[actuator] == act_val
        a_act = attack[actuator] == act_val

        if op == "<":
            n_sensor = normal[sensor] < threshold
            a_sensor = attack[sensor] < threshold
        else:
            n_sensor = normal[sensor] > threshold
            a_sensor = attack[sensor] > threshold

        n_count = (n_act & n_sensor).sum()
        a_count = (a_act & a_sensor).sum()
        n_pct = n_count / len(normal) * 100
        a_pct = a_count / len(attack) * 100
        print(f"{desc:<30} {n_count:>10,} {a_count:>10,} "
              f"{n_pct:>9.2f}% {a_pct:>9.2f}%")


def _print_always_on_changes(
    normal: pd.DataFrame, attack: pd.DataFrame, discrete: list[str],
) -> None:
    """Find sensors that are constant in normal but change in attack."""
    print("\n--- Always-on sensors that change during attacks ---")
    for sensor in discrete:
        if normal[sensor].nunique() == 1 and attack[sensor].nunique() > 1:
            n_val = normal[sensor].iloc[0]
            a_vals = sorted(attack[sensor].unique())
            print(f"  {sensor:<10} always {n_val} in normal, "
                  f"attack values: {a_vals}")


def _plot_state_changes(
    state_diffs: list[tuple],
) -> None:
    """Bar chart of biggest state distribution changes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    top = sorted(state_diffs, key=lambda x: x[4], reverse=True)[:12]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [t[0] for t in top]
    normal_vals = [t[2] * 100 for t in top]
    attack_vals = [t[3] * 100 for t in top]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, normal_vals, width, label="Normal", color="#2a78d6", alpha=0.8)
    ax.bar(x + width / 2, attack_vals, width, label="Attack", color="#e34948", alpha=0.8)
    ax.set_ylabel("% Time ON/OPEN")
    ax.set_title("SWaT — Discrete Sensor State Changes (Normal vs Attack)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "discrete_state_changes.png", dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'discrete_state_changes.png'}")


if __name__ == "__main__":
    print("Loading data...")
    normal, attack = load_or_create_processed()
    normal = clean_normal_data(normal)
    part6_discrete_features(normal, attack)
