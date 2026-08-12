"""Detect data drift using KS tests against training baselines.

Compares incoming sensor distributions to the training data baselines
stored in feature_config.json. Uses the same KS infrastructure from
feature engineering — one method serves both purposes.
"""

from pathlib import Path

import numpy as np
from scipy import stats

from src.preprocessing.feature_config import load_config

ROOT = Path(__file__).resolve().parents[2]


def compute_drift_scores(
    recent_values: dict[str, np.ndarray],
    baselines: dict[str, dict],
) -> list[dict]:
    """Compare recent sensor values against training baselines using KS test.

    Args:
        recent_values: {sensor_name: array of recent values}
        baselines: {sensor_name: {"mean": float, "std": float}} from config

    Returns:
        List of {sensor, ks_statistic, p_value, drifted} sorted by KS desc.
    """
    results = []
    for sensor, values in recent_values.items():
        if sensor not in baselines or len(values) < 30:
            continue

        bl = baselines[sensor]
        reference = np.random.RandomState(42).normal(
            bl["mean"], max(bl["std"], 1e-6), size=len(values),
        )

        ks_stat, p_value = stats.ks_2samp(values, reference)
        results.append({
            "sensor": sensor,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 6),
            "drifted": bool(p_value < 0.01),
        })

    results.sort(key=lambda x: x["ks_statistic"], reverse=True)
    return results


def check_drift(buffer_values: dict[str, np.ndarray]) -> dict:
    """Run drift detection on the current buffer.

    Returns a summary with per-sensor results and overall status.
    """
    config = load_config()
    baselines = config.get("baselines", {})

    scores = compute_drift_scores(buffer_values, baselines)
    drifted_sensors = [s for s in scores if s["drifted"]]

    return {
        "status": "drift_detected" if drifted_sensors else "stable",
        "drifted_count": len(drifted_sensors),
        "total_checked": len(scores),
        "sensors": scores,
    }
