"""Generate feature configuration from KS test results."""

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from src.preprocessing.load_data import get_sensor_columns
from src.preprocessing.prepare_data import get_continuous_sensors, get_discrete_sensors

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

WINDOW_SIZES = [60, 120, 300]

HIGH_TIER_THRESHOLD = 0.2
MEDIUM_TIER_THRESHOLD = 0.05

HIGH_TIER_FEATURES = [
    "rolling_mean", "rolling_std", "rate_of_change", "deviation_from_baseline",
]
MEDIUM_TIER_FEATURES = ["rolling_mean", "rate_of_change"]

ALWAYS_CONSTANT_SENSORS = ["P204", "P206"]

CONTRADICTION_PAIRS = [
    {"actuator": "MV101", "on_value": 2, "sensor": "FIT101", "expected": "positive"},
    {"actuator": "P101", "on_value": 2, "sensor": "FIT101", "expected": "positive"},
    {"actuator": "P501", "on_value": 2, "sensor": "FIT501", "expected": "positive"},
    {"actuator": "P301", "on_value": 1, "sensor": "FIT301", "expected": "positive"},
]

CORRELATION_PAIRS = [
    {"sensor_a": "AIT402", "sensor_b": "PIT502"},
    {"sensor_a": "AIT402", "sensor_b": "AIT501"},
    {"sensor_a": "FIT101", "sensor_b": "LIT301"},
    {"sensor_a": "PIT501", "sensor_b": "PIT502"},
    {"sensor_a": "AIT201", "sensor_b": "AIT402"},
]


def run_ks_tests(
    train: pd.DataFrame, test: pd.DataFrame,
) -> list[dict]:
    """Run KS test on each continuous sensor, return sorted results."""
    continuous = get_continuous_sensors(train)
    results = []
    for sensor in continuous:
        ks_stat, p_val = stats.ks_2samp(
            train[sensor].dropna(), test[sensor].dropna(),
        )
        results.append({
            "sensor": sensor,
            "ks_statistic": round(ks_stat, 4),
            "p_value": float(f"{p_val:.2e}"),
        })
    results.sort(key=lambda x: x["ks_statistic"], reverse=True)
    return results


def assign_tier(ks_stat: float) -> str:
    """Assign feature tier based on KS statistic."""
    if ks_stat >= HIGH_TIER_THRESHOLD:
        return "high"
    if ks_stat >= MEDIUM_TIER_THRESHOLD:
        return "medium"
    return "low"


def build_sensor_config(
    train: pd.DataFrame, test: pd.DataFrame,
) -> dict:
    """Build complete feature configuration from data."""
    ks_results = run_ks_tests(train, test)

    continuous_config = {}
    for result in ks_results:
        sensor = result["sensor"]
        tier = assign_tier(result["ks_statistic"])

        if tier == "high":
            features = HIGH_TIER_FEATURES
        elif tier == "medium":
            features = MEDIUM_TIER_FEATURES
        else:
            features = []

        continuous_config[sensor] = {
            "tier": tier,
            "ks_statistic": result["ks_statistic"],
            "features": features,
        }

    baselines = {}
    continuous = get_continuous_sensors(train)
    for sensor in continuous:
        baselines[sensor] = {
            "mean": round(float(train[sensor].mean()), 4),
            "std": round(float(train[sensor].std()), 4),
        }

    config = {
        "window_sizes": WINDOW_SIZES,
        "tiers": {
            "high": {"threshold": HIGH_TIER_THRESHOLD, "features": HIGH_TIER_FEATURES},
            "medium": {"threshold": MEDIUM_TIER_THRESHOLD, "features": MEDIUM_TIER_FEATURES},
            "low": {"threshold": 0, "features": []},
        },
        "continuous_sensors": continuous_config,
        "discrete": {
            "always_constant": ALWAYS_CONSTANT_SENSORS,
            "contradiction_pairs": CONTRADICTION_PAIRS,
        },
        "cross_sensor_pairs": CORRELATION_PAIRS,
        "baselines": baselines,
    }
    return config


def save_config(config: dict) -> Path:
    """Save configuration to JSON file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "feature_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def load_config() -> dict:
    """Load configuration from JSON file."""
    path = CONFIG_DIR / "feature_config.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    from src.preprocessing.prepare_data import prepare_train_test

    print("Running KS tests and building feature config...")
    train, test, _ = prepare_train_test()
    config = build_sensor_config(train, test)
    path = save_config(config)

    print(f"\nSaved: {path}")

    tiers = {"high": 0, "medium": 0, "low": 0}
    for sensor, info in config["continuous_sensors"].items():
        tiers[info["tier"]] += 1

    print(f"\nTier distribution:")
    for tier, count in tiers.items():
        print(f"  {tier:<8} {count} sensors")

    print(f"\nWindow sizes: {config['window_sizes']}")
    print(f"Cross-sensor pairs: {len(config['cross_sensor_pairs'])}")
    print(f"Contradiction pairs: {len(config['discrete']['contradiction_pairs'])}")
    print(f"Always-constant sensors: {config['discrete']['always_constant']}")
