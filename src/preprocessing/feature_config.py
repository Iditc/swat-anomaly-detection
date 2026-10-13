"""Generate feature configuration from training data only.

All continuous sensors get the full feature set — no test data
or labels are used for tier assignment, avoiding any leakage.
"""

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.prepare_data import get_continuous_sensors

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

WINDOW_SIZES = [60, 120, 300]

FEATURES = [
    "rolling_mean", "rolling_std", "rate_of_change", "deviation_from_baseline",
]

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


def build_sensor_config(train: pd.DataFrame) -> dict:
    """Build feature configuration from training data only.

    All continuous sensors get the full feature set.
    No test data or labels needed — the model learns which features matter.
    """
    continuous = get_continuous_sensors(train)

    continuous_config = {}
    for sensor in continuous:
        continuous_config[sensor] = {
            "tier": "high",
            "features": FEATURES,
        }

    baselines = {}
    for sensor in continuous:
        baselines[sensor] = {
            "mean": round(float(train[sensor].mean()), 4),
            "std": round(float(train[sensor].std()), 4),
        }

    config = {
        "window_sizes": WINDOW_SIZES,
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

    print("Building feature config from training data only...")
    train, _, _ = prepare_train_test()
    config = build_sensor_config(train)
    path = save_config(config)

    print(f"\nSaved: {path}")
    print(f"Continuous sensors: {len(config['continuous_sensors'])} (all high tier)")
    print(f"Window sizes: {config['window_sizes']}")
    print(f"Cross-sensor pairs: {len(config['cross_sensor_pairs'])}")
    print(f"Contradiction pairs: {len(config['discrete']['contradiction_pairs'])}")
    print(f"Always-constant sensors: {config['discrete']['always_constant']}")
