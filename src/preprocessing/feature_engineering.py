"""Generate features from sensor data using KS-driven configuration."""

import pandas as pd

from src.preprocessing.feature_config import load_config


def compute_rolling_features(
    series: pd.Series, sensor: str, windows: list[int], features: list[str],
) -> dict[str, pd.Series]:
    """Compute rolling window features for a single sensor."""
    result = {}
    for window in windows:
        if "rolling_mean" in features:
            result[f"{sensor}_mean_{window}"] = (
                series.rolling(window, min_periods=1).mean()
            )
        if "rolling_std" in features:
            result[f"{sensor}_std_{window}"] = (
                series.rolling(window, min_periods=1).std().fillna(0)
            )
    return result


def compute_rate_of_change(series: pd.Series, sensor: str) -> dict[str, pd.Series]:
    """Compute rate of change (first difference) for a sensor."""
    return {f"{sensor}_roc": series.diff().fillna(0)}


def compute_deviation(
    series: pd.Series, sensor: str, mean: float, std: float,
) -> dict[str, pd.Series]:
    """Compute absolute deviation from training baseline in std units."""
    if std > 0:
        return {f"{sensor}_dev": ((series - mean) / std).abs()}
    return {f"{sensor}_dev": (series - mean).abs()}


def compute_pair_residuals(
    df: pd.DataFrame, pairs: list[dict],
) -> dict[str, pd.Series]:
    """Compute residual features for correlated sensor pairs."""
    result = {}
    for pair in pairs:
        a, b = pair["sensor_a"], pair["sensor_b"]
        if a in df.columns and b in df.columns:
            result[f"{a}_{b}_residual"] = (df[a] - df[b]).abs()
    return result


def compute_contradiction_features(
    df: pd.DataFrame, pairs: list[dict],
) -> dict[str, pd.Series]:
    """Compute binary features for physical contradictions."""
    result = {}
    for pair in pairs:
        actuator = pair["actuator"]
        sensor = pair["sensor"]
        on_value = pair["on_value"]

        if actuator not in df.columns or sensor not in df.columns:
            continue

        actuator_on = df[actuator] == on_value
        no_flow = df[sensor].abs() < 0.1

        if pair["expected"] == "positive":
            result[f"{actuator}_{sensor}_contradiction"] = (
                (actuator_on & no_flow).astype(int)
            )
        else:
            result[f"{actuator}_{sensor}_contradiction"] = (
                (actuator_on & ~no_flow).astype(int)
            )
    return result


def compute_constant_sensor_changes(
    df: pd.DataFrame, always_constant: list[str],
) -> dict[str, pd.Series]:
    """Compute binary features for sensors that should never change."""
    result = {}
    for sensor in always_constant:
        if sensor in df.columns:
            changed = df[sensor].diff().ne(0).astype(int)
            changed.iloc[0] = 0
            result[f"{sensor}_changed"] = changed
    return result


def compute_switching_rate(
    df: pd.DataFrame, discrete_cols: list[str], window: int = 60,
) -> dict[str, pd.Series]:
    """Compute switching frequency for discrete sensors."""
    result = {}
    for sensor in discrete_cols:
        if sensor in df.columns:
            switches = df[sensor].diff().ne(0).astype(int)
            result[f"{sensor}_switch_rate_{window}"] = (
                switches.rolling(window, min_periods=1).sum()
            )
    return result


def get_discrete_columns(df: pd.DataFrame) -> list[str]:
    """Return discrete sensor columns present in the dataframe."""
    exclude = {"Timestamp", "label", "Normal/Attack"}
    return [
        col for col in df.columns
        if col not in exclude and df[col].dtype in ("int64", "int32")
    ]


def generate_features(
    df: pd.DataFrame, config: dict | None = None,
) -> pd.DataFrame:
    """Generate all features based on configuration.

    Works identically in training and production — same config, same output.
    """
    if config is None:
        config = load_config()

    windows = config["window_sizes"]
    baselines = config.get("baselines", {})
    new_cols: dict[str, pd.Series] = {}

    for sensor, info in config["continuous_sensors"].items():
        if sensor not in df.columns:
            continue

        series = df[sensor]
        features = info["features"]

        if "rolling_mean" in features or "rolling_std" in features:
            new_cols.update(
                compute_rolling_features(series, sensor, windows, features),
            )

        if "rate_of_change" in features:
            new_cols.update(compute_rate_of_change(series, sensor))

        if "deviation_from_baseline" in features and sensor in baselines:
            bl = baselines[sensor]
            new_cols.update(compute_deviation(series, sensor, bl["mean"], bl["std"]))

    new_cols.update(
        compute_pair_residuals(df, config.get("cross_sensor_pairs", [])),
    )

    discrete = config.get("discrete", {})
    new_cols.update(
        compute_contradiction_features(df, discrete.get("contradiction_pairs", [])),
    )
    new_cols.update(
        compute_constant_sensor_changes(df, discrete.get("always_constant", [])),
    )

    discrete_cols = get_discrete_columns(df)
    new_cols.update(compute_switching_rate(df, discrete_cols, window=60))

    features_df = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, features_df], axis=1)


def save_featured_data(
    train: pd.DataFrame, test: pd.DataFrame,
) -> None:
    """Save feature-engineered datasets to parquet."""
    from src.preprocessing.prepare_data import PROCESSED_DIR

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(PROCESSED_DIR / "train_featured.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test_featured.parquet", index=False)

    print(f"Saved: train_featured.parquet ({train.shape[0]:,} x {train.shape[1]})")
    print(f"Saved: test_featured.parquet ({test.shape[0]:,} x {test.shape[1]})")


if __name__ == "__main__":
    from src.preprocessing.prepare_data import prepare_train_test

    print("Loading scaled data...")
    train, test, _ = prepare_train_test()

    config = load_config()
    print(f"Config: {len(config['continuous_sensors'])} sensors, "
          f"{len(config['window_sizes'])} windows")

    original_cols = len(train.columns)

    print("\nGenerating train features...")
    train_feat = generate_features(train, config)
    print(f"  {original_cols} -> {len(train_feat.columns)} columns "
          f"(+{len(train_feat.columns) - original_cols} features)")

    print("\nGenerating test features...")
    test_feat = generate_features(test, config)
    print(f"  {original_cols} -> {len(test_feat.columns)} columns")

    new_cols = [c for c in train_feat.columns if c not in train.columns]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for col in sorted(new_cols)[:20]:
        print(f"  {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")

    print(f"\nNaN check: {train_feat[new_cols].isna().sum().sum()} NaN values")

    save_featured_data(train_feat, test_feat)
