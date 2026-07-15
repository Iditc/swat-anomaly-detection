"""Prepare training and test datasets with normalization."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.load_data import (
    clean_normal_data,
    get_sensor_columns,
    load_or_create_processed,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def get_continuous_sensors(df: pd.DataFrame) -> list[str]:
    """Return list of continuous (float) sensor columns."""
    sensors = get_sensor_columns(df)
    return df[sensors].select_dtypes(include="float64").columns.tolist()


def get_discrete_sensors(df: pd.DataFrame) -> list[str]:
    """Return list of discrete (int) sensor columns."""
    sensors = get_sensor_columns(df)
    return df[sensors].select_dtypes(include="int64").columns.tolist()


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on continuous sensors from training data only."""
    continuous = get_continuous_sensors(train_df)
    scaler = StandardScaler()
    scaler.fit(train_df[continuous])
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply fitted scaler to continuous sensors, keep discrete and metadata."""
    df = df.copy()
    continuous = get_continuous_sensors(df)
    df[continuous] = scaler.transform(df[continuous])
    return df


def prepare_train_test() -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Load data, clean, normalize, and return train/test splits.

    Train = normal data (cleaned). Scaler fitted here.
    Test = attack data (includes both normal and attack seconds).
    """
    normal, attack = load_or_create_processed()
    train = clean_normal_data(normal)

    scaler = fit_scaler(train)

    train_scaled = apply_scaler(train, scaler)
    test_scaled = apply_scaler(attack, scaler)

    return train_scaled, test_scaled, scaler


def save_prepared_data(
    train: pd.DataFrame, test: pd.DataFrame, scaler: StandardScaler,
) -> None:
    """Save scaled datasets and scaler to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(PROCESSED_DIR / "train_scaled.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test_scaled.parquet", index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    print(f"Train: {train.shape[0]:,} rows x {train.shape[1]} cols")
    print(f"Test:  {test.shape[0]:,} rows x {test.shape[1]} cols")
    print(f"Scaler saved: {MODELS_DIR / 'scaler.joblib'}")


if __name__ == "__main__":
    print("Preparing train/test data...")
    train, test, scaler = prepare_train_test()
    save_prepared_data(train, test, scaler)

    continuous = get_continuous_sensors(train)
    print(f"\nContinuous sensors: {len(continuous)}")
    print(f"Discrete sensors: {len(get_discrete_sensors(train))}")

    print("\nTrain scaled stats (should be ~0 mean, ~1 std):")
    print(train[continuous].describe().loc[["mean", "std"]].round(3).to_string())
