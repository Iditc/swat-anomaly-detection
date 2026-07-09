"""Load and prepare SWaT dataset for analysis."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Timestamp column to datetime and set as index."""
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Normal/Attack label to binary (0=Normal, 1=Attack)."""
    label_map = {"Normal": 0, "Attack": 1}
    df["label"] = df["Normal/Attack"].map(label_map).fillna(1).astype(int)
    df = df.drop(columns=["Normal/Attack"])
    return df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normal and attack CSVs, return cleaned DataFrames."""
    normal_path = RAW_DIR / "normal.csv"
    attack_path = RAW_DIR / "attack.csv"

    if not normal_path.exists() or not attack_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {RAW_DIR}. "
            "Download SWaT dataset from Kaggle and place in src/data/raw/"
        )

    normal_df = pd.read_csv(normal_path)
    attack_df = pd.read_csv(attack_path)

    for df in [normal_df, attack_df]:
        clean_column_names(df)
        parse_timestamp(df)

    normal_df = encode_labels(normal_df)
    attack_df = encode_labels(attack_df)

    return normal_df, attack_df


def load_or_create_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed parquet if exists, otherwise create from raw."""
    normal_parquet = PROCESSED_DIR / "normal.parquet"
    attack_parquet = PROCESSED_DIR / "attack.parquet"

    if normal_parquet.exists() and attack_parquet.exists():
        normal_df = pd.read_parquet(normal_parquet)
        attack_df = pd.read_parquet(attack_parquet)
        return normal_df, attack_df

    normal_df, attack_df = load_raw_data()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    normal_df.to_parquet(normal_parquet, index=False)
    attack_df.to_parquet(attack_parquet, index=False)

    return normal_df, attack_df


SENSOR_START_TIME = pd.Timestamp("2015-12-28 10:00:00")

DISCRETE_COLUMNS = ["MV101", "MV201", "P201", "P202", "P204", "MV303"]


def clean_normal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop early period with missing sensors, remove duplicates, fix dtypes."""
    df = df[df["Timestamp"] >= SENSOR_START_TIME].copy()
    df = df.drop_duplicates()
    for col in DISCRETE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def get_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Return list of sensor/actuator feature columns (exclude Timestamp and label)."""
    exclude = {"Timestamp", "label", "Normal/Attack"}
    return [col for col in df.columns if col not in exclude]


if __name__ == "__main__":
    normal, attack = load_or_create_processed()
    print(f"Normal: {normal.shape}, Attack: {attack.shape}")
    print(f"Sensor columns: {len(get_sensor_columns(normal))}")
    print(f"Normal date range: {normal['Timestamp'].min()} to {normal['Timestamp'].max()}")
    print(f"Attack date range: {attack['Timestamp'].min()} to {attack['Timestamp'].max()}")
