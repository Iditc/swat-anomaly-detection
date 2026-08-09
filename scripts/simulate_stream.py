"""Simulate real-time sensor data streaming to the API.

Reads attack data from the processed parquet file and sends rows
to the /predict endpoint, mimicking a live plant feed.
Usage:
    python scripts/simulate_stream.py --rows 500 --delay 0.1
"""

import argparse
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"
API_URL = "http://localhost:8000"

SENSOR_COLS = [
    "FIT101", "LIT101", "MV101", "P101", "P102",
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201",
    "P201", "P202", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302",
    "MV303", "MV304", "P301", "P302",
    "AIT401", "AIT402", "FIT401", "LIT401",
    "P401", "P402", "P403", "P404", "UV401",
    "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "P501", "P502", "PIT501", "PIT502", "PIT503",
    "FIT601", "P601", "P602", "P603",
]


def load_warmup_data(n_rows: int = 300) -> pd.DataFrame:
    """Load normal data to warm up the rolling buffer."""
    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    return normal.head(n_rows)


def load_test_data(n_rows: int) -> pd.DataFrame:
    """Load attack data with a mix of normal and attack rows."""
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")

    normal_sample = normal.iloc[300:300 + n_rows // 2]
    attack_sample = attack[attack["label"] == 1].head(n_rows // 2)

    combined = pd.concat([normal_sample, attack_sample], ignore_index=True)
    combined = combined.sort_values("Timestamp").reset_index(drop=True)

    return combined.head(n_rows)


def send_reading(client: httpx.Client, row: dict) -> dict:
    """Send a single reading to the API."""
    reading = {col: row[col] for col in SENSOR_COLS}
    response = client.post(f"{API_URL}/predict", json=reading, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Simulate sensor stream")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to send")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between rows")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0=single)")
    args = parser.parse_args()

    print("Warming up buffer with 300 normal rows...")
    warmup = load_warmup_data(300)
    with httpx.Client() as client:
        readings = [
            {col: row[col] for col in SENSOR_COLS}
            for _, row in warmup.iterrows()
        ]
        resp = client.post(
            f"{API_URL}/predict/batch",
            json={"readings": readings},
            timeout=300,
        )
        resp.raise_for_status()
    print("Buffer ready!\n")

    print(f"Loading {args.rows} rows of test data...")
    df = load_test_data(args.rows)
    labels = df["label"].values if "label" in df.columns else [None] * len(df)

    print(f"Streaming to {API_URL} (delay={args.delay}s)\n")

    anomalies_found = 0
    total_sent = 0
    start_time = time.time()

    with httpx.Client() as client:
        if args.batch_size > 0:
            for i in range(0, len(df), args.batch_size):
                batch_df = df.iloc[i:i + args.batch_size]
                readings = [
                    {col: row[col] for col in SENSOR_COLS}
                    for _, row in batch_df.iterrows()
                ]
                resp = client.post(
                    f"{API_URL}/predict/batch",
                    json={"readings": readings},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                anomalies_found += result["anomalies"]
                total_sent += result["total"]
                print(f"  Batch {i // args.batch_size + 1}: "
                      f"{result['total']} rows, {result['anomalies']} anomalies")
                time.sleep(args.delay)
        else:
            for i, (_, row) in enumerate(df.iterrows()):
                result = send_reading(client, row)
                total_sent += 1
                label = labels[i]

                if result["is_anomaly"]:
                    anomalies_found += 1
                    det = result["detected_by"]
                    atype = result["attack_type"] or "unknown"
                    actual = "ATTACK" if label == 1 else "NORMAL"
                    print(f"  [{i+1:4d}] ANOMALY ({det}) — {atype} "
                          f"[actual: {actual}]")

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  ... {i+1}/{len(df)} sent ({elapsed:.1f}s, "
                          f"{anomalies_found} anomalies so far)")

                time.sleep(args.delay)

    elapsed = time.time() - start_time
    print(f"\nDone! {total_sent} readings in {elapsed:.1f}s")
    print(f"Anomalies detected: {anomalies_found}/{total_sent} "
          f"({anomalies_found/total_sent*100:.1f}%)")


if __name__ == "__main__":
    main()
