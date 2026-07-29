"""Check which sensors drift between train and test normal periods."""

from pathlib import Path

import pandas as pd

from src.preprocessing.load_data import clean_normal_data

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = clean_normal_data(normal)

    a_start = attack["Timestamp"].min()
    a_end = attack["Timestamp"].max()
    nt = normal[(normal["Timestamp"] >= a_start) & (normal["Timestamp"] <= a_end)]

    for label, df in [("TRAIN (all normal)", normal),
                      ("TEST normal rows", nt),
                      ("TEST attack rows", attack)]:
        v = df["AIT401"]
        print(f"{label}:")
        print(f"  mean={v.mean():.4f}  std={v.std():.6f}  "
              f"min={v.min():.4f}  max={v.max():.4f}  unique={v.nunique()}")

    print("\nTop sensors shifted between train-normal and test-normal:")
    sensors = [c for c in normal.columns
               if c not in ("Timestamp", "label") and normal[c].dtype == "float64"]
    shifts = {}
    for s in sensors:
        tm = normal[s].mean()
        ts = normal[s].std()
        tem = nt[s].mean()
        if ts > 0:
            shifts[s] = abs(tem - tm) / ts

    for s, v in sorted(shifts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {s:<12} shift={v:>7.1f}σ  "
              f"train_mean={normal[s].mean():>10.2f}  test_mean={nt[s].mean():>10.2f}")


if __name__ == "__main__":
    main()
