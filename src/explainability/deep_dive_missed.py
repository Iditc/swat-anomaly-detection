"""Deep dive into the 7 stubbornly missed attack windows."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.load_data import clean_normal_data, get_sensor_columns
from src.preprocessing.prepare_data import get_continuous_sensors

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "src" / "data" / "processed"

MISSED_WINDOWS = [10, 11, 14, 21, 23, 25, 28]


def get_attack_windows(y_true):
    windows = []
    in_attack = False
    start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_attack:
            start = i
            in_attack = True
        elif y_true[i] == 0 and in_attack:
            windows.append((start, i - 1))
            in_attack = False
    if in_attack:
        windows.append((start, len(y_true) - 1))
    return windows


def main():
    normal = pd.read_parquet(PROCESSED_DIR / "normal.parquet")
    attack = pd.read_parquet(PROCESSED_DIR / "attack.parquet")
    normal = clean_normal_data(normal)

    train_df = normal.copy()
    attack_start = attack["Timestamp"].min()
    attack_end = attack["Timestamp"].max()
    normal_in_test = normal[
        (normal["Timestamp"] >= attack_start) & (normal["Timestamp"] <= attack_end)
    ].copy()
    test_raw = pd.concat([normal_in_test, attack], ignore_index=True)
    test_raw = test_raw.sort_values("Timestamp").reset_index(drop=True)

    continuous_cols = get_continuous_sensors(train_df)
    discrete_cols = [c for c in get_sensor_columns(train_df)
                     if c not in continuous_cols]

    train_stats = {}
    for col in continuous_cols:
        train_stats[col] = {
            "mean": train_df[col].mean(),
            "std": train_df[col].std(),
            "min": train_df[col].min(),
            "max": train_df[col].max(),
            "p01": train_df[col].quantile(0.01),
            "p99": train_df[col].quantile(0.99),
        }

    # Normal baseline for discrete sensors
    train_discrete_stats = {}
    for col in discrete_cols:
        vc = train_df[col].value_counts(normalize=True)
        train_discrete_stats[col] = {
            "dominant_state": vc.index[0],
            "dominant_pct": round(vc.iloc[0] * 100, 1),
            "states": sorted(vc.index.tolist()),
        }

    windows = get_attack_windows(test_raw["label"].values)

    for widx in MISSED_WINDOWS:
        i = widx - 1
        start, end = windows[i]
        window = test_raw.iloc[start:end + 1]
        duration = end - start + 1

        # Context: 60s before and after
        ctx_start = max(0, start - 60)
        ctx_end = min(len(test_raw) - 1, end + 60)
        before = test_raw.iloc[ctx_start:start]
        _ = test_raw.iloc[end + 1:ctx_end + 1]

        print(f"\n{'='*80}")
        print(f"  P{widx} — {duration}s — "
              f"{window['Timestamp'].iloc[0]} to {window['Timestamp'].iloc[-1]}")
        print(f"{'='*80}")

        # Continuous sensor analysis
        print("\n  CONTINUOUS SENSORS (deviation from training normal):")
        print(f"  {'Sensor':<12} {'Max σ':>7} {'Mean σ':>7} "
              f"{'Window Mean':>12} {'Train Mean':>12} {'Train Std':>10} {'Pattern'}")
        print(f"  {'-'*85}")

        deviations = []
        for col in continuous_cols:
            ts = train_stats[col]
            if ts["std"] == 0:
                continue
            vals = window[col].values
            max_dev = np.max(np.abs(vals - ts["mean"])) / ts["std"]
            mean_dev = np.mean(np.abs(vals - ts["mean"])) / ts["std"]

            # Detect patterns
            patterns = []
            if window[col].std() == 0 and duration > 10:
                patterns.append("FROZEN")
            if np.max(np.abs(np.diff(vals))) > 3 * ts["std"]:
                patterns.append("SPIKE")
            # Monotonic drift
            diffs = np.diff(vals)
            if len(diffs) > 20:
                pos_ratio = np.mean(diffs > 0)
                if pos_ratio > 0.8:
                    patterns.append("DRIFT_UP")
                elif pos_ratio < 0.2:
                    patterns.append("DRIFT_DOWN")
            # Value outside training range
            if np.any(vals > ts["max"]) or np.any(vals < ts["min"]):
                patterns.append("OUT_OF_RANGE")
            # Unusual variance (much less than normal)
            if window[col].std() > 0 and window[col].std() < ts["std"] * 0.1:
                patterns.append("LOW_VAR")

            pattern_str = ", ".join(patterns) if patterns else ""
            deviations.append((col, max_dev, mean_dev, np.mean(vals),
                              ts["mean"], ts["std"], pattern_str))

        # Sort by max deviation, show top 8
        deviations.sort(key=lambda x: x[1], reverse=True)
        for col, max_d, mean_d, w_mean, t_mean, t_std, pat in deviations[:8]:
            print(f"  {col:<12} {max_d:>6.1f} {mean_d:>7.1f} "
                  f"{w_mean:>12.4f} {t_mean:>12.4f} {t_std:>10.4f} {pat}")

        # Discrete sensor analysis
        disc_anomalies = []
        for col in discrete_cols:
            if col not in window.columns:
                continue
            w_vals = window[col].values
            w_unique = np.unique(w_vals)

            # Check if state differs from normal dominant state
            t_info = train_discrete_stats.get(col, {})
            if not t_info:
                continue

            # State changes during window
            changes = int(np.sum(np.diff(w_vals) != 0))

            # Check before context for state comparison
            b_vals = before[col].values if len(before) > 0 else np.array([])
            state_before = b_vals[-1] if len(b_vals) > 0 else None
            state_start = w_vals[0]
            state_end = w_vals[-1]

            if changes > 0 or (state_before is not None and state_start != state_before):
                disc_anomalies.append({
                    "sensor": col,
                    "before": state_before,
                    "start": state_start,
                    "end": state_end,
                    "changes": changes,
                    "states_in_window": w_unique.tolist(),
                    "train_dominant": t_info["dominant_state"],
                    "train_dominant_pct": t_info["dominant_pct"],
                })

        if disc_anomalies:
            print("\n  DISCRETE SENSORS (state changes):")
            print(f"  {'Sensor':<10} {'Before':>7} {'Start':>7} {'End':>7} "
                  f"{'Changes':>8} {'Train Dom':>10} {'%':>6}")
            print(f"  {'-'*60}")
            for d in disc_anomalies:
                b = str(d["before"]) if d["before"] is not None else "?"
                print(f"  {d['sensor']:<10} {b:>7} {d['start']:>7} {d['end']:>7} "
                      f"{d['changes']:>8} {d['train_dominant']:>10} "
                      f"{d['train_dominant_pct']:>5.1f}%")

        # Physical contradictions check
        print("\n  PHYSICAL CHECKS:")
        # P101/P102 (pumps) vs FIT101 (flow)
        for pump in ["P101", "P102"]:
            if pump in window.columns and "FIT101" in window.columns:
                pump_on = window[pump] == 2
                flow = window["FIT101"]
                pump_on_no_flow = (pump_on & (flow.abs() < 0.1)).sum()
                if pump_on_no_flow > 0:
                    print(f"    {pump} ON but FIT101≈0: {pump_on_no_flow}/{duration} rows")

        # MV101 (valve) vs FIT101
        if "MV101" in window.columns and "FIT101" in window.columns:
            valve_open = window["MV101"] == 2
            flow = window["FIT101"]
            valve_open_no_flow = (valve_open & (flow.abs() < 0.1)).sum()
            if valve_open_no_flow > 0:
                print(f"    MV101 OPEN but FIT101≈0: {valve_open_no_flow}/{duration} rows")

        # Cross-sensor: did something change that normally correlates?
        # Compare window start vs end for key sensors
        key_continuous = ["LIT101", "FIT101", "LIT301", "FIT301", "LIT401",
                          "AIT501", "AIT502", "FIT601"]
        changes_summary = []
        for col in key_continuous:
            if col not in window.columns:
                continue
            ts = train_stats[col]
            if ts["std"] == 0:
                continue
            start_val = window[col].iloc[0]
            end_val = window[col].iloc[-1]
            change_sigma = abs(end_val - start_val) / ts["std"]
            if change_sigma > 0.5:
                direction = "↑" if end_val > start_val else "↓"
                changes_summary.append(
                    f"{col}: {start_val:.2f}→{end_val:.2f} "
                    f"({direction}{change_sigma:.1f}σ)"
                )
        if changes_summary:
            print("\n  VALUE CHANGES (start→end):")
            for c in changes_summary:
                print(f"    {c}")


if __name__ == "__main__":
    main()
