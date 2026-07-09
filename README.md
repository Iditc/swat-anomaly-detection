# SWaT Anomaly Detection — Full Product

End-to-end anomaly detection system for industrial control systems (OT/ICS).  
From raw sensor data to a deployed, monitored product on AWS.

## Overview

This project detects cyberattacks on a water treatment system using the **SWaT dataset** (Secure Water Treatment, iTrust SUTD). It covers the full ML lifecycle: data exploration, model training, API serving, containerization, monitoring, and cloud deployment.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data & ML | Python, pandas, scikit-learn, TensorFlow |
| API | FastAPI |
| Database | PostgreSQL |
| Dashboard | Grafana |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2 |
| Monitoring | KS-test drift detection |

## Dataset — SWaT (Secure Water Treatment)

The **SWaT testbed** is a real water treatment plant built at SUTD university (Singapore). It purifies water through 6 physical stages — from raw water intake to reverse osmosis. 51 sensors and actuators report readings every second.

| Stage | Process | Key Sensors |
|-------|---------|-------------|
| P1 | Raw water supply | FIT101 (flow), LIT101 (water level), MV101 (valve), P101-P102 (pumps) |
| P2 | Chemical treatment | AIT201-203 (pH, ORP, conductivity), FIT201, P201-P206 |
| P3 | Ultrafiltration | DPIT301 (pressure), FIT301, LIT301, MV301-304 |
| P4 | Dechlorination | AIT401-402, FIT401, LIT401, P401-404, UV401 |
| P5 | Reverse osmosis | AIT501-504, FIT501-504, P501-502, PIT501-503 |
| P6 | Backwash | FIT601, P601-P603 |

### Sensor types

- **FIT** — Flow rate (continuous)
- **LIT** — Water level in tank (continuous)
- **AIT** — Chemical analysis: pH, ORP, conductivity (continuous)
- **PIT / DPIT** — Pressure / differential pressure (continuous)
- **MV** — Motorized valve (discrete: open/close)
- **P** — Pump (discrete: on/off)
- **UV** — UV dechlorinator (discrete: on/off)

### Data structure

Each row = **one second** of simultaneous readings from all 51 sensors.

| File | Raw Rows | Description |
|------|----------|-------------|
| `normal.csv` | 1,387,098 | 11 days of normal operation (training data) |
| `attack.csv` | 54,621 | Rows labeled as attack during 36 cyberattack scenarios |

### Attack types

The researchers executed 36 real cyberattacks on the physical system, including:

- **Sensor manipulation** — Attacker falsifies a sensor reading. Example: water level sensor reports 500 when the real level is 300, causing the system to shut off a pump while the tank is actually empty.
- **Actuator control** — Attacker directly opens/closes a valve or starts/stops a pump. The sensors report truthfully, but the action itself is abnormal (e.g., a pump turns on when the tank is empty).
- **Combined attacks** — Both sensor falsification and actuator manipulation at the same time.

### What we detect

Not "is this row fake" — but **"is the system behaving abnormally over time."** Examples:

- LIT101 (water level) drops 50% in 3 seconds — physically impossible
- A pump is running but there is zero flow — contradiction
- pH value jumps without any chemical dosing — unexplained change

This is why **time-window features** (rolling mean, rolling std, rate of change) are critical — a single row in isolation may look normal, but the pattern over time reveals the attack.

## EDA Findings

### 1. Data quality issues

| Issue | Details | Resolution |
|-------|---------|------------|
| **Missing values** | 7 columns (`MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, `MV303`) have NaN for the first 5.75 days (991,800 rows). These sensors simply weren't recording before 28/12/2015 10:00. All 7 share the exact same NaN pattern — not random, one contiguous block. | Drop rows before 28/12/2015 10:00 (keep only the period with complete data) |
| **Duplicate rows** | 495,000 timestamps appear twice with fully identical values across all 51 columns. Pure duplicates — likely an export bug. | `drop_duplicates()` — no information lost |
| **dtype mismatch** | 6 discrete columns (`MV101`, `MV201`, `P201`, `P202`, `P204`, `MV303`) are `float64` in normal data but `int64` in attack data. Caused by NaN forcing float — not a real type difference. | Resolves automatically after dropping the NaN period and casting to int |

**After cleanup:** 395,298 clean normal rows + 54,621 attack rows, all 51 sensors fully populated.

### 2. Class distribution (after cleanup)

| Class | Rows | Percentage |
|-------|------|-----------|
| Normal | 395,298 | 87.9% |
| Attack | 54,621 | 12.1% |

**Imbalance ratio:** 7.2:1 (Normal:Attack) — still imbalanced, so **F1 macro** is the primary evaluation metric, not accuracy. The cleanup itself improved the ratio significantly (raw data was 25:1).

## Project Structure

```
swat-anomaly-detection/
├── src/
│   ├── preprocessing/   # Data loading, cleaning, feature engineering
│   ├── eda/             # Exploratory data analysis
│   ├── models/          # Isolation Forest, Autoencoder, LightGBM
│   ├── api/             # FastAPI application
│   ├── monitoring/      # Model drift detection
│   └── utils/           # Shared utilities
├── tests/               # Unit + integration tests
├── docker/              # Dockerfiles
├── grafana/             # Dashboard configuration
├── scripts/             # Data simulator, demo scripts
└── results/             # Figures and metrics
```

## Setup

```bash
git clone https://github.com/Iditc/swat-anomaly-detection.git
cd swat-anomaly-detection
pip install -r requirements.txt
```

## Status

Under development
