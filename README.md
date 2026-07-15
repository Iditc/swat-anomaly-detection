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

### 3. Time-series behavior

We plotted 6 key sensors across the full timeline (normal + attack periods combined) to understand how attacks manifest over time.

![Sensor time-series with attack periods highlighted](results/figures/sensor_timeseries.png)

**Key observations:**

| Sensor | Stage | Normal behavior | Attack behavior |
|--------|-------|----------------|-----------------|
| LIT101 | P1 | Smooth fill/drain cycle (~42 min period) | Sharp drops or frozen values |
| FIT101 | P1 | Stable flow when valve is open | Flow drops to zero while valve reports open |
| AIT201 | P2 | Gradual chemical changes | Sudden jumps without dosing |
| LIT301 | P3 | Regular UF tank level oscillation | Level stuck at constant value |
| FIT401 | P4 | Steady dechlorination flow | Flow spikes or disappears |
| AIT501 | P5 | Stable RO analysis readings | Large deviations — cascade effect from upstream attacks |

**Attack periods:** 34 distinct attack periods identified, ranging from a few seconds to several hours.

**Time-series analysis for feature engineering:**

We tested four time-series analysis techniques on LIT101 to understand which patterns distinguish normal from attack behavior:

| Analysis | What it measures | Normal vs Attack | Feature it creates |
|----------|-----------------|------------------|-------------------|
| **Autocorrelation** | How "smooth" the signal is over time | Normal: 0.979 at lag 60s / Attack: 0.946 — attack data is less smooth | `rolling_autocorr` — drop in smoothness = anomaly |
| **Periodicity (FFT)** | Repeating cycles in the signal | Dominant cycle of ~42 min (pump fill/drain). Attack breaks the cycle | `cycle_deviation` — how far from expected cycle phase |
| **Cross-correlation** | Physical relationship between sensors | MV101→FIT101 correlation = 0.961 at 1s lag. Valve controls flow | `sensor_pair_residual` — break in physical relationship = anomaly |
| **Rate of change** | Speed of value changes | Normal p99: 1.18 / Attack p99: 1.00 — some attacks *freeze* values | `rate_of_change` + `cusum` — sudden jumps or suspicious stillness |

These analyses will drive automatic feature generation in the preprocessing stage (Day 2).

### 4. Sensor distributions — normal vs attack (KS test)

We compared the distribution of each continuous sensor between normal and attack data using the Kolmogorov-Smirnov (KS) test. KS measures the maximum distance between two distributions — a high KS statistic means the sensor behaves very differently during attacks.

![Top 12 most different sensor distributions](results/figures/sensor_distributions.png)

**KS test results for all 25 continuous sensors:**

| Sensor | KS statistic | Different? | Interpretation |
|--------|-------------|-----------|----------------|
| LIT101 | 0.3253 | YES | Water level distribution shifts significantly during attacks |
| FIT101 | 0.2007 | YES | Flow rate changes when attacks affect Stage 1 |
| AIT201 | 0.1283 | YES | Chemical readings shift during Stage 2 attacks |
| AIT202 | 0.3476 | YES | ORP sensor — one of the most affected |
| AIT203 | 0.0528 | YES | Conductivity — smaller but detectable shift |
| FIT201 | 0.2128 | YES | Flow in chemical treatment changes |
| AIT401 | 0.0284 | YES | Dechlorination analysis — mild shift |
| AIT402 | 0.0380 | YES | Second dechlorination sensor |
| FIT401 | 0.0478 | YES | Dechlorination flow — small shift |
| LIT401 | 0.0694 | YES | Dechlorination tank level |
| AIT501 | 0.3082 | YES | RO analysis — heavily affected (cascade from upstream) |
| AIT502 | 0.2723 | YES | Second RO sensor — also heavily affected |
| AIT503 | 0.0076 | no | Minimal distribution change |
| AIT504 | 0.0043 | no | Minimal distribution change |
| FIT501 | 0.1296 | YES | RO flow — moderate shift |
| FIT502 | 0.1309 | YES | Second RO flow sensor |
| FIT503 | 0.0025 | no | Nearly identical distributions |
| FIT504 | 0.0026 | no | Nearly identical distributions |
| PIT501 | 0.0785 | YES | RO pressure affected |
| PIT502 | 0.0709 | YES | Second pressure sensor |
| PIT503 | 0.0040 | no | Minimal change |
| FIT601 | 0.0131 | YES | Backwash flow — small but detectable |
| DPIT301 | 0.1118 | YES | UF differential pressure — moderate shift |
| FIT301 | 0.1069 | YES | UF flow rate — moderate shift |
| LIT301 | 0.2453 | YES | UF tank level — significant shift |

**Key insight — cascade effect:** Stage P5 (Reverse Osmosis) sensors like AIT501 (KS=0.308) and AIT502 (KS=0.272) are among the most affected, even though most attacks target upstream stages (P1, P2). This is because water flows through the system — an attack on P1 propagates downstream. This makes cross-stage features valuable for detection.

**KS tiers for feature engineering (Day 2):**

| Tier | KS range | Sensors | Features to generate |
|------|----------|---------|---------------------|
| High | > 0.2 | AIT202, LIT101, AIT501, AIT502, LIT301, FIT201, FIT101 | All: rolling_mean, rolling_std, rate_of_change, deviation_from_baseline |
| Medium | 0.05–0.2 | FIT501, FIT502, AIT201, DPIT301, FIT301, PIT501, PIT502, LIT401, AIT203, FIT401 | Basic: rolling_mean, rate_of_change |
| Low | < 0.05 | AIT503, AIT504, FIT503, FIT504, PIT503, AIT401, AIT402, FIT601 | Raw value only |

### 5. Sensor correlations — physical relationships and attack signatures

We computed the correlation matrix for all 25 continuous sensors in normal data, then compared it to attack data to find relationships that break during attacks.

![Sensor correlation heatmap — normal data](results/figures/correlation_heatmap_normal.png)

**Strongest physical relationships (normal data):**

| Pair | Correlation | Physical meaning |
|------|------------|-----------------|
| PIT501 — PIT503 | 0.993 | Two pressure sensors on same RO stage — nearly identical |
| DPIT301 — FIT301 | 0.971 | Differential pressure drives UF flow — direct physical cause |
| FIT503 — FIT504 | 0.938 | Two RO flow sensors — redundant measurement |
| AIT201 — AIT202 | -0.910 | pH vs ORP — inverse chemistry (pH up → ORP down) |
| AIT402 — AIT502 | 0.890 | Chemical readings across stages P4→P5 — cascade |

![Correlation changes during attacks](results/figures/correlation_change_heatmap.png)

**Biggest correlation changes during attacks:**

| Pair | Normal | Attack | Change | What it means |
|------|--------|--------|--------|--------------|
| AIT402 — PIT502 | +0.31 | -0.85 | **1.16** | Correlation flipped sign — physically impossible |
| AIT402 — AIT501 | +0.29 | -0.84 | **1.13** | P4→P5 chemical relationship reversed |
| AIT502 — PIT502 | +0.32 | -0.78 | **1.10** | RO sensors contradicting each other |
| PIT501 — PIT502 | -0.08 | +0.90 | **0.99** | Uncorrelated sensors suddenly lock together |
| FIT101 — LIT301 | +0.52 | -0.47 | **0.99** | P1 flow vs P3 tank level — cross-stage break |

![Top 10 correlation changes — normal vs attack](results/figures/correlation_changes_top10.png)

**Key insight — sign flips:** The strongest attack signal is correlations that **flip sign** (positive → negative or vice versa). A physical relationship that reverses direction is impossible — it means at least one sensor is compromised. This motivates `sensor_pair_residual` features: learn the normal relationship, measure deviation in real time.

**Best sensor pairs for feature engineering (Day 2):**

| Pair | Why it's a good feature |
|------|------------------------|
| AIT402 — PIT502 | Biggest flip (1.16) — P4 chemistry vs P5 pressure |
| AIT402 — AIT501 | Cross-stage chemistry violation (P4→P5) |
| FIT101 — LIT301 | Cross-stage flow-to-level (P1→P3) |
| PIT501 — PIT502 | Redundant sensors — should always agree |
| AIT201 — AIT402 | Chemistry cascade (P2→P4) |

### 6. Discrete features — actuators (valves, pumps, UV)

26 discrete sensors control the physical system: motorized valves (MV, 3 states: 0=closed, 1=transitioning, 2=open), pumps (P, 2 states: 1=off, 2=on), and UV dechlorinator.

![Discrete sensor state changes — normal vs attack](results/figures/discrete_state_changes.png)

**Biggest state changes during attacks:**

| Sensor | Type | Normal %ON | Attack %ON | Diff | Meaning |
|--------|------|-----------|-----------|------|---------|
| P403 | Pump | 0% | 100% | 100% | Never runs normally, always on during attack |
| P204 | Pump | 100% | 0.1% | 99.9% | Always on, shuts down during attack |
| P206 | Pump | 100% | 0.1% | 99.9% | Always on, shuts down during attack |
| UV401 | UV | 100% | 39.2% | 60.8% | UV dechlorinator turns off — water safety risk |
| MV304 | Valve | 3.5% | 63.6% | 60.1% | Rarely open valve suddenly stays open |
| P402 | Pump | 100% | 41.5% | 58.5% | Dechlorination pump turning off |

**Physical contradictions — impossible actuator + sensor combinations:**

| Contradiction | Normal | Attack | What it means |
|--------------|--------|--------|--------------|
| P501 off + RO flow exists | 0.01% (59 rows) | **2.34%** (1,276 rows) | Pump is off but water is flowing — physically impossible |
| P301 on + no UF flow | 13.6% | **68.5%** | Pump running but no flow — 5x increase during attack |

**Always-on sensors that change during attacks:**
- **P204** — always OFF in normal data, turns ON during attack (never seen in training)
- **P206** — always OFF in normal data, turns ON during attack (never seen in training)

Any state change in these sensors is an immediate anomaly indicator.

**Feature engineering ideas (Day 2):**

| Feature | What it captures |
|---------|-----------------|
| `actuator_sensor_contradiction` | Pump on + no flow, or pump off + flow exists |
| `unexpected_state_change` | Always-constant sensor suddenly changes state |
| `switching_rate` | Number of state changes in rolling window — abnormal switching frequency |

### 7. Cascade timing — how fast attacks propagate between stages

Attacks on one stage propagate to downstream stages. We measured the delay:

| Path | Median delay | Meaning |
|------|-------------|---------|
| P4 → P5 | 145s (~2.5 min) | Adjacent stages — fast propagation |
| P3 → P4 | 362s (~6 min) | Cross-stage — moderate delay |
| P3 → P6 | 340s (~5.5 min) | Cross-system — similar timing |

**Window size decision:** Rolling features should use windows of **60s, 120s, and 300s** to capture immediate effects, fast cascades, and full cross-stage propagation.

### 8. Attack-by-attack profiling

Each of the 34 attack periods was analyzed individually:

| Metric | Min | Median | Max |
|--------|-----|--------|-----|
| Duration | 100s | 465s (~8 min) | 35,899s (~10 hrs) |
| Sensors affected | 0 | 4 | 25 |
| Stages affected | 0 | 3 | 5 |

- **P1 most targeted** — 22/34 attacks (65%) affect Stage 1 (raw water supply)
- **35% are subtle** — 12 attacks affect only 1-2 sensors, making them harder to detect
- **Most affected sensors:** P101, P203, P205 (pumps) — affected in ~47% of attacks

### 9. Normal operating modes

The system cycles between three operating modes based on LIT101 water level:

| Mode | Time | Behavior |
|------|------|----------|
| Filling | 43.1% | Water level rising, MV101 open 90%, P101 on 58% |
| Draining | 43.2% | Water level dropping, MV101 open 53%, P101 on 91% |
| Steady | 13.8% | Level stable, transitional state |

**Key finding:** Stages P1-P3 behave very differently between modes (FIT101 varies 0.84σ), while P4-P6 run independently. The anomaly detector must account for operating mode to avoid false alarms during normal mode transitions.

> **Full EDA report with detailed explanations:** see [results/eda_detailed_report.md](results/eda_detailed_report.md)

## Project Structure

```
swat-anomaly-detection/
├── src/
│   ├── preprocessing/   # Data loading, cleaning, feature engineering
│   ├── eda/             # Exploratory data analysis
│   ├── models/          # Isolation Forest, Autoencoder, LightGBM
│   ├── explainability/  # Attack classification + feature attribution
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
