# SWaT Dataset — Detailed EDA Report

This report documents the full exploratory data analysis (EDA) of the SWaT (Secure Water Treatment) dataset. It is written for anyone unfamiliar with the project — no prior knowledge of the dataset, water treatment, or anomaly detection is assumed.

---

## What is SWaT?

SWaT is a **real water treatment plant** built at Singapore University of Technology and Design (SUTD). It takes raw water and purifies it through 6 physical stages:

1. **P1 — Raw water supply:** Water enters the system. A valve (MV101) controls inflow, a pump (P101) moves water forward, and a sensor (LIT101) measures the water level in the tank.
2. **P2 — Chemical treatment:** Chemicals are added to treat the water. Sensors measure pH (AIT201), ORP (AIT202), and conductivity (AIT203).
3. **P3 — Ultrafiltration (UF):** Water is filtered through membranes. Pressure (DPIT301) drives the flow (FIT301), and the tank level (LIT301) is monitored.
4. **P4 — Dechlorination:** Chlorine is removed using UV light (UV401) and chemical treatment. Sensors monitor the process (AIT401, AIT402).
5. **P5 — Reverse Osmosis (RO):** Water is purified at the molecular level. Multiple flow (FIT501-504) and pressure (PIT501-503) sensors monitor the process.
6. **P6 — Backwash:** Filters are cleaned periodically.

**51 sensors and actuators** report readings **every second**. The dataset contains:
- **Normal data:** 11 days of the plant running without interference (~395K clean rows)
- **Attack data:** 4 days during which researchers executed 36 real cyberattacks (~55K rows)

The goal is to build a system that detects when the plant is under attack — not by looking at one reading in isolation, but by recognizing abnormal **patterns over time**.

---

## Part 1 — Data Quality Issues

Before analysis, we checked the data for problems.

### Missing values — 7 sensors didn't record for the first 5.75 days

Seven columns (MV101, AIT201, MV201, P201, P202, P204, MV303) had no data (NaN) for the first 991,800 rows. This isn't random — it's one contiguous block. These sensors simply weren't turned on before December 28, 2015, at 10:00 AM.

**Resolution:** Drop all rows before that timestamp. This gives us complete data for all 51 sensors.

### Duplicate rows — 495,000 identical duplicates

Nearly 500K timestamps appeared twice with fully identical values across all 51 columns. These are pure duplicates (likely an export bug), not conflicting readings.

**Resolution:** `drop_duplicates()` — no information is lost.

### Data type mismatch — caused by NaN, not real

Six discrete columns were stored as float64 in normal data but int64 in attack data. This happened because Python's NaN forces integer columns to float. After dropping the NaN period, the columns are cast to int and the issue disappears.

**After cleanup:** 395,298 normal rows + 54,621 attack rows, all 51 sensors populated.

---

## Part 2 — Class Distribution

| Class | Rows | Percentage |
|-------|------|-----------|
| Normal | 395,298 | 87.9% |
| Attack | 54,621 | 12.1% |

The imbalance ratio is **7.2:1** (normal to attack). This means accuracy is a misleading metric — a model that always predicts "normal" would be 88% accurate but completely useless. We use **F1 macro** as the primary evaluation metric instead.

The cleanup improved the ratio from 25:1 (raw data) to 7.2:1 by removing 5.75 days of normal-only data.

---

## Part 3 — Time-Series Behavior

Unlike tabular datasets, this data has **temporal order** — each row is one second, and the order matters. We plotted 6 key sensors across the full timeline to understand how attacks look over time.

![Sensor time-series with attack periods](figures/sensor_timeseries.png)

### Key observations per sensor

| Sensor | Normal behavior | Attack behavior |
|--------|----------------|-----------------|
| LIT101 (water level) | Smooth fill/drain cycle, ~42 min period | Sharp drops or frozen values |
| FIT101 (flow rate) | Stable flow when valve is open | Flow drops to zero while valve reports open |
| AIT201 (pH) | Gradual chemical changes | Sudden jumps without chemical dosing |
| LIT301 (UF tank) | Regular oscillation | Level stuck at constant value |
| FIT401 (dechlorination flow) | Steady flow | Flow spikes or disappears |
| AIT501 (RO analysis) | Stable readings | Large deviations from cascading upstream attacks |

### 34 distinct attack periods

The attack data contains 34 separate attack periods, ranging from under 2 minutes to nearly 10 hours.

### Time-series analyses and what they teach us

We tested four time-series analysis techniques on LIT101 to understand which patterns distinguish normal from attack:

**Autocorrelation** measures how "smooth" a signal is over time. LIT101 has autocorrelation of 0.979 at lag 60s in normal data, dropping to 0.946 during attacks. A smoothly changing water level suddenly becoming erratic signals an anomaly.

**Periodicity (FFT)** reveals repeating cycles. LIT101 has a dominant cycle of ~42 minutes — the pump fill/drain cycle. When an attack disrupts this cycle, the system deviates from its expected pattern.

**Cross-correlation** measures physical relationships between sensors. MV101 (valve) and FIT101 (flow) have a correlation of 0.961 at 1-second lag — when you open the valve, water flows. If the valve is open but there's no flow, a physical law has been violated.

**Rate of change** measures how fast values change. The normal p99 rate of change for LIT101 is 1.18, while the attack p99 is actually lower at 1.00. This reveals a specific attack type: the attacker *freezes* the sensor value, so it stops changing entirely.

---

## Part 4 — Sensor Distributions (KS Test)

We compared the statistical distribution of each sensor between normal and attack data using the **Kolmogorov-Smirnov (KS) test**. KS measures the maximum distance between two distributions: a score of 0 means identical, 1 means completely different.

![Top 12 sensor distribution comparisons](figures/sensor_distributions.png)

### What KS reveals

- **AIT202 (ORP)** has the highest KS (0.348) — its distribution shifts dramatically during attacks
- **LIT101 (water level)** KS = 0.325 — attacks cause the water level to reach unusual values
- **AIT501 (RO analysis)** KS = 0.308 — heavily affected even though most attacks target upstream stages

### Cascade effect

Stage P5 (Reverse Osmosis) sensors like AIT501 and AIT502 are among the most affected, even though most attacks target P1 or P2. This is because water flows through the system — an attack on P1 physically propagates downstream. An attacker manipulating the water level in P1 eventually causes abnormal readings in P5, minutes later.

### KS tiers for automatic feature engineering

We use KS scores to automatically decide which features to generate per sensor:

| Tier | KS range | What it means | Features |
|------|----------|--------------|----------|
| High | > 0.2 | Distribution very different during attacks | All features: rolling_mean, rolling_std, rate_of_change, deviation_from_baseline |
| Medium | 0.05–0.2 | Moderate difference | Basic: rolling_mean, rate_of_change |
| Low | < 0.05 | Nearly identical | Raw value only |

This is data-driven, not manual: a script runs KS tests and generates a configuration file. The same infrastructure is reused later for **drift detection** in production.

---

## Part 5 — Sensor Correlations

Sensors that measure related physical processes are correlated. We computed correlations between all 25 continuous sensors and compared them between normal and attack data.

![Correlation heatmap — normal data](figures/correlation_heatmap_normal.png)

### Strongest physical relationships

| Pair | Correlation | Why |
|------|------------|-----|
| PIT501 — PIT503 | 0.993 | Two pressure sensors on the same RO stage — should read nearly identical values |
| DPIT301 — FIT301 | 0.971 | Differential pressure drives ultrafiltration flow — direct cause and effect |
| AIT201 — AIT202 | -0.910 | pH and ORP are chemically inverse — when pH goes up, ORP goes down |
| AIT402 — AIT502 | 0.890 | Chemical readings across stages P4→P5 — water carries chemistry downstream |

These aren't statistical coincidences — they're **laws of physics and chemistry**. When an attack violates these laws, it produces a signal no legitimate operation can produce.

### Correlations that flip during attacks

![Correlation changes — normal vs attack](figures/correlation_changes_top10.png)

The most powerful finding: some correlations **reverse sign** during attacks.

| Pair | Normal | Attack | What happened |
|------|--------|--------|--------------|
| AIT402 — PIT502 | +0.31 | -0.85 | Was positively correlated, became strongly negative |
| AIT402 — AIT501 | +0.29 | -0.84 | Chemical relationship reversed — physically impossible |
| FIT101 — LIT301 | +0.52 | -0.47 | Flow and tank level relationship broke |

A correlation that flips sign means the physical relationship has been violated. This is the basis for `sensor_pair_residual` features: learn the normal relationship between two sensors, then measure how far reality deviates from expectation in real time.

![Correlation change heatmap](figures/correlation_change_heatmap.png)

---

## Part 6 — Discrete Features (Actuators)

26 sensors are discrete (on/off or open/close): motorized valves (MV), pumps (P), and UV dechlorinator. Unlike continuous sensors that measure values, these control the physical system.

![Discrete sensor state changes](figures/discrete_state_changes.png)

### Dramatic state changes during attacks

| Sensor | Normal %ON | Attack %ON | What it means |
|--------|-----------|-----------|--------------|
| P403 | 0% | 100% | A pump that **never** runs in normal operation runs 100% of the time during attack |
| P204, P206 | 100% | 0.1% | Pumps that are **always** on shut down during attacks |
| UV401 | 100% | 39.2% | UV dechlorinator turns off — directly risks water safety |
| MV304 | 3.5% | 63.6% | A rarely-opened valve stays open during attack |

### Physical contradictions

Some actuator + sensor combinations are physically impossible:

- **P501 off + RO flow exists:** Increased from 0.01% to 2.34% during attacks. If the pump is off, water cannot flow — but the flow sensor reports flow. Either the pump state or the flow reading is compromised.
- **P301 on + no UF flow:** Jumped from 13.6% to 68.5%. The pump is running but no water is moving — something is blocking the system or the sensor is lying.

### Always-constant sensors

P204 and P206 never change state in normal operation (always OFF). Any state change at all is an immediate anomaly — no threshold needed, no complex model required.

---

## Part 7 — Cascade Timing

When an attack hits one stage, how fast does the effect propagate to other stages? This question directly determines the window sizes for our rolling features.

### Propagation speeds

| Path | Median delay | Explanation |
|------|-------------|-------------|
| P4 → P5 | 145s (~2.5 min) | Adjacent stages — water flows directly from dechlorination to reverse osmosis |
| P3 → P4 | 362s (~6 min) | One stage apart — takes time for water to travel |
| P3 → P6 | 340s (~5.5 min) | Cross-system — backwash is affected by upstream changes |
| P5 → P4 | 2s | Feedback effect — pressure changes propagate almost instantly |

### What this means for feature engineering

Rolling features need windows that match these timescales:
- **60-second window:** Captures immediate effects and sharp changes within a single stage
- **120-second window:** Captures fast cascades between adjacent stages (P4↔P5)
- **300-second window:** Captures most cross-stage propagation (covers the ~5-6 minute median)

Using only a short window would miss slow cascades. Using only a long window would blur fast attacks. The combination covers the full range.

---

## Part 8 — Attack-by-Attack Profiling

Each of the 34 attack periods was analyzed individually to understand the diversity of attacks.

### Attack statistics

| Metric | Min | Median | Max |
|--------|-----|--------|-----|
| Duration | 100s (~2 min) | 465s (~8 min) | 35,899s (~10 hrs) |
| Sensors affected | 0 | 4 | 25 (out of 51) |
| Stages affected | 0 | 3 | 5 (out of 6) |

### Attacks per stage

| Stage | Attacks | % of total | Why |
|-------|---------|-----------|-----|
| P1 | 22 | 65% | Entry point of the system — most accessible target |
| P2 | 18 | 53% | Chemical treatment — high-impact target |
| P3 | 16 | 47% | Ultrafiltration — central to the process |
| P4 | 15 | 44% | Dechlorination — safety-critical |
| P6 | 9 | 26% | Backwash — less impactful |
| P5 | 7 | 21% | RO — most affected indirectly via cascade, less directly targeted |

### Two categories of attacks

**Subtle attacks (35%):** Only 1-2 sensors are affected. These are the hardest to detect and require sensitive features like rate_of_change and sensor_pair_residual. A single sensor showing an unusual value might be dismissed as noise — the model needs to be confident from minimal evidence.

**Multi-sensor attacks (65%):** 3 or more sensors change simultaneously. These are easier to detect because multiple signals fire at once. Cross-sensor features (correlations, contradictions) excel here.

### Most frequently affected sensors

P101 (pump, P1), P203 and P205 (pumps, P2) are each affected in ~47% of all attacks. These are actuator-control attacks where the attacker turns pumps on or off to disrupt the physical process.

---

## Part 9 — Normal Operating Modes

The plant doesn't run in a single steady state. It cycles between three operating modes based on the P1 water tank level (LIT101):

| Mode | % of time | What's happening |
|------|----------|-----------------|
| Filling (43%) | Water flows in, level rises. Valve MV101 is open 90% of the time. |
| Draining (43%) | Water is pumped out, level drops. Pump P101 is on 91% of the time. |
| Steady (14%) | Level is stable — transitional moment between filling and draining. |

### Why this matters for anomaly detection

**Stages P1-P3 behave very differently between modes.** FIT101 (P1 flow rate) has a mean of 2.30 during filling but 1.34 during draining — a difference of 0.84 standard deviations. If the model doesn't know which mode the system is in, it might flag a normal mode transition as an anomaly.

**Stages P4-P6 are mode-independent.** Their sensor readings barely change between filling and draining. These stages run at a constant pace regardless of what P1 is doing.

**Practical implication:** Adding an `operating_mode` feature (filling/draining/steady) helps the model learn that certain value ranges are normal in certain modes. Without it, the model sees filling→draining transitions as suspicious because the sensor values shift significantly.

---

## Summary — What We Learned and What Comes Next

### Key findings that drive the model design

1. **Time matters:** Single-row analysis is insufficient. Attacks manifest as patterns over time (frozen values, broken cycles, gradual drift). Rolling window features are essential.

2. **Physics matters:** The strongest anomaly signals come from violations of physical laws — pumps on with no flow, correlated sensors suddenly disagreeing, relationships flipping sign.

3. **Cascade matters:** Attacks propagate between stages in 0-6 minutes. Features must use multiple window sizes (60s, 120s, 300s) to capture this.

4. **Context matters:** The same sensor value can be normal or abnormal depending on the operating mode (filling vs draining) and the state of other sensors.

5. **Diversity matters:** Attacks range from subtle (1 sensor, 2 minutes) to massive (25 sensors, 10 hours). The model needs both sensitive single-sensor features and robust cross-sensor features.

### Feature engineering plan (Day 2)

| Feature type | What it captures | Source |
|-------------|-----------------|--------|
| `rolling_mean`, `rolling_std` | Smoothed values, volatility | Baseline behavior |
| `rate_of_change` | Sharp jumps or frozen values | Time-series analysis |
| `deviation_from_baseline` | How far from training mean | Distribution analysis |
| `rolling_autocorr` | Loss of signal smoothness | Autocorrelation analysis |
| `cycle_deviation` | Broken pump cycles | Periodicity analysis |
| `cusum` | Cumulative drift | Change point detection |
| `sensor_pair_residual` | Broken physical relationships | Correlation analysis |
| `actuator_sensor_contradiction` | Pump on + no flow | Discrete feature analysis |
| `unexpected_state_change` | Always-constant sensor changes | Discrete feature analysis |
| `switching_rate` | Abnormal on/off frequency | Discrete feature analysis |
| `operating_mode` | Current system state | Operating mode analysis |

The KS-driven tier system automatically decides which features to generate per sensor, making the pipeline adaptable to new sensors without code changes.
