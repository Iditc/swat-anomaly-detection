# SWaT Anomaly Detection — Full Product

## Goal
End-to-end anomaly detection system for industrial control systems (OT/ICS),
from raw sensor data to a deployed product on AWS.

## Tech Stack
- Python 3.10+
- pandas, numpy, scikit-learn
- TensorFlow/Keras (Autoencoder)
- FastAPI + uvicorn (REST API)
- PostgreSQL + SQLAlchemy (database)
- Docker + Docker Compose (containerization)
- Grafana (dashboard)
- GitHub Actions (CI/CD)
- AWS EC2 (deployment)

## Dataset
- Name: SWaT (Secure Water Treatment)
- Source: iTrust, SUTD / Kaggle
- Type: Time-series sensor data from water treatment testbed
- 51 sensor/actuator features, ~946K rows
- 7 days normal + 4 days with 36 attack scenarios
- Known issues: requires temporal train/test split (no random shuffling)

## Project Structure
- src/data/raw/ — Raw dataset (NOT committed)
- src/data/processed/ — Cleaned data (parquet, NOT committed)
- src/preprocessing/ — Data loading, cleaning, feature engineering
- src/eda/ — Exploratory analysis
- src/models/ — Anomaly detection models
- src/explainability/ — Attack classification + feature attribution
- src/api/ — FastAPI application
- src/monitoring/ — Drift detection
- src/utils/ — Shared utilities
- tests/ — Unit and integration tests
- docker/ — Dockerfiles
- grafana/ — Dashboard configuration
- scripts/ — Data simulator, demo scripts
- results/figures/ — Plots
- results/metrics/ — Model performance

## Data Split Strategy
- normal.csv (Dec 22 - Jan 2, 1.39M rows, all label=0) and attack.csv (Dec 28 - Jan 2, 54K rows, all label=1) overlap in time (Dec 28 - Jan 2)
- Use only the overlap period — combine normal + attack by timestamp
- 80/20 temporal split: Train 358,532 rows (308K normal + 50K attack), Test 89,633 rows (85K normal + 4.7K attack)
- Scaler fitted on normal rows in train split only (no data leakage)
- Same test set used for ALL models (fair comparison)
- Supervised models (LightGBM, RF): train on all train data with scale_pos_weight
- Unsupervised models (Autoencoder): train on label=0 from train only

## Feature Selection
- LightGBM used for feature importance-based selection
- 142 features kept (importance > 0), 114 zero-importance features removed
- Saved in config/selected_features.json, auto-applied by feature_engineering.py
- Feature types largely removed: rate_of_change (21/25 zero), contradiction (4/4 zero), changed (2/2 zero)

## Current Model Results (LightGBM after feature selection)
- F1 Macro: 0.5657
- Precision (Attack): 91.8%, Recall (Attack): 8.6%
- TP: 402, FP: 36, FN: 4,302, TN: 84,893
- Low recall expected — attacks in test period differ from train period (temporal split)

## Models to Compare (same test set)
- Isolation Forest, LightGBM, Autoencoder, One-Class SVM, Random Forest, LSTM

## Key Design Decisions
- Primary metric: F1 macro (imbalanced data)
- Temporal train/test split (no data leakage)
- Unsupervised models trained on normal data only
- Every prediction logged to PostgreSQL
- Model drift detection via KS-test

## Coding Conventions
- All code comments, docstrings, and documentation in English
- Small, single-purpose functions with docstrings
- Each source file < 300 lines
- Use pathlib for all file paths
- Save processed data as parquet
- Do not reload raw CSVs if processed/ already exists

## Git Workflow Rules
- After every logical step, commit with specific file names
- Never use `git add .` or `git add -A`
- Always push after every commit
- Provide exact commands including specific files to add

## README Workflow
- Update the README after every completed EDA part or project milestone
- Organize content in the order the work was done (chronological)
- Each EDA finding gets a numbered subsection under "EDA Findings"
- Include tables with actual numbers, not just descriptions
- Document decisions made and their reasoning

## How to Build This Project (Workflow Rules)
- Work step-by-step — finish one part completely before moving to the next
- After each step: show results, discuss findings, update README, then commit
- Before writing code: explain the concept and make sure it's understood
- When encountering data issues: investigate root cause before deciding on a fix
- Present results visually (tables, charts) — not just raw console output
- Every decision should be explained: what we chose, why, and what alternatives exist
- The user leads the pace — don't skip ahead or combine steps without asking

## AI Workflow Strategy (Claude Code)
- Opus: use for high-level decisions — algorithm selection, architecture,
  debugging complex issues, reviewing results
- Sonnet: use for execution — writing code, refactoring, implementing features
- Rule: start every task with Sonnet. Switch to Opus only when stuck
  or making a design decision that affects the whole project.
