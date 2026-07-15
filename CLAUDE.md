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
