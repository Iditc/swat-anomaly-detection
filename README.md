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

🚧 Under development
