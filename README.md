# 🗽 NYC Taxi Trip Duration & Fare Predictor

> An end-to-end ML application predicting NYC taxi trip duration and fare using real NYC TLC open data, XGBoost, and Streamlit.

[![CI](https://github.com/YOUR_USERNAME/nyc-taxi-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/nyc-taxi-predictor/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🚀 Live Demo →](https://nyc-smart-trip-planner.streamlit.app/)** |

---

## 📌 Project Overview

Every day, **400,000+ taxi trips** happen in New York City. Knowing how long a trip will take and what it will cost is valuable for both riders and drivers. This project builds an ML system that predicts trip duration (in minutes) and estimates the fare using:

- **1M+ real trip records** from the [NYC TLC Open Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **XGBoost** with 16 engineered features including geospatial distance, cyclical time encoding, rush-hour flags, and airport surcharges
- **Streamlit** for an interactive, deployable web app
- **FastAPI** for a production REST endpoint
- **GitHub Actions** for CI/CD

### Results
| Metric | Value |
|--------|-------|
| RMSE | ~4.2 minutes |
| MAE | ~3.1 minutes |
| R² | ~0.87 |
| Training data | 500K trips (2024) |

---

## 🗂 Project Structure

```
nyc-taxi-predictor/
├── app.py                  # Streamlit web app (main entry point)
├── src/
│   ├── train.py            # Full training pipeline (download → clean → engineer → train → save)
│   └── api.py              # FastAPI REST API
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── models/                 # Trained model artifacts (generated after training)
│   ├── taxi_model.pkl
│   ├── feature_names.json
│   └── model_metrics.json
├── data/                   # Downloaded TLC parquet files (gitignored, ~100MB)
├── tests/
│   └── test_pipeline.py    # Pytest unit + integration tests
├── .github/
│   └── workflows/ci.yml    # GitHub Actions CI
├── .streamlit/config.toml  # Streamlit theme
├── Dockerfile
├── requirements.txt
└── README.md
```

---



## 🧠 ML Pipeline Details

### Data Source
NYC TLC Yellow Taxi Trip Records — public domain, updated monthly.
URL: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`

### Feature Engineering

| Feature | Description | Why It Matters |
|---------|-------------|---------------|
| `distance_miles` | Haversine distance between borough centroids | Strongest predictor |
| `hour` | Hour of pickup (0–23) | Traffic varies dramatically |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour | Preserves periodicity (23→0) |
| `day_sin`, `day_cos` | Cyclical encoding of day | Weekend vs weekday patterns |
| `is_rush_hour` | 1 if 7–9am or 4–7pm | 40% longer trips in rush |
| `is_weekend` | 1 for Sat/Sun | Different traffic patterns |
| `is_airport_trip` | 1 if JFK/LGA/EWR involved | Longer, more predictable |
| `is_night` | 1 if after 10pm or before 6am | Faster, empty streets |
| `pickup_zone`, `dropoff_zone` | Borough-level zone (1–8) | Inter-borough patterns |
| `is_manhattan` | 1 if either end is Manhattan | High traffic density |
| `passenger_count` | 1–6 | Minor effect on duration |

### Model: XGBoost Regressor
- `n_estimators`: 500 (with early stopping)
- `max_depth`: 7
- `learning_rate`: 0.05
- `subsample`: 0.8, `colsample_bytree`: 0.8
- `tree_method`: "hist" (fast GPU-compatible)

---


---

## 🧪 Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Tests cover:
- Haversine distance calculation
- Feature engineering correctness (rush hour, airport, weekend flags)
- Cyclical encoding behavior
- Fare estimation formula (NYC TLC rules)
- Fallback prediction within realistic range
- All borough combinations


## 📄 Data License

NYC TLC Trip Record Data is published under the [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse). Free for commercial and non-commercial use.

---



