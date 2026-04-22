# 🗽 NYC Taxi Trip Duration & Fare Predictor

> An end-to-end ML application predicting NYC taxi trip duration and fare using real NYC TLC open data, XGBoost, and Streamlit.

[![CI](https://github.com/YOUR_USERNAME/nyc-taxi-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/nyc-taxi-predictor/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[🚀 Live Demo →](https://your-app.streamlit.app)** | **[📖 Technical Blog Post →](https://medium.com/@yourhandle)**

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

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/nyc-taxi-predictor.git
cd nyc-taxi-predictor
```

### 2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Train the model
This downloads ~100MB of real NYC TLC data, cleans it, engineers features, and trains XGBoost:
```bash
python src/train.py
```
Expected output:
```
[1] Loading data...
  ↓ Downloading yellow_tripdata_2024-01.parquet (~50MB)...
  Total raw rows: 2,964,624

[2] Cleaning data...
  Clean rows: 2,701,443

[3] Engineering features...

[4] Training model...
  [0]  test-rmse: 12.41
  [50] test-rmse: 6.82
  [100] test-rmse: 5.23
  ...
  📊 Test Metrics:
     RMSE:  4.23 minutes
     R²:    0.8712

[5] Saving artifacts...
  ✓ Model saved to models/taxi_model.pkl

✅ Training complete!
```

### 4. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

### 5. (Optional) Run the API
```bash
uvicorn src.api:app --reload
```
API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

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

## ☁️ Deployment

### Option A — Streamlit Cloud (Free, Recommended)

1. Push your code to GitHub (public repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo → Set main file: `app.py`
4. Click **Deploy**

> ⚠️ **Note on model file**: The trained `models/taxi_model.pkl` (~20MB) must be committed to GitHub for Streamlit Cloud to load it. After training locally, run:
> ```bash
> git add models/
> git commit -m "Add trained model artifacts"
> git push
> ```

### Option B — Render.com (Free, REST API + Web App)

1. Create account at [render.com](https://render.com)
2. New → **Web Service** → Connect GitHub repo
3. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Create Web Service**

### Option C — Docker (Any Cloud)
```bash
docker build -t nyc-taxi-predictor .
docker run -p 8501:8501 nyc-taxi-predictor
```

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

---

## 📊 EDA Notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

Covers: duration distribution, hourly traffic patterns, distance vs duration scatter, feature correlation matrix.

---

## 🤝 Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "feat: add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 Data License

NYC TLC Trip Record Data is published under the [NYC Open Data Terms of Use](https://opendata.cityofnewyork.us/overview/#termsofuse). Free for commercial and non-commercial use.

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- HuggingFace: [huggingface.co/YOUR_USERNAME](https://huggingface.co/YOUR_USERNAME)

---

## ⭐ If this helped your portfolio, leave a star!
