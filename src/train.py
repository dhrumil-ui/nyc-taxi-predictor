"""
NYC Taxi Trip Duration Predictor — Training Script
Uses real NYC TLC (Taxi & Limousine Commission) open data.

Data source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Free, public domain, updated monthly.

Run: python src/train.py
"""

import pandas as pd
import numpy as np
import requests
import os
import json
import joblib
import warnings
from io import BytesIO
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# NYC TLC Yellow Taxi data — Parquet files (much faster than CSV)
# Format: yellow_tripdata_YYYY-MM.parquet
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
# Use 2 months of data for training (adjust for more accuracy)
DATA_FILES = [
    "yellow_tripdata_2024-01.parquet",
    "yellow_tripdata_2024-02.parquet",
]

# NYC TLC Taxi Zone lookup (borough mapping)
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"

# Borough to simplified zone mapping
BOROUGH_ZONE_MAP = {
    "Manhattan": 1,
    "Brooklyn": 2,
    "Queens": 3,
    "Bronx": 4,
    "Staten Island": 5,
    "EWR": 8,
}

# Known airport location IDs in TLC data
AIRPORT_LOCATION_IDS = {132, 138, 1}  # JFK, LaGuardia, Newark

FEATURE_COLS = [
    "distance_miles",
    "hour",
    "day_of_week",
    "month",
    "passenger_count",
    "pickup_zone",
    "dropoff_zone",
    "is_rush_hour",
    "is_weekend",
    "is_airport_trip",
    "is_night",
    "is_manhattan",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]

TARGET_COL = "trip_duration_minutes"


# ─── Step 1: Download Data ─────────────────────────────────────────────────────
def download_data():
    """Download NYC TLC trip data files."""
    dfs = []
    for filename in DATA_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename} already cached")
        else:
            url = f"{TLC_BASE_URL}/{filename}"
            print(f"  ↓ Downloading {filename} (~50MB)...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✓ {filename} saved")

        df = pd.read_parquet(filepath)
        dfs.append(df)
        print(f"    Loaded {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total raw rows: {len(combined):,}")
    return combined


def load_zone_lookup():
    """Load NYC taxi zone to borough mapping."""
    filepath = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
    if not os.path.exists(filepath):
        print("  ↓ Downloading zone lookup...")
        df = pd.read_csv(ZONE_URL)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)

    zone_map = dict(zip(df["LocationID"], df["Borough"]))
    return zone_map


# ─── Step 2: Data Cleaning ────────────────────────────────────────────────────
def clean_data(df):
    """Remove invalid trips based on NYC TLC data quality rules."""
    print("\n[2] Cleaning data...")
    initial = len(df)

    # Parse datetimes
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Calculate duration
    df["trip_duration_seconds"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    df[TARGET_COL] = df["trip_duration_seconds"] / 60

    # Remove invalid trips
    df = df[df[TARGET_COL] >= 1]           # min 1 minute
    df = df[df[TARGET_COL] <= 180]         # max 3 hours
    df = df[df["trip_distance"] >= 0.1]    # min 0.1 miles
    df = df[df["trip_distance"] <= 100]    # max 100 miles
    df = df[df["passenger_count"] >= 1]
    df = df[df["passenger_count"] <= 6]
    df = df[df["fare_amount"] >= 2.50]     # min NYC fare
    df = df.dropna(subset=["PULocationID", "DOLocationID"])

    removed = initial - len(df)
    print(f"  Removed {removed:,} invalid rows ({removed/initial*100:.1f}%)")
    print(f"  Clean rows: {len(df):,}")
    return df


# ─── Step 3: Feature Engineering ─────────────────────────────────────────────
def haversine_approx(lat1, lon1, lat2, lon2):
    """Fast vectorized haversine distance in miles."""
    R = 3959
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a.clip(0, 1)))


# Approximate borough centroid coordinates for geospatial features
BOROUGH_COORDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "EWR": (40.6895, -74.1745),
    "Unknown": (40.7128, -74.0060),
}


def engineer_features(df, zone_map):
    """Build ML features from raw TLC data."""
    print("\n[3] Engineering features...")

    dt = df["tpep_pickup_datetime"]

    # Time features with cyclical encoding
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Boolean flags
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_airport_trip"] = (
        df["PULocationID"].isin(AIRPORT_LOCATION_IDS) |
        df["DOLocationID"].isin(AIRPORT_LOCATION_IDS)
    ).astype(int)

    # Borough mapping
    df["pickup_borough"] = df["PULocationID"].map(zone_map).fillna("Unknown")
    df["dropoff_borough"] = df["DOLocationID"].map(zone_map).fillna("Unknown")
    df["is_manhattan"] = (
        (df["pickup_borough"] == "Manhattan") |
        (df["dropoff_borough"] == "Manhattan")
    ).astype(int)

    # Zone encoding
    df["pickup_zone"] = df["pickup_borough"].map(BOROUGH_ZONE_MAP).fillna(0).astype(int)
    df["dropoff_zone"] = df["dropoff_borough"].map(BOROUGH_ZONE_MAP).fillna(0).astype(int)

    # Distance (use reported distance, fall back to geospatial)
    df["distance_miles"] = df["trip_distance"].clip(0.1, 100)

    # Passenger count
    df["passenger_count"] = df["passenger_count"].fillna(1).clip(1, 6).astype(int)

    print(f"  Features engineered: {FEATURE_COLS}")
    return df


# ─── Step 4: Train Model ──────────────────────────────────────────────────────
def train_model(df):
    """Train XGBoost model on engineered features."""
    print("\n[4] Training model...")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Sample up to 500k rows to keep training fast
    if len(X) > 500_000:
        print(f"  Sampling 500k rows from {len(X):,} for speed...")
        idx = np.random.choice(len(X), 500_000, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # fast histogram-based
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  📊 Test Metrics:")
    print(f"     RMSE:  {rmse:.2f} minutes")
    print(f"     MAE:   {mae:.2f} minutes")
    print(f"     R²:    {r2:.4f}")

    metrics = {
        "rmse_minutes": round(rmse, 3),
        "mae_minutes": round(mae, 3),
        "r2": round(r2, 4),
        "n_estimators": model.best_iteration,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    return model, metrics, X_test, y_test


# ─── Step 5: Save Artifacts ───────────────────────────────────────────────────
def save_artifacts(model, metrics):
    """Save model and metadata for the Streamlit app."""
    print("\n[5] Saving artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, "taxi_model.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_COLS, f)
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Model saved to {MODEL_DIR}/taxi_model.pkl")
    print(f"  ✓ Metrics: {metrics}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NYC TAXI TRIP DURATION PREDICTOR — TRAINING PIPELINE")
    print("=" * 60)

    print("\n[1] Loading data...")
    raw_df = download_data()

    zone_map = load_zone_lookup()

    clean_df = clean_data(raw_df)
    feat_df = engineer_features(clean_df, zone_map)

    model, metrics, X_test, y_test = train_model(feat_df)
    save_artifacts(model, metrics)

    print("\n" + "=" * 60)
    print("✅ Training complete! Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
