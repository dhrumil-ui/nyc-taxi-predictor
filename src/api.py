"""
NYC Taxi Predictor — FastAPI REST API
Serves the trained model as a production REST endpoint.

Run locally:  uvicorn src.api:app --reload
Docs at:      http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import json
import numpy as np
import pandas as pd
import os
from typing import Optional

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Duration Predictor API",
    description="Predicts NYC taxi trip duration using XGBoost trained on real TLC data.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
MODEL_PATH = "models/taxi_model.pkl"
FEATURES_PATH = "models/feature_names.json"
METRICS_PATH = "models/model_metrics.json"

model = None
feature_names = None
metrics = None

@app.on_event("startup")
def load_model():
    global model, feature_names, metrics
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH) as f:
            feature_names = json.load(f)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No model found. Run src/train.py first.")


# ─── Borough Configuration ────────────────────────────────────────────────────
NYC_BOROUGHS = {
    "Manhattan": {"lat": 40.7831, "lon": -73.9712, "zone": 1},
    "Brooklyn": {"lat": 40.6782, "lon": -73.9442, "zone": 2},
    "Queens": {"lat": 40.7282, "lon": -73.7949, "zone": 3},
    "Bronx": {"lat": 40.8448, "lon": -73.8648, "zone": 4},
    "Staten Island": {"lat": 40.5795, "lon": -74.1502, "zone": 5},
    "JFK Airport": {"lat": 40.6413, "lon": -73.7781, "zone": 6},
    "LaGuardia Airport": {"lat": 40.7769, "lon": -73.8740, "zone": 7},
    "Newark Airport (EWR)": {"lat": 40.6895, "lon": -74.1745, "zone": 8},
}


# ─── Request/Response Models ──────────────────────────────────────────────────
class TripRequest(BaseModel):
    pickup_location: str = Field(..., example="Manhattan", description="Pickup borough/location")
    dropoff_location: str = Field(..., example="JFK Airport", description="Dropoff borough/location")
    hour: int = Field(..., ge=0, le=23, example=8, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=1, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, example=6, description="Month (1-12)")
    passenger_count: int = Field(default=1, ge=1, le=6, description="Number of passengers")

    @validator("pickup_location", "dropoff_location")
    def validate_location(cls, v):
        if v not in NYC_BOROUGHS:
            raise ValueError(f"Location must be one of: {list(NYC_BOROUGHS.keys())}")
        return v


class TripResponse(BaseModel):
    duration_minutes: float
    distance_miles: float
    estimated_fare_usd: float
    estimated_tip_usd: float
    estimated_total_usd: float
    pickup_location: str
    dropoff_location: str
    confidence: str
    features_used: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_metrics: Optional[dict]
    available_locations: list


# ─── Feature Engineering ──────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3959
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_features(req: TripRequest):
    pu = NYC_BOROUGHS[req.pickup_location]
    do = NYC_BOROUGHS[req.dropoff_location]
    dist = haversine_distance(pu["lat"], pu["lon"], do["lat"], do["lon"])

    is_rush = 1 if req.hour in [7, 8, 9, 16, 17, 18, 19] else 0
    is_weekend = 1 if req.day_of_week >= 5 else 0
    is_airport = 1 if "Airport" in req.pickup_location or "Airport" in req.dropoff_location else 0
    is_night = 1 if req.hour >= 22 or req.hour <= 5 else 0
    is_manhattan = 1 if "Manhattan" in req.pickup_location or "Manhattan" in req.dropoff_location else 0

    features = {
        "distance_miles": dist,
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "month": req.month,
        "passenger_count": req.passenger_count,
        "pickup_zone": pu["zone"],
        "dropoff_zone": do["zone"],
        "is_rush_hour": is_rush,
        "is_weekend": is_weekend,
        "is_airport_trip": is_airport,
        "is_night": is_night,
        "is_manhattan": is_manhattan,
        "hour_sin": np.sin(2 * np.pi * req.hour / 24),
        "hour_cos": np.cos(2 * np.pi * req.hour / 24),
        "day_sin": np.sin(2 * np.pi * req.day_of_week / 7),
        "day_cos": np.cos(2 * np.pi * req.day_of_week / 7),
    }
    return pd.DataFrame([features]), dist, bool(is_airport)


def estimate_fare(distance, duration_min, is_airport, hour):
    base = 3.00 + 1.70 * distance + 0.50 * (duration_min / 60 * 12) + 0.50 + 1.00
    if is_airport:
        base += 8.00
    if 4 <= hour < 8:
        base += 1.00
    elif 16 <= hour < 20:
        base += 2.50
    tip = round(base * 0.20, 2)
    return round(base, 2), tip, round(base + tip, 2)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["info"])
def root():
    return {
        "message": "NYC Taxi Predictor API",
        "docs": "/docs",
        "predict": "/predict",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_metrics=metrics,
        available_locations=list(NYC_BOROUGHS.keys()),
    )


@app.post("/predict", response_model=TripResponse, tags=["prediction"])
def predict(request: TripRequest):
    """Predict NYC taxi trip duration and estimate fare."""
    if request.pickup_location == request.dropoff_location:
        raise HTTPException(status_code=400, detail="Pickup and dropoff must differ.")

    features_df, dist, is_airport = build_features(request)

    if model is not None:
        duration = float(model.predict(features_df)[0])
        confidence = "high"
    else:
        # Fallback: rule-based
        speed = 9 if request.hour in [7, 8, 9, 16, 17, 18, 19] else 14
        if is_airport:
            speed = 25
        duration = (dist / speed) * 60
        confidence = "low (model not loaded — run src/train.py)"

    duration = max(5.0, round(duration, 1))
    fare, tip, total = estimate_fare(dist, duration, is_airport, request.hour)

    return TripResponse(
        duration_minutes=duration,
        distance_miles=round(dist, 2),
        estimated_fare_usd=fare,
        estimated_tip_usd=tip,
        estimated_total_usd=total,
        pickup_location=request.pickup_location,
        dropoff_location=request.dropoff_location,
        confidence=confidence,
        features_used=features_df.iloc[0].to_dict(),
    )


@app.get("/locations", tags=["info"])
def get_locations():
    """Get all available pickup/dropoff locations."""
    return {
        "locations": list(NYC_BOROUGHS.keys()),
        "count": len(NYC_BOROUGHS),
    }


@app.get("/example", tags=["info"])
def example_request():
    """Returns an example API request body."""
    return {
        "example_request": {
            "pickup_location": "Manhattan",
            "dropoff_location": "JFK Airport",
            "hour": 17,
            "day_of_week": 2,
            "month": 6,
            "passenger_count": 2,
        },
        "curl": 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"pickup_location":"Manhattan","dropoff_location":"JFK Airport","hour":17,"day_of_week":2,"month":6,"passenger_count":1}\'',
    }
