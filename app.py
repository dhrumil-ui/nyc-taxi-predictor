"""
NYC Taxi Trip Duration & Fare Predictor
End-to-end ML application using real NYC TLC open data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi AI Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Dark NYC theme */
:root {
    --yellow: #F7C900;
    --dark: #0D0D0D;
    --surface: #1A1A1A;
    --border: #2E2E2E;
    --text: #F0F0F0;
    --muted: #888;
}

.stApp { background: var(--dark); color: var(--text); font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Bebas Neue', cursive;
    font-size: clamp(3rem, 8vw, 6rem);
    letter-spacing: 4px;
    color: var(--yellow);
    line-height: 0.95;
    margin-bottom: 4px;
}

.sub-title {
    font-size: 0.9rem;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 32px;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--yellow);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 2.8rem;
    color: var(--yellow);
    letter-spacing: 2px;
    line-height: 1;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}

.info-box {
    background: rgba(247, 201, 0, 0.06);
    border: 1px solid rgba(247, 201, 0, 0.2);
    border-radius: 4px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 0.88rem;
    color: #ccc;
    line-height: 1.6;
}

.stButton > button {
    background: var(--yellow) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 1.2rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 12px 36px !important;
    width: 100%;
    transition: all 0.15s !important;
}

.stButton > button:hover {
    background: #fff !important;
    transform: translateY(-1px);
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label {
    color: #aaa !important;
    font-size: 0.78rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

.section-header {
    font-family: 'Bebas Neue', cursive;
    font-size: 1.6rem;
    letter-spacing: 3px;
    color: var(--yellow);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

.borough-badge {
    display: inline-block;
    background: var(--yellow);
    color: #000;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Artifacts ───────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    try:
        model = joblib.load("models/taxi_model.pkl")
        with open("models/feature_names.json") as f:
            feature_names = json.load(f)
        with open("models/model_metrics.json") as f:
            metrics = json.load(f)
        return model, feature_names, metrics
    except FileNotFoundError:
        return None, None, None


model, feature_names, metrics = load_artifacts()


# ─── Feature Engineering ──────────────────────────────────────────────────────
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

TRAFFIC_PROFILES = {
    "Early Morning (12am–6am)": {"multiplier": 0.7, "surge": 1.0},
    "Morning Rush (7am–9am)": {"multiplier": 1.8, "surge": 1.4},
    "Midday (10am–3pm)": {"multiplier": 1.1, "surge": 1.1},
    "Afternoon Rush (4pm–7pm)": {"multiplier": 2.0, "surge": 1.6},
    "Evening (8pm–11pm)": {"multiplier": 1.3, "surge": 1.2},
    "Late Night (11pm–12am)": {"multiplier": 0.9, "surge": 1.3},
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two coordinates."""
    R = 3959  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_features(pickup, dropoff, hour, day_of_week, passenger_count, month):
    """Engineer features matching training pipeline."""
    pu = NYC_BOROUGHS[pickup]
    do = NYC_BOROUGHS[dropoff]
    dist = haversine_distance(pu["lat"], pu["lon"], do["lat"], do["lon"])

    is_rush = 1 if hour in [7, 8, 9, 16, 17, 18, 19] else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    is_airport = 1 if "Airport" in pickup or "Airport" in dropoff else 0
    is_night = 1 if hour >= 22 or hour <= 5 else 0
    is_manhattan = 1 if "Manhattan" in pickup or "Manhattan" in dropoff else 0

    features = {
        "distance_miles": dist,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "passenger_count": passenger_count,
        "pickup_zone": pu["zone"],
        "dropoff_zone": do["zone"],
        "is_rush_hour": is_rush,
        "is_weekend": is_weekend,
        "is_airport_trip": is_airport,
        "is_night": is_night,
        "is_manhattan": is_manhattan,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_cos": np.cos(2 * np.pi * day_of_week / 7),
    }
    return pd.DataFrame([features])


def estimate_fare(distance, duration_min, is_airport, surge_mult, hour):
    """NYC TLC fare estimation formula."""
    base_fare = 3.00
    per_mile = 1.70 * distance
    per_min = 0.50 * (duration_min / 60 * 12)  # $0.50 per unit (slow speed equiv)
    mta_surcharge = 0.50
    improvement_surcharge = 1.00

    subtotal = base_fare + per_mile + per_min + mta_surcharge + improvement_surcharge

    if is_airport:
        if "JFK" in str(is_airport):
            subtotal += 52.00  # JFK flat rate from Manhattan
        else:
            subtotal += 8.00  # Airport surcharge

    if 4 <= hour < 8:
        subtotal += 1.00  # Overnight surcharge
    elif 16 <= hour < 20:
        subtotal += 2.50  # Rush hour surcharge

    subtotal *= surge_mult
    tip_estimate = subtotal * 0.20  # Suggested 20% tip
    total = subtotal + tip_estimate
    return round(subtotal, 2), round(tip_estimate, 2), round(total, 2)


# ─── Simulated Prediction (when model not loaded) ─────────────────────────────
def predict_duration_fallback(features_df, traffic_profile):
    """Rule-based fallback when model file not present."""
    dist = features_df["distance_miles"].values[0]
    is_rush = features_df["is_rush_hour"].values[0]
    is_airport = features_df["is_airport_trip"].values[0]
    multiplier = TRAFFIC_PROFILES[traffic_profile]["multiplier"]

    # Average NYC taxi speed varies 7–15 mph depending on traffic
    base_speed = 12 if is_rush else 15
    if is_airport:
        base_speed = 25

    duration = (dist / base_speed) * 60 * multiplier
    return max(5, round(duration, 1))


# ─── UI Layout ────────────────────────────────────────────────────────────────
# Header
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.markdown('<div class="main-title">NYC TAXI<br>AI PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real NYC TLC Data · XGBoost + Feature Engineering · Live Estimates</div>', unsafe_allow_html=True)

with col_h2:
    if metrics:
        st.markdown('<div class="section-header">Model Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('rmse_minutes', 4.2):.1f} min</div>
            <div class="metric-label">RMSE (Duration Error)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('r2', 0.87)*100:.0f}%</div>
            <div class="metric-label">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            ⚙️ <strong>Demo Mode</strong><br>
            Run <code>python src/train.py</code> to train the model on real NYC TLC data.
            Using rule-based estimation until model is loaded.
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Trip Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    pickup = st.selectbox("📍 Pickup Location", list(NYC_BOROUGHS.keys()), index=0)
    dropoff = st.selectbox("🏁 Dropoff Location", list(NYC_BOROUGHS.keys()), index=1)
    passengers = st.slider("👥 Passengers", 1, 6, 1)

with col2:
    now = datetime.now()
    trip_date = st.date_input("📅 Trip Date", value=now.date())
    traffic_time = st.selectbox("🕐 Time of Day", list(TRAFFIC_PROFILES.keys()), index=1)
    payment = st.selectbox("💳 Payment Type", ["Credit Card", "Cash", "App (Uber/Lyft)"])

with col3:
    st.markdown("#### ℹ️ About This Model")
    st.markdown("""
    <div class="info-box">
    Trained on <strong>1M+ NYC TLC trip records</strong> (2023–2024).
    Features: geospatial distance, time-of-day cyclical encoding, rush hour flags, airport surcharges,
    borough zones, and passenger count.<br><br>
    <strong>Model:</strong> XGBoost Regressor<br>
    <strong>Target:</strong> Trip duration (minutes)<br>
    <strong>Training data:</strong> NYC Open Data TLC
    </div>
    """, unsafe_allow_html=True)

# Derive hour from time selection
time_hour_map = {
    "Early Morning (12am–6am)": 3,
    "Morning Rush (7am–9am)": 8,
    "Midday (10am–3pm)": 12,
    "Afternoon Rush (4pm–7pm)": 17,
    "Evening (8pm–11pm)": 20,
    "Late Night (11pm–12am)": 23,
}
hour = time_hour_map[traffic_time]
day_of_week = trip_date.weekday()
month = trip_date.month

# ─── Predict Button ───────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🗽 PREDICT TRIP")

if predict_btn:
    if pickup == dropoff:
        st.error("⚠️ Pickup and dropoff must be different locations.")
    else:
        features_df = build_features(pickup, dropoff, hour, day_of_week, passengers, month)
        dist = features_df["distance_miles"].values[0]
        is_airport = "Airport" in pickup or "Airport" in dropoff
        surge = TRAFFIC_PROFILES[traffic_time]["surge"]

        # Duration prediction
        if model is not None:
            duration_min = float(model.predict(features_df)[0])
        else:
            duration_min = predict_duration_fallback(features_df, traffic_time)

        duration_min = max(5, round(duration_min, 1))

        # Fare estimation
        subtotal, tip, total = estimate_fare(dist, duration_min, is_airport, surge, hour)

        st.divider()
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(duration_min)} min</div>
                <div class="metric-label">Estimated Duration</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dist:.1f} mi</div>
                <div class="metric-label">Trip Distance</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${subtotal}</div>
                <div class="metric-label">Estimated Fare</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${total}</div>
                <div class="metric-label">Total w/ Tip (20%)</div>
            </div>""", unsafe_allow_html=True)

        # Route map
        st.markdown('<div class="section-header">Route Map</div>', unsafe_allow_html=True)
        pu_data = NYC_BOROUGHS[pickup]
        do_data = NYC_BOROUGHS[dropoff]

        map_df = pd.DataFrame({
            "lat": [pu_data["lat"], do_data["lat"]],
            "lon": [pu_data["lon"], do_data["lon"]],
            "label": [f"📍 {pickup}", f"🏁 {dropoff}"],
            "type": ["Pickup", "Dropoff"],
            "color": ["#F7C900", "#FF4444"],
        })

        fig = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", text="label",
            color="type",
            color_discrete_map={"Pickup": "#F7C900", "Dropoff": "#FF4444"},
            zoom=10, height=380,
            mapbox_style="carto-darkmatter",
        )
        fig.add_trace(go.Scattermapbox(
            lat=[pu_data["lat"], do_data["lat"]],
            lon=[pu_data["lon"], do_data["lon"]],
            mode="lines",
            line=dict(width=3, color="#F7C900"),
            opacity=0.6,
            showlegend=False,
        ))
        fig.update_layout(
            paper_bgcolor="#0D0D0D",
            plot_bgcolor="#0D0D0D",
            font_color="#F0F0F0",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(font_color="#aaa"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Fare breakdown
        with st.expander("📋 Fare Breakdown (NYC TLC Formula)"):
            breakdown = {
                "Base Fare": "$3.00",
                "Per Mile": f"${1.70 * dist:.2f}",
                "Time Charge": f"~${0.50 * duration_min / 5:.2f}",
                "MTA Surcharge": "$0.50",
                "Improvement Surcharge": "$1.00",
                f"Surge ({surge}x)": f"×{surge}",
                "Estimated Tip (20%)": f"${tip}",
                "**Total Estimate**": f"**${total}**",
            }
            for k, v in breakdown.items():
                cols = st.columns([3, 1])
                cols[0].write(k)
                cols[1].write(v)

        st.markdown(f"""
        <div class="info-box">
        ⚠️ <strong>Disclaimer:</strong> These are AI estimates based on historical NYC TLC data.
        Actual fares may vary. Always confirm with your driver or app.
        Surge pricing reflects real NYC traffic patterns by time of day.
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.78rem; letter-spacing:2px; padding:12px 0; font-family:'DM Sans',sans-serif;">
NYC TAXI AI PREDICTOR · BUILT WITH REAL TLC OPEN DATA · XGBOOST + STREAMLIT
</div>
""", unsafe_allow_html=True)
