"""
Tests for NYC Taxi Predictor
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Test Feature Engineering ────────────────────────────────────────────────
class TestFeatureEngineering:

    def test_haversine_distance_known_route(self):
        """Manhattan to JFK is roughly 15-17 miles."""
        from app import haversine_distance
        dist = haversine_distance(40.7831, -73.9712, 40.6413, -73.7781)
        assert 14 <= dist <= 18, f"Expected 14-18 miles, got {dist:.2f}"

    def test_haversine_distance_same_point(self):
        """Zero distance for same coordinates."""
        from app import haversine_distance
        dist = haversine_distance(40.7831, -73.9712, 40.7831, -73.9712)
        assert dist < 0.01

    def test_build_features_shape(self):
        """Feature builder returns correct shape."""
        from app import build_features, NYC_BOROUGHS
        df = build_features("Manhattan", "Brooklyn", 8, 1, 2, 6)
        assert df.shape == (1, 16), f"Expected (1, 16), got {df.shape}"

    def test_build_features_rush_hour_flag(self):
        """Rush hour flag correctly set for morning rush."""
        from app import build_features
        df = build_features("Manhattan", "Brooklyn", 8, 1, 2, 6)
        assert df["is_rush_hour"].values[0] == 1

    def test_build_features_not_rush_hour(self):
        """Rush hour flag correctly cleared at midnight."""
        from app import build_features
        df = build_features("Manhattan", "Brooklyn", 0, 1, 2, 6)
        assert df["is_rush_hour"].values[0] == 0

    def test_build_features_weekend_flag(self):
        """Weekend flag set correctly for Saturday (day_of_week=5)."""
        from app import build_features
        df = build_features("Manhattan", "Brooklyn", 12, 5, 2, 6)  # Saturday
        assert df["is_weekend"].values[0] == 1

    def test_build_features_airport_trip(self):
        """Airport trip flag set for JFK pickup."""
        from app import build_features
        df = build_features("JFK Airport", "Manhattan", 14, 2, 2, 6)
        assert df["is_airport_trip"].values[0] == 1

    def test_build_features_cyclical_hour_encoding(self):
        """Hour 0 and hour 24 should map to same cyclical values."""
        from app import build_features
        df_midnight = build_features("Manhattan", "Brooklyn", 0, 1, 2, 6)
        assert abs(df_midnight["hour_sin"].values[0]) < 0.01
        assert abs(df_midnight["hour_cos"].values[0] - 1.0) < 0.01

    def test_build_features_distance_positive(self):
        """Distance must always be positive."""
        from app import build_features
        df = build_features("Bronx", "Staten Island", 10, 3, 4, 7)
        assert df["distance_miles"].values[0] > 0


# ─── Test Fare Estimation ────────────────────────────────────────────────────
class TestFareEstimation:

    def test_fare_minimum_applies(self):
        """Fare should meet NYC minimum."""
        from app import estimate_fare
        fare, tip, total = estimate_fare(0.5, 5, False, 12)
        assert fare >= 5.00, "Fare below expected minimum"

    def test_tip_is_20_percent(self):
        """Tip estimate should be approximately 20% of fare."""
        from app import estimate_fare
        fare, tip, total = estimate_fare(3.0, 15, False, 12)
        assert abs(tip / fare - 0.20) < 0.01

    def test_total_equals_fare_plus_tip(self):
        """Total should equal fare + tip."""
        from app import estimate_fare
        fare, tip, total = estimate_fare(3.0, 15, False, 12)
        assert abs(total - (fare + tip)) < 0.01

    def test_rush_hour_surcharge(self):
        """4pm-8pm should have higher fare than midnight."""
        from app import estimate_fare
        fare_rush, _, _ = estimate_fare(3.0, 20, False, 17)
        fare_off, _, _ = estimate_fare(3.0, 20, False, 2)
        assert fare_rush > fare_off, "Rush hour should cost more"

    def test_airport_surcharge(self):
        """Airport trips should cost more."""
        from app import estimate_fare
        fare_airport, _, _ = estimate_fare(5.0, 30, True, 12)
        fare_regular, _, _ = estimate_fare(5.0, 30, False, 12)
        assert fare_airport > fare_regular


# ─── Test Prediction Fallback ────────────────────────────────────────────────
class TestPredictionFallback:

    def test_fallback_duration_reasonable(self):
        """Fallback prediction should return realistic duration."""
        from app import build_features, predict_duration_fallback
        df = build_features("Manhattan", "Brooklyn", 10, 2, 2, 6)
        duration = predict_duration_fallback(df, "Midday (10am–3pm)")
        assert 5 <= duration <= 120, f"Duration {duration} out of realistic range"

    def test_minimum_duration_enforced(self):
        """Very short trips shouldn't go below 5 minutes."""
        from app import build_features, predict_duration_fallback
        df = build_features("Manhattan", "Manhattan", 10, 2, 2, 6)
        # Override distance to be very small for test
        df["distance_miles"] = 0.1
        duration = predict_duration_fallback(df, "Early Morning (12am–6am)")
        assert duration >= 5


# ─── Integration Test ─────────────────────────────────────────────────────────
class TestIntegration:

    def test_full_pipeline_output_types(self):
        """Full pipeline returns correct types."""
        from app import build_features, predict_duration_fallback, estimate_fare
        df = build_features("Manhattan", "JFK Airport", 17, 3, 6, 2)
        dist = df["distance_miles"].values[0]
        duration = predict_duration_fallback(df, "Afternoon Rush (4pm–7pm)")
        fare, tip, total = estimate_fare(dist, duration, True, 17)

        assert isinstance(duration, float)
        assert isinstance(fare, float)
        assert isinstance(tip, float)
        assert isinstance(total, float)

    def test_all_boroughs_work(self):
        """All borough combinations should not raise errors."""
        from app import build_features, NYC_BOROUGHS
        boroughs = list(NYC_BOROUGHS.keys())
        for pu in boroughs[:3]:
            for do in boroughs[:3]:
                if pu != do:
                    df = build_features(pu, do, 10, 2, 6, 2)
                    assert len(df) == 1
