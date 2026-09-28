"""
Structured pytest test suite for POLARIS-AI Antarctic DSS.
Verifies PyTorch ConvLSTM forecasting, physics-based iceberg drift,
polar A* graph route optimization, and FastAPI operational endpoints.
"""
import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.models.sea_ice_convlstm import sea_ice_predictor
from app.models.iceberg_drift import iceberg_drift_engine
from app.models.route_optimizer import polar_route_optimizer, calculate_polaris_rio
from app.config import INITIAL_ICEBERGS, ANTARCTIC_WAYPOINTS


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def sample_spatial_grid():
    """Antarctic spatial grid fixture."""
    lats = np.linspace(-75.0, -60.0, 15)
    lons = np.linspace(60.0, 90.0, 20)
    return lats, lons


@pytest.fixture
def sample_iceberg():
    """Sample tracked iceberg fixture (A-23a)."""
    return INITIAL_ICEBERGS[0]


class TestSeaIceConvLSTM:
    """Tests for PyTorch ConvLSTM spatiotemporal forecasting and dynamic metrics."""

    def test_convlstm_forecast_shape_and_metrics(self, sample_spatial_grid):
        lats, lons = sample_spatial_grid
        days_ahead = 3
        result = sea_ice_predictor.forecast(
            lat_grid=lats,
            lon_grid=lons,
            days_ahead=days_ahead,
            current_day_of_year=45
        )

        assert "forecast_days" in result
        assert len(result["forecast_days"]) == days_ahead
        assert "lead_time_evaluations" in result
        assert len(result["lead_time_evaluations"]) == days_ahead

        # Check first lead day output structure
        day1 = result["forecast_days"][0]
        assert "model_grid" in day1
        assert "persistence_grid" in day1
        assert "ground_truth_grid" in day1
        assert "metrics" in day1

        # Check spatial dimensions match requested grid
        grid_shape = (len(lats), len(lons))
        assert np.array(day1["model_grid"]).shape == grid_shape

        # Verify dynamic evaluation metrics keys
        m = day1["metrics"]
        assert "convlstm_rmse" in m
        assert "persistence_rmse" in m
        assert "convlstm_iiee_km2" in m
        assert "persistence_iiee_km2" in m
        assert m["convlstm_rmse"] >= 0.0
        assert m["convlstm_iiee_km2"] >= 0.0

    def test_convlstm_dynamic_evaluation_metrics(self):
        metrics = sea_ice_predictor.get_evaluation_metrics()
        assert "lead_time_evaluations" in metrics
        assert "benchmark_summary" in metrics
        assert len(metrics["lead_time_evaluations"]) == 7

        summary = metrics["benchmark_summary"]
        assert "avg_model_rmse" in summary
        assert "avg_persistence_rmse" in summary
        assert "avg_iiee_reduction_pct" in summary
        assert summary["avg_model_rmse"] < summary["avg_persistence_rmse"]


class TestIcebergDrift:
    """Tests for physics-based 2D momentum equation iceberg drift."""

    def test_iceberg_drift_physics(self, sample_iceberg):
        forecast_hours = 48
        res = iceberg_drift_engine.predict_trajectory(
            sample_iceberg,
            forecast_hours=forecast_hours,
            time_step_hours=6
        )

        assert "trajectory" in res
        traj = res["trajectory"]
        assert len(traj) == (forecast_hours // 6) + 1

        t0 = traj[0]
        t_end = traj[-1]

        # Initial point should match the iceberg's current position
        assert pytest.approx(t0["lat"], abs=1e-3) == sample_iceberg["lat"]
        assert pytest.approx(t0["lon"], abs=1e-3) == sample_iceberg["lon"]

        # Uncertainty radius must grow monotonically over forecast horizon
        radii = [step["uncertainty_radius_km"] for step in traj]
        assert radii[0] >= 0.0
        assert all(radii[i] <= radii[i + 1] for i in range(len(radii) - 1))
        assert t_end["uncertainty_radius_km"] > radii[0]

    @pytest.mark.parametrize("time_step", [3, 6, 12])
    def test_iceberg_drift_time_steps(self, sample_iceberg, time_step):
        res = iceberg_drift_engine.predict_trajectory(
            sample_iceberg,
            forecast_hours=24,
            time_step_hours=time_step
        )
        assert len(res["trajectory"]) == (24 // time_step) + 1


class TestPolarRouteOptimizer:
    """Tests for A* polar ocean risk mesh route optimization and IMO POLARIS RIO."""

    def test_a_star_polar_route_optimizer(self):
        # Route from Cape Town to Bharati Station
        routes = polar_route_optimizer.find_pareto_routes(
            origin_lat=-33.918,
            origin_lon=18.423,
            dest_lat=-69.407,
            dest_lon=76.187,
            icebergs=INITIAL_ICEBERGS,
            vessel_ice_class="PC5"
        )

        assert "routes" in routes
        for mode in ["balanced", "safest", "fastest", "eco_fuel"]:
            assert mode in routes["routes"]
            route = routes["routes"][mode]
            assert "waypoints" in route
            assert len(route["waypoints"]) >= 2
            assert route["total_distance_nm"] > 0.0
            assert route["total_fuel_mt"] > 0.0
            assert 0.0 <= route["overall_safety_score"] <= 100.0

    @pytest.mark.parametrize("ice_class,expected_status", [
        ("PC1", "NORMAL_OPERATION"),
        ("PC5", "ESCORT_REQUIRED"),
        ("OPEN_WATER", "PROHIBITED")
    ])
    def test_polaris_rio_calculation(self, ice_class, expected_status):
        # Multi-year thick ice scenario (SIC 0.85)
        result = calculate_polaris_rio(ice_class=ice_class, ice_concentration=0.85)
        assert result["status"] == expected_status


class TestApiEndpoints:
    """Integration tests for FastAPI endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_stations_endpoint(self, client):
        response = client.get("/api/navigation/stations")
        assert response.status_code == 200
        stations = response.json()
        assert len(stations) >= 4
        # Verify Indian stations present
        assert "BHARATI_STATION" in stations
        assert "MAITRI_STATION" in stations

    def test_polar_classes_endpoint(self, client):
        response = client.get("/api/navigation/polar-classes")
        assert response.status_code == 200
        classes = response.json()
        assert "PC1" in classes
        assert "PC5" in classes

    def test_icebergs_endpoint(self, client):
        response = client.get("/api/icebergs")
        assert response.status_code == 200
        bergs = response.json()
        assert len(bergs) >= 1

    def test_forecast_metrics_endpoint(self, client):
        response = client.get("/api/forecast/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "lead_time_evaluations" in data
        assert "benchmark_summary" in data

    def test_route_optimization_endpoint(self, client):
        payload = {
            "origin_key": "PORT_CAPE_TOWN",
            "dest_key": "BHARATI_STATION",
            "vessel_ice_class": "PC5",
            "cruising_speed_knots": 13.5
        }
        response = client.post("/api/navigation/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "routes" in data
        assert "balanced" in data["routes"]
