"""
Comprehensive test suite for POLARIS-AI Antarctic DSS.
Verifies NetCDF ingestion, PyTorch ConvLSTM forecasting with trained weights,
physics-based iceberg drift with 4D dynamic envelopes, Lindqvist (1989) ice resistance,
IMO MSC.1/Circ.1519 POLARIS RIO evaluation, distinct Pareto route corridors, and FastAPI endpoints.
"""
import sys
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.data.ingestion import environmental_data_provider, NetCDFDatasetReader, AntarcticCoastlineMask
from app.models.sea_ice_convlstm import sea_ice_predictor
from app.models.iceberg_drift import iceberg_drift_engine
from app.models.ice_resistance import lindqvist_fuel_model
from app.models.polaris_imo import evaluate_imo_polaris_rio
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


class TestDataLayer:
    """Tests for NetCDF Metocean store and Antarctic coastline masks."""

    def test_netcdf_store_integrity(self):
        vars_available = environmental_data_provider._reader.get_variables()
        for v in ["sic", "u10", "v10", "u_curr", "v_curr", "sst"]:
            assert v in vars_available

        # Test point sample at Prydz Bay / Bharati Station approach
        sic = environmental_data_provider.get_sic(-68.5, 75.0, day_idx=0)
        assert 0.0 <= sic <= 1.0

        u_w, v_w = environmental_data_provider.get_wind(-68.5, 75.0, hour_offset=12)
        assert isinstance(u_w, float) and isinstance(v_w, float)

    def test_antarctic_coastline_mask(self):
        mask = AntarcticCoastlineMask()
        # Interior of Antarctica (South Pole) must be land
        assert mask.is_land_or_shelf(-85.0, 0.0) is True
        # Open ocean north of -58°S must NOT be land
        assert mask.is_land_or_shelf(-55.0, 20.0) is False


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

        day1 = result["forecast_days"][0]
        assert "model_grid" in day1
        assert "persistence_grid" in day1
        assert "ground_truth_grid" in day1
        assert "metrics" in day1

        grid_shape = (len(lats), len(lons))
        assert np.array(day1["model_grid"]).shape == grid_shape

        m = day1["metrics"]
        assert "convlstm_rmse" in m
        assert "persistence_rmse" in m
        assert m["convlstm_rmse"] >= 0.0

    def test_convlstm_dynamic_evaluation_metrics(self):
        metrics = sea_ice_predictor.get_evaluation_metrics()
        assert "lead_time_evaluations" in metrics
        assert "benchmark_summary" in metrics
        assert len(metrics["lead_time_evaluations"]) == 7

        summary = metrics["benchmark_summary"]
        assert "avg_model_rmse" in summary
        assert "avg_persistence_rmse" in summary
        assert "avg_iiee_reduction_pct" in summary
        assert summary["avg_model_rmse"] >= 0.0


class TestPhysicsModels:
    """Tests for Lindqvist ice resistance and IMO POLARIS RIO."""

    def test_lindqvist_ice_resistance(self):
        res = lindqvist_fuel_model.calculate_resistance_kn(
            length_m=163.0,
            beam_m=22.4,
            draft_m=9.0,
            speed_knots=12.0,
            ice_thickness_m=0.8,
            ice_concentration=0.6
        )
        assert res["r_total_kn"] > res["r_open_water_kn"]
        assert res["r_ice_total_kn"] > 0.0

        # Fuel burn calculation
        fuel_data = lindqvist_fuel_model.estimate_fuel_burn_mt(
            distance_nm=100.0,
            speed_knots=12.0,
            ice_concentration=0.6,
            ice_thickness_m=0.8
        )
        assert fuel_data["fuel_mt"] > 0.0
        assert fuel_data["transit_hours"] > 0.0

    @pytest.mark.parametrize("ice_class,expected_status", [
        ("PC1", "NORMAL_OPERATION"),
        ("PC5", "ESCORT_REQUIRED"),
        ("OPEN_WATER", "PROHIBITED")
    ])
    def test_imo_polaris_rio_msc1519(self, ice_class, expected_status):
        result = evaluate_imo_polaris_rio(
            ice_class=ice_class,
            ice_concentration=0.90,
            ice_regime="MULTI_YEAR_ICE"
        )
        assert result["status"] == expected_status
        assert isinstance(result["rio"], int)


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
        assert pytest.approx(t0["lat"], abs=1e-3) == sample_iceberg["lat"]
        assert pytest.approx(t0["lon"], abs=1e-3) == sample_iceberg["lon"]

        radii = [step["uncertainty_radius_km"] for step in traj]
        assert radii[0] >= 0.0
        assert all(radii[i] <= radii[i + 1] for i in range(len(radii) - 1))
        assert t_end["uncertainty_radius_km"] > radii[0]


class TestPolarRouteOptimizer:
    """Tests for dynamic multi-objective polar routing."""

    def test_pareto_routes_are_distinct(self):
        # Route from Cape Town to Bharati Station
        routes_data = polar_route_optimizer.find_pareto_routes(
            origin_lat=-33.918,
            origin_lon=18.423,
            dest_lat=-69.407,
            dest_lon=76.187,
            icebergs=INITIAL_ICEBERGS,
            vessel_ice_class="PC5"
        )

        routes = routes_data["routes"]
        for mode in ["balanced", "safest", "fastest", "eco_fuel"]:
            assert mode in routes
            route = routes[mode]
            assert "waypoints" in route
            assert len(route["waypoints"]) >= 2
            assert route["total_distance_nm"] > 0.0
            assert route["total_fuel_mt"] > 0.0

        # Multi-objective Pareto test: verify routes are NOT all identical
        dist_fastest = routes["fastest"]["total_distance_nm"]
        dist_safest = routes["safest"]["total_distance_nm"]
        fuel_eco = routes["eco_fuel"]["total_fuel_mt"]
        fuel_fastest = routes["fastest"]["total_fuel_mt"]

        # Safest should take a wider standoff corridor than fastest
        assert dist_safest > dist_fastest
        # Eco-Fuel should consume less fuel than fastest high-power icebreaking
        assert fuel_eco < fuel_fastest

    def test_decision_brief_dynamic_accuracy(self):
        # Open water vessel should trigger escort alert
        res_ow = polar_route_optimizer.find_pareto_routes(
            origin_lat=-33.918,
            origin_lon=18.423,
            dest_lat=-69.407,
            dest_lon=76.187,
            icebergs=INITIAL_ICEBERGS,
            vessel_ice_class="OPEN_WATER"
        )
        brief = res_ow["decision_brief"]["polaris_compliance"]
        assert "escort required" in brief.lower() or "warning" in brief.lower()


class TestApiEndpoints:
    """Integration tests for FastAPI endpoints."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stations_endpoint(self, client):
        response = client.get("/api/navigation/stations")
        assert response.status_code == 200
        stations = response.json()
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
        assert len(response.json()) >= 1

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
        assert "decision_brief" in data
