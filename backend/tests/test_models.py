"""
Unit test suite verifying ConvLSTM, Iceberg Drift, and Polar Route Optimization.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from app.models.sea_ice_convlstm import sea_ice_predictor
from app.models.iceberg_drift import iceberg_drift_engine
from app.models.route_optimizer import polar_route_optimizer, calculate_polaris_rio
from app.config import INITIAL_ICEBERGS

def test_sea_ice_convlstm():
    print("Testing Sea Ice ConvLSTM Forecast & Baseline Evaluation...")
    lats = np.linspace(-75.0, -60.0, 15)
    lons = np.linspace(60.0, 90.0, 20)
    result = sea_ice_predictor.forecast(lat_grid=lats, lon_grid=lons, days_ahead=3, current_day_of_year=45)
    assert "forecast_days" in result
    assert len(result["forecast_days"]) == 3
    assert "metrics" in result["forecast_days"][0]
    m = result["forecast_days"][0]["metrics"]
    print(f"  Day 1 Model RMSE: {m['model_rmse']}, Persistence RMSE: {m['persistence_rmse']}")
    print(f"  IIEE Model: {m['model_iiee_km2']} km2 vs Persistence: {m['persistence_iiee_km2']} km2")
    print("[PASS] Sea Ice ConvLSTM test passed!")

def test_iceberg_drift():
    print("Testing Physics-informed Iceberg Drift Engine...")
    berg = INITIAL_ICEBERGS[0] # A-23a
    res = iceberg_drift_engine.predict_trajectory(berg, forecast_hours=48, time_step_hours=6)
    assert "trajectory" in res
    assert len(res["trajectory"]) > 0
    t0 = res["trajectory"][0]
    t_end = res["trajectory"][-1]
    print(f"  Iceberg {berg['name']} drifted from ({t0['lat']}, {t0['lon']}) to ({t_end['lat']}, {t_end['lon']})")
    print(f"  Uncertainty radius at 48h: {t_end['uncertainty_radius_km']} km")
    print("[PASS] Iceberg Drift test passed!")

def test_route_optimizer():
    print("Testing Polar Route Optimizer and IMO POLARIS RIO...")
    # From Cape Town to Bharati Station
    routes = polar_route_optimizer.find_pareto_routes(
        origin_lat=-33.918,
        origin_lon=18.423,
        dest_lat=-69.407,
        dest_lon=76.187,
        icebergs=INITIAL_ICEBERGS,
        vessel_ice_class="PC5"
    )
    assert "routes" in routes
    assert "balanced" in routes["routes"]
    assert "safest" in routes["routes"]
    assert "fastest" in routes["routes"]
    assert "eco_fuel" in routes["routes"]

    balanced = routes["routes"]["balanced"]
    safest = routes["routes"]["safest"]
    fastest = routes["routes"]["fastest"]
    eco = routes["routes"]["eco_fuel"]

    print(f"  Balanced: {balanced['total_distance_nm']} NM, Fuel: {balanced['total_fuel_mt']} MT, Safety Score: {balanced['overall_safety_score']}")
    print(f"  Safest:   {safest['total_distance_nm']} NM, Fuel: {safest['total_fuel_mt']} MT, Safety Score: {safest['overall_safety_score']}")
    print(f"  Fastest:  {fastest['total_distance_nm']} NM, Fuel: {fastest['total_fuel_mt']} MT, Safety Score: {fastest['overall_safety_score']}")
    print(f"  Eco-Fuel: {eco['total_distance_nm']} NM, Fuel: {eco['total_fuel_mt']} MT, Safety Score: {eco['overall_safety_score']}")
    print("[PASS] Polar Route Optimizer test passed!")

if __name__ == "__main__":
    test_sea_ice_convlstm()
    test_iceberg_drift()
    test_route_optimizer()
    print("\nALL BACKEND MODEL TESTS PASSED SUCCESSFULLY!")
