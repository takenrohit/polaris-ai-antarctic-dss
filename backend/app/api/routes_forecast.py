"""
FastAPI Endpoints for Sea-Ice Forecasting and ML vs Persistence Benchmarks.
"""
from fastapi import APIRouter, Query
from typing import Optional
import numpy as np
from ..models.sea_ice_convlstm import sea_ice_predictor
from ..config import settings

router = APIRouter(prefix="/forecast", tags=["Sea-Ice Forecasting"])

# Precompute default spatial grid covering Antarctic waters (-54° to -78°S, -180° to 180°E)
# Subsampled for real-time web responsiveness (49 lats x 73 lons)
DEFAULT_LATS = np.linspace(-78.0, -54.0, 49)
DEFAULT_LONS = np.linspace(-180.0, 180.0, 73)

@router.get("/sea-ice")
def get_sea_ice_forecast(
    days_ahead: int = Query(7, ge=1, le=14, description="Forecast lead time in days"),
    day_of_year: int = Query(45, ge=1, le=365, description="Day of year (e.g. 45 for Feb summer minimum, 260 for Sept max)")
):
    """
    Executes PyTorch ConvLSTM spatiotemporal model alongside Persistence baseline.
    Returns:
    - 2D grid matrix of Sea Ice Concentration (SIC) [0.0 - 1.0] for each forecast day
    - Persistence baseline predictions
    - Quantified evaluation metrics (RMSE, Integrated Ice Edge Error IIEE in km²)
    """
    return sea_ice_predictor.forecast(
        lat_grid=DEFAULT_LATS,
        lon_grid=DEFAULT_LONS,
        days_ahead=days_ahead,
        current_day_of_year=day_of_year
    )

@router.get("/metrics")
def get_model_benchmarks():
    """
    Returns verified validation benchmark statistics comparing ConvLSTM vs Persistence
    over historical Antarctic seasons (NSIDC / AMSR2 / ERA5 validation test set).
    Dynamically computed from the PyTorch model evaluation pipeline.
    """
    return sea_ice_predictor.get_evaluation_metrics()
