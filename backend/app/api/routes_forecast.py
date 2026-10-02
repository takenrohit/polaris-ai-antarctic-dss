"""
FastAPI Endpoints for Sea-Ice Forecasting and ML vs Persistence Benchmarks.
"""
import os
import secrets
import threading
from fastapi import APIRouter, Header, HTTPException, Query
from typing import Optional
import numpy as np
from ..models.sea_ice_convlstm import sea_ice_predictor
from ..data.ingestion import environmental_data_provider, LIVE_NC_PATH, LIVE_CACHE_DIR
from ..core.validators import assess_data_quality
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
    if environmental_data_provider.mode == "live":
        # Operational forecast from the newest observed day (no ground truth, no skill metrics)
        result = sea_ice_predictor.forecast_live(
            lat_grid=DEFAULT_LATS, lon_grid=DEFAULT_LONS, days_ahead=days_ahead
        )
    else:
        # Frozen snapshot: fixed-window hindcast scored against held-out observed days
        result = sea_ice_predictor.forecast(
            lat_grid=DEFAULT_LATS,
            lon_grid=DEFAULT_LONS,
            days_ahead=days_ahead,
            current_day_of_year=day_of_year
        )
        result["forecast_mode"] = "hindcast_snapshot"
    result["data_source"] = environmental_data_provider.metadata()
    return result

@router.get("/metrics")
def get_model_benchmarks():
    """
    Returns verified validation benchmark statistics comparing ConvLSTM vs Persistence
    over historical Antarctic seasons (NSIDC / AMSR2 / ERA5 validation test set).
    Dynamically computed from the PyTorch model evaluation pipeline.
    """
    return sea_ice_predictor.get_evaluation_metrics()

@router.post("/ingest/sentinel1")
def ingest_sentinel1_sar(granule_id: Optional[str] = "S1A_EW_GRDM_1SDH_20260121T063000"):
    """
    Ingestion stub for Copernicus Sentinel-1 Extra-Wide (EW) Swath SAR Level-1 GRD imagery.
    Produces 40m resolution ice-edge boundaries and detected tabular iceberg targets.
    """
    from ..data.ingestion import sentinel1_stub
    return sentinel1_stub.ingest_granule(granule_id or "S1A_EW_GRDM_1SDH_20260121T063000")

@router.post("/ingest/amsr2")
def ingest_amsr2_radiometer(date_str: Optional[str] = "2026-01-21", sector: Optional[str] = "Prydz_Bay"):
    """
    Ingestion stub for JAXA GCOM-W1 AMSR2 6.25km daily Sea Ice Concentration products.
    """
    from ..data.ingestion import amsr2_stub
    return amsr2_stub.ingest_daily_product(date_str or "2026-01-21", sector or "Prydz_Bay")

@router.get("/ingestion-status")
def get_ingestion_sources_status():
    """
    Real provenance and freshness of the active data store, tied to the fail-safe gate.
    Sentinel-1 / AMSR2 are reported as not ingested (stubs only); no latency is invented.
    """
    meta = environmental_data_provider.metadata()
    quality = assess_data_quality(observation_iso=meta["latest_observation_iso"])
    return {
        "status": quality["quality_status"],
        "fail_safe_gate_tripped": quality["fail_safe_gate_tripped"],
        "data_freshness_indicator": quality["data_freshness_indicator"],
        "alerts": quality["alerts"],
        "data_mode": meta["mode"],
        "primary_feed": {
            "name": "NOAA@NSIDC G02135 Sea Ice Index v4.0 daily concentration",
            "resolution": "25 km polar stereographic (EPSG:3412)",
            "latest_observation_iso": meta["latest_observation_iso"],
            "latency_hours": meta["data_age_hours"],
            "observed_days": meta["observed_days"],
            "gap_filled_dates": meta["gap_filled_dates"],
            "fetched_at_utc": meta["fetched_at_utc"],
            "status": "ACTIVE" if meta["is_live"] else "FROZEN_SNAPSHOT",
        },
        "sentinel1_sar": {"name": "Copernicus Sentinel-1 EW GRD", "status": "NOT_INGESTED_STUB"},
        "amsr2_microwave": {"name": "JAXA GCOM-W1 AMSR2 Level-3", "status": "NOT_INGESTED_STUB"},
        "atmospheric_forcing": {
            "name": meta["wind_source"],
            "forecast_days_available": meta["forecast_wind_days"],
            "status": "ACTIVE" if meta["is_live"] else "FROZEN_SNAPSHOT",
        },
        "ocean_currents": {
            "name": meta["currents_source"],
            "status": "SYNTHETIC_PROXY",
        },
        "sst": {"name": meta["sst_source"], "status": "PROXY"},
    }


_refresh_lock = threading.Lock()


@router.post("/refresh")
def refresh_live_data(x_refresh_token: Optional[str] = Header(default=None)):
    """
    Fetch the latest NSIDC sea ice + Open-Meteo winds, rebuild the live store, and reload it.

    Disabled unless the POLARIS_REFRESH_TOKEN environment variable is set; callers must send
    the same value in the X-Refresh-Token header. On failure the previous store stays active.
    """
    expected = os.environ.get("POLARIS_REFRESH_TOKEN")
    if not expected:
        raise HTTPException(status_code=403, detail="Live refresh is disabled (POLARIS_REFRESH_TOKEN not set).")
    if not x_refresh_token or not secrets.compare_digest(x_refresh_token, expected):
        raise HTTPException(status_code=401, detail="Invalid refresh token.")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A refresh is already running.")
    try:
        from ..data.live_fetch import build_live_store, LiveFetchError
        try:
            summary = build_live_store(LIVE_NC_PATH, LIVE_CACHE_DIR)
        except LiveFetchError as exc:
            raise HTTPException(status_code=502, detail=f"Live fetch failed; previous store kept: {exc}")
        environmental_data_provider.reload(str(LIVE_NC_PATH))
        sea_ice_predictor.invalidate_cache()
        return {"refreshed": True, "summary": summary, "data_source": environmental_data_provider.metadata()}
    finally:
        _refresh_lock.release()
