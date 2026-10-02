"""
FastAPI Endpoints for Antarctic Iceberg Surveillance and Drift Prediction.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from ..services.iceberg_service import iceberg_service

router = APIRouter(prefix="/icebergs", tags=["Iceberg Trajectory"])

class NewIcebergPayload(BaseModel):
    id: Optional[str] = None
    name: str
    calving_source: str
    lat: float
    lon: float
    area_km2: float
    length_km: float
    width_km: float
    thickness_m: float = 250.0
    drift_speed_knots: float = 1.0
    drift_bearing_deg: float = 45.0
    surveillance_source: str = "Sentinel-1 SAR Real-time Ingestion"

@router.get("")
def list_icebergs(refresh_live: bool = Query(False, description="Trigger on-demand live satellite feed re-sync from BYU/ASCAT")):
    """Returns active tracked Antarctic icebergs (BYU / US National Ice Center registry)."""
    if refresh_live:
        iceberg_service.sync_live_byu_feed(timeout_s=4.0)
    return iceberg_service.list_icebergs()

@router.post("/sync-live")
def sync_live_satellite_icebergs():
    """Fetches near-real-time satellite scatterometer fixes directly from BYU/ASCAT live feed."""
    return iceberg_service.sync_live_byu_feed(timeout_s=5.0)

@router.get("/trajectories")
def get_all_trajectories(
    hours: int = Query(120, ge=12, le=240, description="Trajectory forecast horizon in hours")
):
    """Computes physics+ML drift trajectories for all tracked icebergs."""
    return iceberg_service.forecast_all_icebergs(forecast_hours=hours)

@router.get("/{iceberg_id}/trajectory")
def get_single_iceberg_trajectory(
    iceberg_id: str,
    hours: int = Query(120, ge=12, le=240)
):
    """Computes physics+ML drift trajectory and uncertainty cone for a specific iceberg."""
    try:
        return iceberg_service.forecast_iceberg(iceberg_id, forecast_hours=hours)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/register")
def register_iceberg(payload: NewIcebergPayload):
    """Registers a newly detected iceberg or fragment (e.g. from automated SAR detection)."""
    created = iceberg_service.register_iceberg(payload.dict())
    return {"status": "success", "iceberg": created}
