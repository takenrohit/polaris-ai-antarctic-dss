"""
FastAPI Endpoints for Antarctic Route Optimization and POLARIS Decision Support.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from ..config import ANTARCTIC_WAYPOINTS, IMO_POLAR_CLASSES
from ..models.route_optimizer import polar_route_optimizer
from ..services.iceberg_service import iceberg_service
from ..services.vessel_service import vessel_service

router = APIRouter(prefix="/navigation", tags=["Navigation & Route Optimization"])

class RouteOptimizationRequest(BaseModel):
    origin_key: Optional[str] = "PORT_CAPE_TOWN"
    dest_key: Optional[str] = "BHARATI_STATION"
    custom_origin_lat: Optional[float] = None
    custom_origin_lon: Optional[float] = None
    custom_dest_lat: Optional[float] = None
    custom_dest_lon: Optional[float] = None
    vessel_ice_class: str = Field(default="PC5", description="IMO Polar Class: PC1, PC3, PC5, PC7, or OPEN_WATER")
    cruising_speed_knots: float = Field(default=13.5, ge=6.0, le=25.0)

@router.get("/stations")
def get_polar_waypoints():
    """Returns Antarctic research stations (Bharati, Maitri) and maritime gateway ports."""
    return ANTARCTIC_WAYPOINTS

@router.get("/polar-classes")
def get_imo_polar_classes():
    """Returns IMO Polar Code vessel classes and operational limits."""
    return IMO_POLAR_CLASSES

@router.get("/vessels")
def get_ncpor_vessels():
    """Returns active expedition vessels with live Antarctic positions."""
    return vessel_service.list_vessels()

@router.post("/optimize")
def optimize_polar_route(payload: RouteOptimizationRequest):
    """
    Computes Pareto-optimal polar navigation corridors:
    - Balanced (NCPOR Recommended)
    - Safest (Iceberg standoff & low sea ice)
    - Fastest (Direct icebreaker transit)
    - Eco-Fuel (Minimizes fuel burn & ice resistance)
    """
    # Resolve coordinates
    if payload.custom_origin_lat is not None and payload.custom_origin_lon is not None:
        o_lat, o_lon = payload.custom_origin_lat, payload.custom_origin_lon
    elif payload.origin_key and payload.origin_key in ANTARCTIC_WAYPOINTS:
        o_lat = ANTARCTIC_WAYPOINTS[payload.origin_key]["lat"]
        o_lon = ANTARCTIC_WAYPOINTS[payload.origin_key]["lon"]
    else:
        o_lat, o_lon = -33.918, 18.423 # Default Cape Town

    if payload.custom_dest_lat is not None and payload.custom_dest_lon is not None:
        d_lat, d_lon = payload.custom_dest_lat, payload.custom_dest_lon
    elif payload.dest_key and payload.dest_key in ANTARCTIC_WAYPOINTS:
        d_lat = ANTARCTIC_WAYPOINTS[payload.dest_key]["lat"]
        d_lon = ANTARCTIC_WAYPOINTS[payload.dest_key]["lon"]
    else:
        d_lat, d_lon = -69.407, 76.187 # Default Bharati Station

    icebergs = iceberg_service.list_icebergs()

    result = polar_route_optimizer.find_pareto_routes(
        origin_lat=o_lat,
        origin_lon=o_lon,
        dest_lat=d_lat,
        dest_lon=d_lon,
        icebergs=icebergs,
        vessel_ice_class=payload.vessel_ice_class,
        cruising_speed_knots=payload.cruising_speed_knots
    )

    # Attach human-readable station names if matched
    origin_name = ANTARCTIC_WAYPOINTS.get(payload.origin_key, {}).get("name", f"Custom ({o_lat:.2f}, {o_lon:.2f})")
    dest_name = ANTARCTIC_WAYPOINTS.get(payload.dest_key, {}).get("name", f"Custom ({d_lat:.2f}, {d_lon:.2f})")
    result["origin_name"] = origin_name
    result["dest_name"] = dest_name

    return result

@router.post("/export-geojson")
def export_route_geojson(payload: RouteOptimizationRequest):
    """Generates standard GeoJSON for OpenCPN, QGIS, or ECDIS bridge displays."""
    route_data = optimize_polar_route(payload)
    features = []

    for mode_key, route in route_data["routes"].items():
        coords = [[pt["lon"], pt["lat"]] for pt in route["waypoints"]]
        features.append({
            "type": "Feature",
            "properties": {
                "mode": mode_key,
                "name": route["mode_name"],
                "total_distance_nm": route["total_distance_nm"],
                "total_fuel_mt": route["total_fuel_mt"],
                "polaris_compliance": route["polaris_compliance"],
                "safety_score": route["overall_safety_score"]
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })

    return {
        "type": "FeatureCollection",
        "name": f"NCPOR_Route_{route_data['origin_name']}_to_{route_data['dest_name']}",
        "features": features
    }
