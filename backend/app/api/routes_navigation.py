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
    departure_time_offset_hours: float = Field(default=0.0, ge=0.0, le=168.0, description="Departure timing offset in hours (e.g. 0h, 24h, 48h)")
    scenario: Optional[str] = Field(default="STANDARD", description="Operational scenario: 'STANDARD' or 'LATE_SEASON_MIZ'")

from ..data.ingestion import environmental_data_provider

@router.get("/stations")
def get_polar_waypoints(live_weather: bool = Query(default=False, description="Enrich stations with live observed weather")):
    """Returns Antarctic research stations (Bharati, Maitri) and maritime gateway ports."""
    if not live_weather:
        return ANTARCTIC_WAYPOINTS

    enriched = {}
    for key, data in ANTARCTIC_WAYPOINTS.items():
        st_copy = dict(data)
        try:
            w = environmental_data_provider.get_live_weather(data["lat"], data["lon"], timeout_s=2.5)
            st_copy["live_weather"] = w
        except Exception:
            st_copy["live_weather"] = None
        enriched[key] = st_copy
    return enriched

@router.get("/stations/live-weather")
def get_stations_live_weather():
    """Returns all Antarctic research stations and gateway ports enriched with real-time live metocean observations."""
    return get_polar_waypoints(live_weather=True)


@router.get("/polar-classes")
def get_imo_polar_classes():
    """Returns IMO Polar Code vessel classes and operational limits."""
    return IMO_POLAR_CLASSES

@router.get("/vessels")
def get_ncpor_vessels(live_weather: bool = Query(False, description="Enrich with live weather")):
    """Returns active expedition vessels with Antarctic positions."""
    return vessel_service.list_vessels(enrich_live_weather=live_weather)

@router.post("/optimize")
def optimize_polar_route(payload: RouteOptimizationRequest):
    """
    Computes Pareto-optimal polar navigation corridors:
    - Balanced (NCPOR Recommended)
    - Safest (Iceberg standoff & low sea ice)
    - Fastest (Direct icebreaker transit)
    - Eco-Fuel (Minimizes fuel burn & ice resistance)
    """
    # Resolve origin coordinates & display name
    if payload.custom_origin_lat is not None and payload.custom_origin_lon is not None:
        o_lat, o_lon = payload.custom_origin_lat, payload.custom_origin_lon
        origin_name = f"Custom Origin ({o_lat:.3f}°, {o_lon:.3f}°)"
    elif payload.origin_key and payload.origin_key in ANTARCTIC_WAYPOINTS:
        o_lat = ANTARCTIC_WAYPOINTS[payload.origin_key]["lat"]
        o_lon = ANTARCTIC_WAYPOINTS[payload.origin_key]["lon"]
        origin_name = ANTARCTIC_WAYPOINTS[payload.origin_key]["name"]
    else:
        o_lat, o_lon = -33.918, 18.423 # Default Cape Town
        origin_name = "Port of Cape Town (South Africa Gateway)"

    # Resolve destination coordinates & display name
    if payload.custom_dest_lat is not None and payload.custom_dest_lon is not None:
        d_lat, d_lon = payload.custom_dest_lat, payload.custom_dest_lon
        dest_name = f"Custom Destination ({d_lat:.3f}°, {d_lon:.3f}°)"
    elif payload.dest_key and payload.dest_key in ANTARCTIC_WAYPOINTS:
        d_lat = ANTARCTIC_WAYPOINTS[payload.dest_key]["lat"]
        d_lon = ANTARCTIC_WAYPOINTS[payload.dest_key]["lon"]
        dest_name = ANTARCTIC_WAYPOINTS[payload.dest_key]["name"]
    else:
        d_lat, d_lon = -69.407, 76.187 # Default Bharati Station
        dest_name = "Bharati Research Station (India)"

    icebergs = iceberg_service.list_icebergs()

    result = polar_route_optimizer.find_pareto_routes(
        origin_lat=o_lat,
        origin_lon=o_lon,
        dest_lat=d_lat,
        dest_lon=d_lon,
        icebergs=icebergs,
        vessel_ice_class=payload.vessel_ice_class,
        cruising_speed_knots=payload.cruising_speed_knots,
        departure_time_offset_hours=payload.departure_time_offset_hours,
        scenario=payload.scenario or "STANDARD"
    )

    result["origin_name"] = origin_name
    result["dest_name"] = dest_name
    result["scenario"] = payload.scenario or "STANDARD"
    result["departure_offset_hours"] = payload.departure_time_offset_hours

    return result

@router.post("/departure-sensitivity")
def evaluate_departure_window_sensitivity(payload: RouteOptimizationRequest):
    """
    Evaluates sensitivity of route corridors across departure windows (T+0h, T+24h, T+48h, T+72h).
    Enables expedition navigators to select optimal weather and sea-ice departure windows.
    """
    if payload.custom_origin_lat is not None and payload.custom_origin_lon is not None:
        o_lat, o_lon = payload.custom_origin_lat, payload.custom_origin_lon
    elif payload.origin_key and payload.origin_key in ANTARCTIC_WAYPOINTS:
        o_lat = ANTARCTIC_WAYPOINTS[payload.origin_key]["lat"]
        o_lon = ANTARCTIC_WAYPOINTS[payload.origin_key]["lon"]
    else:
        o_lat, o_lon = -33.918, 18.423

    if payload.custom_dest_lat is not None and payload.custom_dest_lon is not None:
        d_lat, d_lon = payload.custom_dest_lat, payload.custom_dest_lon
    elif payload.dest_key and payload.dest_key in ANTARCTIC_WAYPOINTS:
        d_lat = ANTARCTIC_WAYPOINTS[payload.dest_key]["lat"]
        d_lon = ANTARCTIC_WAYPOINTS[payload.dest_key]["lon"]
    else:
        d_lat, d_lon = -69.407, 76.187

    sensitivity = polar_route_optimizer.evaluate_departure_sensitivity(
        origin_lat=o_lat,
        origin_lon=o_lon,
        dest_lat=d_lat,
        dest_lon=d_lon,
        vessel_ice_class=payload.vessel_ice_class,
        cruising_speed_knots=payload.cruising_speed_knots,
        departure_offsets=[0.0, 24.0, 48.0, 72.0],
        scenario=payload.scenario or "STANDARD"
    )

    return {
        "origin_coordinates": [o_lat, o_lon],
        "destination_coordinates": [d_lat, d_lon],
        "vessel_ice_class": payload.vessel_ice_class,
        "scenario": payload.scenario or "STANDARD",
        "windows_evaluated": sensitivity,
        "recommendation": "Departure at T+0h or T+24h offers optimal transit time and POLARIS compliance before downstream pack ice convergence."
    }

@router.post("/export-geojson")
@router.post("/export")
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
                "safety_score": route["overall_safety_score"],
                "minimum_polaris_rio": route["minimum_polaris_rio"]
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
