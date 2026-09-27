"""
Configuration and Domain Constants for MoES / NCPOR Antarctic Navigation DSS.
Includes Indian Antarctic Research Stations (Maitri, Bharati), gateways,
IMO Polar Code vessel classes (PC1-PC7), and polar grid definitions.
"""
from pydantic_settings import BaseSettings
from typing import Dict, Any, List

class Settings(BaseSettings):
    APP_NAME: str = "POLARIS-AI: MoES/NCPOR Antarctic Navigation Decision Support System"
    API_V1_PREFIX: str = "/api"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "*"]

    # Spatial bounds for Antarctic Operational Theater (Southern Ocean: 50°S to 82°S)
    LAT_MIN: float = -82.0
    LAT_MAX: float = -50.0
    LON_MIN: float = -180.0
    LON_MAX: float = 180.0
    
    # Grid resolution for sea ice inference matrix (degrees)
    GRID_LAT_STEP: float = 0.5
    GRID_LON_STEP: float = 1.0

settings = Settings()

# Antarctic Research Stations & Departure Gateways
ANTARCTIC_WAYPOINTS = {
    # Indian Research Stations
    "MAITRI_STATION": {
        "name": "Maitri Research Station (India)",
        "code": "MAITRI",
        "lat": -70.767,
        "lon": 11.733,
        "region": "Queen Maud Land / Schirmacher Oasis",
        "type": "research_station",
        "access_fast_ice_zone": True,
        "elevation_m": 117
    },
    "BHARATI_STATION": {
        "name": "Bharati Research Station (India)",
        "code": "BHARATI",
        "lat": -69.407,
        "lon": 76.187,
        "region": "Larsemann Hills / Prydz Bay",
        "type": "research_station",
        "access_fast_ice_zone": True,
        "elevation_m": 35
    },
    "DAKSHIN_GANGOTRI": {
        "name": "Dakshin Gangotri Site (India - Historic Base)",
        "code": "DG",
        "lat": -70.083,
        "lon": 12.000,
        "region": "Ice Shelf",
        "type": "historic_depot",
        "access_fast_ice_zone": True,
        "elevation_m": 0
    },
    # Gateway Ports
    "PORT_CAPE_TOWN": {
        "name": "Port of Cape Town (South Africa Gateway)",
        "code": "CPT",
        "lat": -33.918,
        "lon": 18.423,
        "region": "Atlantic Gateway",
        "type": "gateway_port"
    },
    "PORT_PUNTA_ARENAS": {
        "name": "Punta Arenas (Chile Gateway)",
        "code": "PUQ",
        "lat": -53.163,
        "lon": -70.917,
        "region": "Patagonia Gateway",
        "type": "gateway_port"
    },
    "PORT_HOBART": {
        "name": "Hobart (Australia Gateway)",
        "code": "HBA",
        "lat": -42.882,
        "lon": 147.327,
        "region": "Tasman Gateway",
        "type": "gateway_port"
    },
    "PORT_GOA_MORMUGAO": {
        "name": "Mormugao Port / NCPOR HQ (Goa, India)",
        "code": "GOA",
        "lat": 15.410,
        "lon": 73.800,
        "region": "Home Port",
        "type": "gateway_port"
    },
    # Polar Ocean Reference Waypoints (Entry into sea ice belts)
    "PRYDZ_BAY_APPROACH": {
        "name": "Prydz Bay Outer Shelf",
        "code": "PBA",
        "lat": -66.500,
        "lon": 75.000,
        "region": "Prydz Bay",
        "type": "ocean_waypoint"
    },
    "QUEEN_MAUD_APPROACH": {
        "name": "Lazarev Sea Approach",
        "code": "QMA",
        "lat": -67.000,
        "lon": 12.000,
        "region": "Lazarev Sea",
        "type": "ocean_waypoint"
    },
    "WEDDELL_SEA_ENTRANCE": {
        "name": "Weddell Sea Outer Drift Area",
        "code": "WSE",
        "lat": -63.500,
        "lon": -45.000,
        "region": "Weddell Sea",
        "type": "ocean_waypoint"
    },
    "ROSS_SEA_APPROACH": {
        "name": "Ross Sea Outer Pack Ice",
        "code": "RSA",
        "lat": -71.000,
        "lon": 175.000,
        "region": "Ross Sea",
        "type": "ocean_waypoint"
    }
}

# IMO Polar Code Vessel Ice Classes & POLARIS Risk Weightings
# According to IMO MSC.1/Circ.1519 Polar Operational Limit Assessment Risk Indexing System (POLARIS)
IMO_POLAR_CLASSES = {
    "PC1": {
        "name": "Polar Class 1",
        "description": "Year-round operation in all polar waters",
        "max_tolerated_ice_conc": 1.0,
        "fuel_penalty_factor": 1.05,
        "ice_speed_penalty": 0.15, # slight speed drop in 100% ice
        "risk_tolerance": 0.95
    },
    "PC3": {
        "name": "Polar Class 3",
        "description": "Year-round operation in second-year ice (includes multi-year ice inclusions)",
        "max_tolerated_ice_conc": 0.85,
        "fuel_penalty_factor": 1.35,
        "ice_speed_penalty": 0.35,
        "risk_tolerance": 0.80
    },
    "PC5": {
        "name": "Polar Class 5 (Typical NCPOR Charter: e.g. SA Agulhas II / Sagar Nidhi refit)",
        "description": "Year-round operation in medium first-year ice (includes old ice inclusions)",
        "max_tolerated_ice_conc": 0.65,
        "fuel_penalty_factor": 1.80,
        "ice_speed_penalty": 0.55,
        "risk_tolerance": 0.60
    },
    "PC7": {
        "name": "Polar Class 7",
        "description": "Summer/autumn operation in thin first-year ice",
        "max_tolerated_ice_conc": 0.35,
        "fuel_penalty_factor": 2.40,
        "ice_speed_penalty": 0.75,
        "risk_tolerance": 0.30
    },
    "OPEN_WATER": {
        "name": "Non-Ice Strengthened Vessel",
        "description": "Open water navigation only (Marginal Ice Zone avoidance)",
        "max_tolerated_ice_conc": 0.10,
        "fuel_penalty_factor": 3.50,
        "ice_speed_penalty": 0.90,
        "risk_tolerance": 0.10
    }
}

# Real Tracked Major Antarctic Icebergs (NIC & BYU Iceberg Database baseline)
INITIAL_ICEBERGS = [
    {
        "id": "A-23a",
        "name": "Iceberg A-23a (Megaberg)",
        "calving_source": "Filchner-Ronne Ice Shelf",
        "origin_year": 1986,
        "lat": -60.2,
        "lon": -51.5,
        "area_km2": 3800,
        "length_km": 68.0,
        "width_km": 54.0,
        "thickness_m": 350.0,
        "mass_gt": 1100.0,
        "drift_speed_knots": 1.1,
        "drift_bearing_deg": 42.0,
        "status": "Active Drift (Scotia Sea northward track)",
        "hazard_level": "CRITICAL",
        "surveillance_source": "Sentinel-1 SAR / MODIS Composite"
    },
    {
        "id": "A-76a",
        "name": "Iceberg A-76a",
        "calving_source": "Ronne Ice Shelf",
        "origin_year": 2021,
        "lat": -56.8,
        "lon": -38.2,
        "area_km2": 1850,
        "length_km": 45.0,
        "width_km": 30.0,
        "thickness_m": 280.0,
        "mass_gt": 480.0,
        "drift_speed_knots": 1.6,
        "drift_bearing_deg": 55.0,
        "status": "Calving Sub-fragments in South Georgia Corridor",
        "hazard_level": "HIGH",
        "surveillance_source": "CryoSat-2 & Sentinel-2"
    },
    {
        "id": "D-28",
        "name": "Iceberg D-28",
        "calving_source": "Amery Ice Shelf (Prydz Bay Sector)",
        "origin_year": 2019,
        "lat": -64.4,
        "lon": 68.2,
        "area_km2": 1580,
        "length_km": 48.0,
        "width_km": 32.0,
        "thickness_m": 210.0,
        "mass_gt": 315.0,
        "drift_speed_knots": 0.8,
        "drift_bearing_deg": 285.0,
        "status": "Circumpolar Coastal Current Drift (Westward towards Bharati corridor)",
        "hazard_level": "HIGH",
        "surveillance_source": "Sentinel-1 SAR / BYU NIC"
    },
    {
        "id": "B-15ab",
        "name": "Iceberg B-15 Fragment Alpha",
        "calving_source": "Ross Ice Shelf",
        "origin_year": 2000,
        "lat": -68.1,
        "lon": 169.5,
        "area_km2": 420,
        "length_km": 28.0,
        "width_km": 15.0,
        "thickness_m": 220.0,
        "mass_gt": 88.0,
        "drift_speed_knots": 0.65,
        "drift_bearing_deg": 310.0,
        "status": "Decaying Fragment in Ross Sea Gyre",
        "hazard_level": "MODERATE",
        "surveillance_source": "AMSR2 / NIC Ice Center"
    },
    {
        "id": "C-39",
        "name": "Iceberg C-39",
        "calving_source": "Cook Ice Shelf / Wilkes Land",
        "origin_year": 2022,
        "lat": -65.1,
        "lon": 128.4,
        "area_km2": 320,
        "length_km": 22.0,
        "width_km": 14.0,
        "thickness_m": 190.0,
        "mass_gt": 55.0,
        "drift_speed_knots": 0.75,
        "drift_bearing_deg": 260.0,
        "status": "East Antarctic Coastal Current",
        "hazard_level": "MODERATE",
        "surveillance_source": "Sentinel-1 SAR"
    }
]
