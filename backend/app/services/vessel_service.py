"""
Vessel Service for NCPOR Indian Antarctic Research Fleet.
Supports vessels like:
- MV Vasiliy Golovnin (Key polar charter vessel for 41st - 44th Indian Antarctic Expeditions)
- SA Agulhas II (Polar Class 5 Icebreaking research vessel)
- ORV Sagar Nidhi (MoES Ice-strengthened Oceanographic Vessel)
"""
from typing import List, Dict, Any, Optional

NCPOR_FLEET = [
    {
        "id": "VESSEL-01",
        "name": "MV Vasiliy Golovnin (44th IAE Expedition Ship)",
        "flag": "Russia (Chartered by NCPOR/MoES)",
        "call_sign": "UBDX",
        "mmsi": "273138000",
        "ice_class": "PC5",
        "length_m": 163.0,
        "beam_m": 22.4,
        "draft_m": 9.0,
        "engine_power_kw": 12800,
        "crew_expeditioners": 72,
        "cargo_capacity_tons": 8500,
        "helicopter_capable": True,
        "current_mission": "44th Indian Antarctic Expedition Resupply (Maitri & Bharati)",
        "current_position": {
            "lat": -67.45,
            "lon": 72.80,
            "speed_knots": 10.2,
            "heading_deg": 145.0,
            "fuel_flow_mth": 0.85,
            "ice_class_rio": 5.4,
            "sea_ice_conc_pct": 28.0
        }
    },
    {
        "id": "VESSEL-02",
        "name": "ORV Sagar Nidhi (MoES Flagship)",
        "flag": "India (Ministry of Earth Sciences)",
        "call_sign": "VWCY",
        "mmsi": "419069500",
        "ice_class": "PC7",
        "length_m": 104.0,
        "beam_m": 18.0,
        "draft_m": 6.8,
        "engine_power_kw": 8400,
        "crew_expeditioners": 55,
        "cargo_capacity_tons": 3200,
        "helicopter_capable": False,
        "current_mission": "Southern Ocean Hydrographic & Bathymetric Survey",
        "current_position": {
            "lat": -56.30,
            "lon": 55.20,
            "speed_knots": 12.8,
            "heading_deg": 195.0,
            "fuel_flow_mth": 0.62,
            "ice_class_rio": 12.0,
            "sea_ice_conc_pct": 0.0
        }
    },
    {
        "id": "VESSEL-03",
        "name": "SA Agulhas II (Deep Polar Research Icebreaker)",
        "flag": "South Africa (MoES Joint Research Partner)",
        "call_sign": "ZR6367",
        "mmsi": "601110000",
        "ice_class": "PC5",
        "length_m": 134.2,
        "beam_m": 21.7,
        "draft_m": 7.7,
        "engine_power_kw": 12000,
        "crew_expeditioners": 100,
        "cargo_capacity_tons": 4000,
        "helicopter_capable": True,
        "current_mission": "Cape Town to Queen Maud Land Maitri Supply Corridor",
        "current_position": {
            "lat": -63.10,
            "lon": 14.50,
            "speed_knots": 11.5,
            "heading_deg": 182.0,
            "fuel_flow_mth": 0.92,
            "ice_class_rio": 7.8,
            "sea_ice_conc_pct": 18.0
        }
    }
]

class VesselService:
    def __init__(self):
        self.vessels: Dict[str, Dict[str, Any]] = {
            v["id"]: v.copy() for v in NCPOR_FLEET
        }

    def list_vessels(self, enrich_live_weather: bool = True) -> List[Dict[str, Any]]:
        from ..data.ingestion import environmental_data_provider
        results = []
        for v in self.vessels.values():
            v_copy = dict(v)
            pos = dict(v_copy["current_position"])
            if enrich_live_weather:
                try:
                    lw = environmental_data_provider.get_live_weather(pos["lat"], pos["lon"], timeout_s=2.5)
                    pos["ambient_temp_c"] = lw.get("temperature_c")
                    pos["wind_speed_knots"] = lw.get("wind_speed_knots")
                    pos["wind_direction_deg"] = lw.get("wind_direction_deg")
                    pos["surface_pressure_hpa"] = lw.get("surface_pressure_hpa")
                    pos["wave_height_m"] = lw.get("wave_height_m")
                    pos["significant_wave_height_m"] = lw.get("wave_height_m")
                    pos["weather_source"] = lw.get("data_source")
                    pos["weather_is_live"] = lw.get("is_live", False)

                except Exception:
                    pass
            v_copy["current_position"] = pos
            results.append(v_copy)
        return results

    def get_vessel(self, vessel_id: str) -> Optional[Dict[str, Any]]:
        return self.vessels.get(vessel_id)

vessel_service = VesselService()
