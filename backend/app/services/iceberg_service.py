"""
Iceberg Tracking and Surveillance Service.
Maintains active Antarctic icebergs, integrates Sentinel-1 SAR/CryoSat observation feeds,
and executes trajectory forecast simulations.
"""
from typing import List, Dict, Any, Optional
from ..config import INITIAL_ICEBERGS
from ..models.iceberg_drift import iceberg_drift_engine

class IcebergService:
    def __init__(self):
        # In-memory database of active tracked icebergs
        self.icebergs: Dict[str, Dict[str, Any]] = {
            berg["id"]: berg.copy() for berg in INITIAL_ICEBERGS
        }

    def list_icebergs(self) -> List[Dict[str, Any]]:
        return list(self.icebergs.values())

    def get_iceberg(self, berg_id: str) -> Optional[Dict[str, Any]]:
        return self.icebergs.get(berg_id)

    def forecast_iceberg(self, berg_id: str, forecast_hours: int = 120) -> Dict[str, Any]:
        berg = self.get_iceberg(berg_id)
        if not berg:
            raise ValueError(f"Iceberg {berg_id} not found in database.")
        return iceberg_drift_engine.predict_trajectory(berg, forecast_hours=forecast_hours)

    def forecast_all_icebergs(self, forecast_hours: int = 120) -> List[Dict[str, Any]]:
        return [
            iceberg_drift_engine.predict_trajectory(berg, forecast_hours=forecast_hours)
            for berg in self.icebergs.values()
        ]

    def register_iceberg(self, berg_data: Dict[str, Any]) -> Dict[str, Any]:
        berg_id = berg_data.get("id") or f"ICE-{len(self.icebergs)+1:03d}"
        berg_data["id"] = berg_id
        if "hazard_level" not in berg_data:
            berg_data["hazard_level"] = "HIGH" if berg_data.get("area_km2", 0) > 500 else "MODERATE"
        self.icebergs[berg_id] = berg_data
        return berg_data

iceberg_service = IcebergService()
