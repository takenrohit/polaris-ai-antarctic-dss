"""
Iceberg Tracking and Surveillance Service.
Maintains active Antarctic icebergs, integrates US NIC / BYU Antarctic Iceberg Database records,
supports automated feed ingestion (CSV / JSON), and executes physics trajectory simulations.
"""
import csv
import io
from typing import List, Dict, Any, Optional
from ..config import INITIAL_ICEBERGS
from ..models.iceberg_drift import iceberg_drift_engine

class IcebergService:
    def __init__(self):
        # Database of active tracked icebergs initialized from official NIC baseline
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

    def ingest_from_nic_feed(self, raw_data: List[Dict[str, Any]]) -> int:
        """Ingests a batch of iceberg observation records from US NIC / BYU JSON feed."""
        count = 0
        for item in raw_data:
            b_id = item.get("id") or item.get("iceberg_name")
            if b_id:
                self.icebergs[b_id] = {
                    "id": b_id,
                    "name": item.get("name", f"Iceberg {b_id}"),
                    "calving_source": item.get("calving_source", "Antarctic Ice Shelf"),
                    "origin_year": item.get("origin_year", 2020),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "area_km2": float(item.get("area_km2", 200.0)),
                    "length_km": float(item.get("length_km", 15.0)),
                    "width_km": float(item.get("width_km", 10.0)),
                    "thickness_m": float(item.get("thickness_m", 200.0)),
                    "mass_gt": float(item.get("mass_gt", 40.0)),
                    "drift_speed_knots": float(item.get("drift_speed_knots", 0.8)),
                    "drift_bearing_deg": float(item.get("drift_bearing_deg", 45.0)),
                    "status": item.get("status", "Active Surveillance"),
                    "hazard_level": item.get("hazard_level", "MODERATE"),
                    "surveillance_source": item.get("surveillance_source", "US NIC / Sentinel-1")
                }
                count += 1
        return count

    def ingest_from_csv(self, csv_content: str) -> int:
        """Ingests iceberg observations from CSV file conforming to standard oceanographic columns."""
        reader = csv.DictReader(io.StringIO(csv_content))
        items = []
        for row in reader:
            items.append({
                "id": row.get("id", "").strip(),
                "name": row.get("name", "").strip(),
                "lat": float(row.get("lat", 0.0)),
                "lon": float(row.get("lon", 0.0)),
                "area_km2": float(row.get("area_km2", 100.0)),
                "length_km": float(row.get("length_km", 10.0)),
                "width_km": float(row.get("width_km", 5.0)),
                "thickness_m": float(row.get("thickness_m", 200.0)),
                "drift_speed_knots": float(row.get("drift_speed_knots", 0.8)),
                "drift_bearing_deg": float(row.get("drift_bearing_deg", 0.0)),
                "surveillance_source": row.get("surveillance_source", "CSV Import")
            })
        return self.ingest_from_nic_feed(items)

iceberg_service = IcebergService()
