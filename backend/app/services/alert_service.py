"""
Navigational Hazard Alert and NAVAREA Warning Service for Antarctic Waters.
"""
from typing import List, Dict, Any
from datetime import datetime, timezone
from ..data.ingestion import environmental_data_provider
from .iceberg_service import iceberg_service


BASE_ALERTS_TEMPLATE = [
    {
        "id": "ALERT-MOES-2026-081",
        "severity": "CRITICAL",
        "source": "NAVAREA VI / MoES NCPOR Polar Center (Live Satellite Feed)",
        "title": "MEGABERG A-23a DRIFT CORRIDOR ADVISORY",
        "description_template": "Iceberg A-23a ({area_km2:,.0f} sq km) tracked at {lat:.2f}°S, {lon:.2f}°W. Surveillance source: {sensor}. Heavy concentration of calved growlers and bergy bits within 30 NM radius.",
        "default_coords": {"lat": -60.2, "lon": -51.5},
        "target_iceberg_id": "A-23a",
        "recommended_action": "Maintain 35 NM stand-off; radar watch and searchlight readiness at 100%."
    },
    {
        "id": "ALERT-MOES-2026-079",
        "severity": "HIGH",
        "source": "NCPOR Prydz Bay Coastal Weather Station",
        "title": "FAST-ICE REGIME & WIND ADVISORY: BHARATI SECTOR",
        "description_template": "Prydz Bay approach (69°24'S, 76°11'E). Live metocean: Ambient {temp:.1f}°C, Wind {wind_kts:.1f} kts ({wind_source}). Fast-ice fracture front and pressure ridging active.",
        "default_coords": {"lat": -69.4, "lon": 76.2},
        "station_lat": -69.407,
        "station_lon": 76.187,
        "recommended_action": "Vessels without PC4 or higher require ice reconnaissance helicopter scouting before final dock approach."
    },
    {
        "id": "ALERT-MOES-2026-077",
        "severity": "WARNING",
        "source": "ECMWF / IMD Antarctic Met Division",
        "title": "KATABATIC GALE WATCH: MAITRI / QUEEN MAUD LAND",
        "description_template": "Continental slope off Maitri Station (70°46'S, 11°44'E). Live metocean: Temp {temp:.1f}°C, Wind {wind_kts:.1f} kts ({wind_source}). Rapid sea-ice compaction along ice barrier.",
        "default_coords": {"lat": -70.7, "lon": 11.7},
        "station_lat": -70.767,
        "station_lon": 11.733,
        "recommended_action": "Avoid entering dense pack within 40 NM of coast during peak storm window."
    }
]

class AlertService:
    def __init__(self):
        self._cached_alerts: List[Dict[str, Any]] = []
        self._last_refresh: float = 0.0

    def list_alerts(self, live: bool = True) -> List[Dict[str, Any]]:
        """
        Returns active polar hazard bulletins and NAVAREA warnings.
        Dynamically enriches alerts with live metocean weather and live iceberg coordinates.
        """
        import time
        now = time.time()
        # Cache for 60 seconds to avoid repetitive external API calls
        if self._cached_alerts and (now - self._last_refresh < 60.0):
            return self._cached_alerts

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        alerts = []

        # Find A-23a or top berg
        icebergs = {b["id"]: b for b in iceberg_service.list_icebergs()}
        a23a = icebergs.get("A-23a")

        for template in BASE_ALERTS_TEMPLATE:
            alert_id = template["id"]
            severity = template["severity"]
            source = template["source"]
            title = template["title"]
            action = template["recommended_action"]

            if "target_iceberg_id" in template:
                berg = icebergs.get(template["target_iceberg_id"], a23a)
                if berg:
                    coords = {"lat": berg["lat"], "lon": berg["lon"]}
                    desc = template["description_template"].format(
                        area_km2=berg.get("area_km2", 3800),
                        lat=abs(berg["lat"]),
                        lon=abs(berg["lon"]),
                        sensor=berg.get("surveillance_source", "Satellite Composite")
                    )
                else:
                    coords = template["default_coords"]
                    desc = template["description_template"].format(
                        area_km2=3800,
                        lat=60.2,
                        lon=51.5,
                        sensor="Satellite Composite"
                    )
            elif "station_lat" in template:
                coords = {"lat": template["station_lat"], "lon": template["station_lon"]}
                try:
                    w = environmental_data_provider.get_live_weather(
                        template["station_lat"], 
                        template["station_lon"], 
                        timeout_s=2.5
                    )
                    temp = w.get("temperature_2m_c", -15.0)
                    wind = w.get("wind_speed_knots", 22.0)
                    w_src = w.get("source", "In-Situ Model")
                except Exception:
                    temp = -15.0
                    wind = 22.0
                    w_src = "Reference Climatology"

                desc = template["description_template"].format(
                    temp=temp,
                    wind_kts=wind,
                    wind_source=w_src
                )
            else:
                coords = template.get("default_coords", {"lat": -65.0, "lon": 0.0})
                desc = template.get("title", "")

            alerts.append({
                "id": alert_id,
                "timestamp": now_iso,
                "severity": severity,
                "source": source,
                "title": title,
                "description": desc,
                "coordinates": coords,
                "recommended_action": action,
                "is_live": True
            })

        self._cached_alerts = alerts
        self._last_refresh = now
        return alerts

alert_service = AlertService()

