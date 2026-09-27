"""
Navigational Hazard Alert and NAVAREA Warning Service for Antarctic Waters.
"""
from typing import List, Dict, Any

ACTIVE_ALERTS = [
    {
        "id": "ALERT-MOES-2026-081",
        "timestamp": "2026-09-27T18:00:00Z",
        "severity": "CRITICAL",
        "source": "NAVAREA VI / MoES NCPOR Polar Center",
        "title": "MEGABERG A-23a RAPID NORTHWARD DRIFT",
        "description": "Iceberg A-23a (approx 3,800 sq km) experiencing intensified northeastward advection into Scotia Sea shipping lanes. Dense cluster of calved 'growlers' and 'bergy bits' within 30 NM radius.",
        "coordinates": {"lat": -60.2, "lon": -51.5},
        "recommended_action": "Maintain 35 NM stand-off; radar watch and searchlight readiness at 100%."
    },
    {
        "id": "ALERT-MOES-2026-079",
        "timestamp": "2026-09-27T14:30:00Z",
        "severity": "HIGH",
        "source": "NCPOR Prydz Bay Coastal Station",
        "title": "FAST-ICE BREAKUP & PRESSURE RIDGING NEAR BHARATI",
        "description": "Sentinel-1 SAR interferometry indicates early fast-ice fracture front in Prydz Bay (69°24'S, 76°11'E). Heavy compressive ice ridging up to 2.8m thickness observed.",
        "coordinates": {"lat": -69.4, "lon": 76.2},
        "recommended_action": "Vessels without PC4 or higher require ice reconnaissance helicopter scouting before final dock approach."
    },
    {
        "id": "ALERT-MOES-2026-077",
        "timestamp": "2026-09-27T08:15:00Z",
        "severity": "WARNING",
        "source": "ECMWF / IMD Antarctic Weather Division",
        "title": "KATABATIC GALE WARNING: QUEEN MAUD LAND",
        "description": "Intense katabatic wind event predicted off continental slope near Maitri Station / Lazarev Sea. Wind gusts exceeding 65 knots, rapid sea-ice compaction along ice barrier.",
        "coordinates": {"lat": -70.7, "lon": 11.7},
        "recommended_action": "Avoid entering dense pack within 40 NM of coast during peak 36-hour storm window."
    }
]

class AlertService:
    def __init__(self):
        self.alerts = list(ACTIVE_ALERTS)

    def list_alerts(self) -> List[Dict[str, Any]]:
        return self.alerts

alert_service = AlertService()
