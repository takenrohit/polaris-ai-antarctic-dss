"""
FastAPI Endpoints and WebSocket for Real-Time Telemetry and NAVAREA Warnings.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import random
from ..services.alert_service import alert_service
from ..services.vessel_service import vessel_service

router = APIRouter(prefix="/telemetry", tags=["Telemetry & Alerts"])

@router.get("/alerts")
def get_active_navigational_alerts():
    """Returns active polar hazard bulletins and NAVAREA warnings."""
    return alert_service.list_alerts()

@router.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Simulated AIS Telemetry Demonstration Channel for shipboard ECDIS and decision dashboard.
    Streams synthetic expedition vessel positions, dynamic micro-drifts, and
    instantaneous POLARIS RIO operational states for bridge interface demonstration.
    """
    await websocket.accept()
    vessels = vessel_service.list_vessels()

    try:
        step = 0
        while True:
            step += 1
            # Broadcast simulated AIS dynamic vessel updates
            telemetry_data = []
            for v in vessels:
                v_copy = dict(v)
                pos = dict(v_copy["current_position"])
                
                # Simulated micro drift advance
                pos["speed_knots"] = round(pos["speed_knots"] + random.uniform(-0.15, 0.15), 1)
                pos["fuel_flow_mth"] = round(max(0.3, pos["fuel_flow_mth"] + random.uniform(-0.02, 0.02)), 2)
                pos["lat"] = round(pos["lat"] + random.uniform(-0.002, 0.002), 4)
                pos["lon"] = round(pos["lon"] + random.uniform(-0.003, 0.003), 4)

                telemetry_data.append({
                    "vessel_id": v["id"],
                    "name": v["name"],
                    "position": pos,
                    "ice_class": v["ice_class"],
                    "status": "EN_ROUTE (SIMULATED)"
                })

            payload = {
                "type": "SIMULATED_TELEMETRY_UPDATE",
                "timestamp_tick": step,
                "fleet": telemetry_data,
                "navarea_status": "NORMAL_WATCH",
                "active_alerts_count": len(alert_service.list_alerts())
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(3.0)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
