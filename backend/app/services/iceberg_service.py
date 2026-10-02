import os
import csv
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from ..config import INITIAL_ICEBERGS
from ..models.iceberg_drift import iceberg_drift_engine

BYU_ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "data" / "byu_icebergs" / "updated7_consol"

class IcebergService:
    def __init__(self):
        # Database of active tracked icebergs
        self.icebergs: Dict[str, Dict[str, Any]] = {
            berg["id"]: berg.copy() for berg in INITIAL_ICEBERGS
        }
        self.last_live_sync_utc: Optional[str] = None
        self.load_from_byu_archive()
        # Attempt near-real-time satellite synchronization from BYU/ASCAT live feed
        try:
            self.sync_live_byu_feed(timeout_s=3.0)
        except Exception:
            pass

    def load_from_byu_archive(self):
        """Loads real satellite scatterometer observations from BYU/NIC consolidated archive."""
        if not BYU_ARCHIVE_DIR.exists():
            return

        archive_map = {
            "A-23a": "a23a.csv",
            "A-76a": "a76a.csv",
            "D-28": "d28.csv",
            "B-15ab": "b15ab.csv",
            "C-39": "c39.csv"
        }

        for berg_id, fname in archive_map.items():
            csv_path = BYU_ARCHIVE_DIR / fname
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
                # Find rows with non-zero coordinates
                non_zero = df[(df["nic_1"] != 0) | (df["ascat_1"] != 0)]
                if len(non_zero) > 0:
                    latest = non_zero.iloc[-1]
                    lat = float(latest["nic_1"] if latest["nic_1"] != 0 else latest["ascat_1"])
                    lon = float(latest["nic_2"] if latest["nic_2"] != 0 else latest["ascat_2"])
                    sz1 = float(latest.get("size_1", 0.0))
                    sz2 = float(latest.get("size_2", 0.0))

                    if berg_id in self.icebergs:
                        self.icebergs[berg_id]["lat"] = round(lat, 3)
                        self.icebergs[berg_id]["lon"] = round(lon, 3)
                        if sz1 > 0:
                            self.icebergs[berg_id]["length_km"] = sz1
                        if sz2 > 0:
                            self.icebergs[berg_id]["width_km"] = sz2
                        self.icebergs[berg_id]["surveillance_source"] = f"BYU/USNIC Archive ({fname}, Obs {int(latest['date'])})"
            except Exception:
                pass

    def sync_live_byu_feed(self, timeout_s: float = 4.0) -> Dict[str, Any]:
        """
        Fetches live satellite scatterometer observations directly from BYU's official
        near-real-time Antarctic Iceberg Tracking feed:
        https://www.scp.byu.edu/current_icebergs.html (ASCAT & OSCAT-2 in tandem).
        Updates positions of existing tracked icebergs and registers active newly calved bergs.
        Falls back smoothly to local BYU archive if offline or network unreachable.
        """
        import urllib.request
        import re
        import time

        def parse_dms(val_str: str) -> float:
            match = re.match(r'(\d+)\s+(\d+)\'?\s*([NSEWnsew])', val_str.strip())
            if not match:
                return 0.0
            deg, m, hemi = match.groups()
            val = float(deg) + float(m) / 60.0
            if hemi.upper() in ['S', 'W']:
                val = -val
            return round(val, 3)

        url = "https://www.scp.byu.edu/current_icebergs.html"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) POLARIS-AI/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            rows = re.findall(r'<tr>\s*<td>([a-zA-Z0-9_-]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*</tr>', html)
            if not rows:
                return {"status": "NO_RECORDS_PARSED", "live_count": 0, "fallback": "local_archive"}

            updated_count = 0
            new_count = 0

            shelf_thickness = {
                "A": 300.0, # Weddell / Ronne-Filchner
                "B": 240.0, # Ross Sea / Amundsen
                "C": 210.0, # Wilkes Land / D'Urville
                "D": 220.0  # Amery / Prydz Bay
            }

            for name, lon_str, lat_str, doy_str in rows:
                raw_name = name.strip()
                norm_id = raw_name.upper()
                m = re.match(r'([A-Z])(\d+)([A-Z]*)', norm_id)
                quad = m.group(1) if m else "A"
                canon_id = f"{quad}-{m.group(2)}{m.group(3).lower()}" if m else norm_id

                lat = parse_dms(lat_str)
                lon = parse_dms(lon_str)
                if lat == 0.0 and lon == 0.0:
                    continue

                doy = doy_str.strip()
                source_str = f"BYU/ASCAT & OSCAT-2 Live Satellite Scatterometer Feed (DOY {doy})"

                # Match existing or register
                matched_id = None
                for existing_id in list(self.icebergs.keys()):
                    if existing_id.upper().replace("-", "") == norm_id.replace("-", "") or existing_id.upper().startswith(canon_id.upper()):
                        matched_id = existing_id
                        break

                if matched_id:
                    self.icebergs[matched_id]["lat"] = lat
                    self.icebergs[matched_id]["lon"] = lon
                    self.icebergs[matched_id]["surveillance_source"] = source_str
                    self.icebergs[matched_id]["is_live"] = True
                    self.icebergs[matched_id]["observation_doy"] = doy
                    updated_count += 1
                else:
                    self.icebergs[canon_id] = {
                        "id": canon_id,
                        "name": f"Iceberg {canon_id}",
                        "calving_source": f"Antarctic Quadrant {quad} Shelf",
                        "lat": lat,
                        "lon": lon,
                        "area_km2": 450.0,
                        "length_km": 25.0,
                        "width_km": 15.0,
                        "thickness_m": shelf_thickness.get(quad, 220.0),
                        "mass_gt": 75.0,
                        "drift_speed_knots": 0.8,
                        "drift_bearing_deg": 315.0,
                        "status": f"Active Satellite Track (DOY {doy})",
                        "hazard_level": "HIGH" if lat > -65.0 else "MODERATE",
                        "surveillance_source": source_str,
                        "is_live": True,
                        "observation_doy": doy
                    }
                    new_count += 1

            self.last_live_sync_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            return {
                "status": "LIVE_FEED_SYNCED",
                "source_url": url,
                "total_icebergs_tracked": len(self.icebergs),
                "updated_icebergs_count": updated_count,
                "newly_registered_count": new_count,
                "sync_timestamp_utc": self.last_live_sync_utc
            }
        except Exception as e:
            return {
                "status": "OFFLINE_FALLBACK",
                "reason": str(e),
                "fallback_source": "Local BYU / USNIC Consolidated Archive",
                "total_icebergs_tracked": len(self.icebergs)
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
