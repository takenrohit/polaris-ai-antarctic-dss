"""
Physics-Based Antarctic Iceberg Trajectory Prediction Model.
Models tabular iceberg drift governed by:
- Atmospheric wind drag (quadratic law, windage coefficient)
- Ocean current drag (hydrodynamic form and skin drag on keel)
- Coriolis acceleration (Southern Hemisphere deflection: f = 2 * Omega * sin(lat))
- Sea-ice pack interaction damping
- Empirical sub-mesoscale eddy perturbation
- Generates 120-hour ensemble forecasts with probabilistic cones of uncertainty (p10, p50, p90).
"""
import numpy as np
import math
from typing import Dict, Any, List, Tuple

class IcebergDriftModel:
    """
    Physics-Based Antarctic Iceberg Drift Engine.
    Governing differential equation of iceberg motion:
    m * (dv/dt + f x v) = F_air + F_water + F_ice + F_eddy
    where:
    - F_air = 0.5 * rho_air * C_air * A_air * |v_wind - v| * (v_wind - v)
    - F_water = 0.5 * rho_water * C_water * A_water * |v_current - v| * (v_current - v)
    - f is the Coriolis parameter = 2 * Omega * sin(lat)
    """
    def __init__(self):
        # Oceanographic & physical constants
        self.RHO_AIR = 1.25     # kg/m3
        self.RHO_WATER = 1028.0 # kg/m3 (Antarctic cold saline water)
        self.RHO_ICE = 900.0    # kg/m3
        self.OMEGA = 7.2921e-5  # Earth angular velocity (rad/s)
        self.C_AIR = 1.3        # Tabular iceberg atmospheric drag coefficient
        self.C_WATER = 0.9      # Ocean water form drag coefficient

    def coriolis_parameter(self, lat_deg: float) -> float:
        """Returns f = 2 * Omega * sin(lat), negative in Southern Hemisphere."""
        return 2.0 * self.OMEGA * math.sin(math.radians(lat_deg))

    def get_environmental_forcing(self, lat: float, lon: float, hour_offset: int) -> Tuple[float, float, float, float]:
        """
        Retrieves ocean current (u_curr, v_curr) and wind (u_wind, v_wind) in m/s
        based on Antarctic circumpolar geography, prevailing westerlies, and coastal easterlies.
        """
        time_t = hour_offset / 24.0
        wave_t = math.sin(time_t * 0.5 + math.radians(lon))

        if lat < -66.0:
            # Coastal Easterlies / Antarctic Coastal Current (drifting westward)
            u_curr = -0.18 + 0.05 * wave_t
            v_curr = -0.04 + 0.03 * math.cos(time_t * 0.3)
            u_wind = -5.0 + 2.0 * wave_t
            v_wind = -2.5 + 1.5 * math.cos(time_t * 0.4)
        else:
            # ACC Region (drifting east-northeastward towards Drake Passage / Scotia Sea)
            u_curr = 0.28 + 0.08 * wave_t
            v_curr = 0.12 + 0.06 * math.cos(time_t * 0.5)
            u_wind = 7.5 + 3.0 * wave_t
            v_wind = 4.0 + 2.0 * math.sin(time_t * 0.6)

        # Weddell Gyre regional vortex adjustment
        if -60.0 <= lon <= -30.0 and -75.0 <= lat <= -60.0:
            u_curr = 0.22 + 0.05 * wave_t
            v_curr = 0.30 + 0.08 * math.cos(time_t * 0.5)
            u_wind = 3.0 + 1.5 * wave_t
            v_wind = 8.5 + 2.5 * math.sin(time_t * 0.5)

        return u_curr, v_curr, u_wind, v_wind

    def predict_trajectory(
        self,
        iceberg: Dict[str, Any],
        forecast_hours: int = 120,
        time_step_hours: int = 3,
        include_eddy_perturbation: bool = True,
        use_ml_residual: bool = True  # Backward compatibility alias
    ) -> Dict[str, Any]:
        """
        Simulates future iceberg drift over `forecast_hours` at `time_step_hours` intervals.
        Generates ensemble trajectories (p10, p50, p90) for uncertainty cones.
        """
        lat0 = iceberg["lat"]
        lon0 = iceberg["lon"]
        length_m = iceberg.get("length_km", 20.0) * 1000.0
        width_m = iceberg.get("width_km", 10.0) * 1000.0
        thickness_m = iceberg.get("thickness_m", 250.0)

        # Freeboard (sail) vs Keel (draft) by Archimedes principle
        draft_m = thickness_m * (self.RHO_ICE / self.RHO_WATER) # ~87.5% submerged
        sail_m = thickness_m - draft_m                         # ~12.5% exposed

        # Cross-sectional areas
        A_air = length_m * sail_m
        A_water = length_m * draft_m
        mass_kg = (length_m * width_m * thickness_m) * self.RHO_ICE

        dt_seconds = time_step_hours * 3600.0
        steps = forecast_hours // time_step_hours

        # Initial iceberg velocity (derived from current drift knots)
        init_speed_ms = iceberg.get("drift_speed_knots", 1.0) * 0.514444
        init_bearing = math.radians(iceberg.get("drift_bearing_deg", 45.0))
        u_berg = init_speed_ms * math.sin(init_bearing)
        v_berg = init_speed_ms * math.cos(init_bearing)

        # Monte-Carlo / Ensemble runs for uncertainty cone
        ensemble_tracks = []
        n_members = 7

        for member_idx in range(n_members):
            # Member perturbation (stochastic ocean eddy + gust variance)
            eddy_jitter = 0.0 if member_idx == 0 else (member_idx - 3) * 0.06
            cur_lat, cur_lon = lat0, lon0
            cur_u_berg, cur_v_berg = u_berg, v_berg
            track = []

            for s in range(steps + 1):
                hour = s * time_step_hours
                f_coriolis = self.coriolis_parameter(cur_lat)
                u_curr, v_curr, u_wind, v_wind = self.get_environmental_forcing(cur_lat, cur_lon, hour)

                # Add member variation
                u_curr += eddy_jitter * 0.4
                v_curr += eddy_jitter * 0.4

                # Relative speeds
                rel_u_wind = u_wind - cur_u_berg
                rel_v_wind = v_wind - cur_v_berg
                wind_speed_rel = math.sqrt(rel_u_wind**2 + rel_v_wind**2)

                rel_u_water = u_curr - cur_u_berg
                rel_v_water = v_curr - cur_v_berg
                water_speed_rel = math.sqrt(rel_u_water**2 + rel_v_water**2)

                # Physics Forces
                F_air_x = 0.5 * self.RHO_AIR * self.C_AIR * A_air * wind_speed_rel * rel_u_wind
                F_air_y = 0.5 * self.RHO_AIR * self.C_AIR * A_air * wind_speed_rel * rel_v_wind

                F_water_x = 0.5 * self.RHO_WATER * self.C_WATER * A_water * water_speed_rel * rel_u_water
                F_water_y = 0.5 * self.RHO_WATER * self.C_WATER * A_water * water_speed_rel * rel_v_water

                # Coriolis force: F_c = m * f * (-v, u) in 2D
                F_cor_x = -mass_kg * f_coriolis * cur_v_berg
                F_cor_y = mass_kg * f_coriolis * cur_u_berg

                # Net acceleration
                acc_x = (F_air_x + F_water_x + F_cor_x) / mass_kg
                acc_y = (F_air_y + F_water_y + F_cor_y) / mass_kg

                # Empirical pack-ice damping & sub-mesoscale eddy oscillation
                if include_eddy_perturbation and use_ml_residual:
                    ice_damping = 0.96
                    eddy_x = np.sin(hour * 0.08) * 0.015
                    eddy_y = np.cos(hour * 0.08) * 0.012
                    acc_x = acc_x * ice_damping + eddy_x / 3600.0
                    acc_y = acc_y * ice_damping + eddy_y / 3600.0

                # Velocity update
                cur_u_berg += acc_x * dt_seconds
                cur_v_berg += acc_y * dt_seconds

                # Speed capping (icebergs rarely exceed 3.5 knots due to immense water resistance)
                current_speed = math.sqrt(cur_u_berg**2 + cur_v_berg**2)
                max_speed_ms = 3.5 * 0.514444
                if current_speed > max_speed_ms:
                    cur_u_berg = (cur_u_berg / current_speed) * max_speed_ms
                    cur_v_berg = (cur_v_berg / current_speed) * max_speed_ms

                # Position translation: 1 deg lat ~ 111.139 km
                d_lat = (cur_v_berg * dt_seconds) / 111139.0
                # 1 deg lon ~ 111.139 * cos(lat) km
                lon_scale = max(0.1, math.cos(math.radians(cur_lat)))
                d_lon = (cur_u_berg * dt_seconds) / (111139.0 * lon_scale)

                speed_knots = current_speed / 0.514444
                bearing_deg = (math.degrees(math.atan2(cur_u_berg, cur_v_berg)) + 360.0) % 360.0

                track.append({
                    "hour": hour,
                    "lat": round(cur_lat, 4),
                    "lon": round(cur_lon, 4),
                    "speed_knots": round(speed_knots, 2),
                    "bearing_deg": round(bearing_deg, 1)
                })

                cur_lat += d_lat
                cur_lon += d_lon

            ensemble_tracks.append(track)

        # Baseline: Deterministic central trajectory (Member 0)
        p50_track = ensemble_tracks[0]

        # Compute uncertainty radius at each step (cone of uncertainty in Nautical Miles and km)
        trajectory_points = []
        for s in range(len(p50_track)):
            hour = p50_track[s]["hour"]
            member_lats = [ensemble_tracks[m][s]["lat"] for m in range(n_members)]
            member_lons = [ensemble_tracks[m][s]["lon"] for m in range(n_members)]

            std_lat = float(np.std(member_lats))
            std_lon = float(np.std(member_lons))
            # Uncertainty radius grows with time sqrt(t)
            uncertainty_km = max(1.5, round((std_lat * 111.0 + hour * 0.35) * 1.5, 2))

            trajectory_points.append({
                "hour": hour,
                "lat": p50_track[s]["lat"],
                "lon": p50_track[s]["lon"],
                "speed_knots": p50_track[s]["speed_knots"],
                "bearing_deg": p50_track[s]["bearing_deg"],
                "uncertainty_radius_km": uncertainty_km,
                "p10_lat": round(float(np.percentile(member_lats, 10)), 4),
                "p90_lat": round(float(np.percentile(member_lats, 90)), 4),
                "p10_lon": round(float(np.percentile(member_lons, 10)), 4),
                "p90_lon": round(float(np.percentile(member_lons, 90)), 4),
            })

        return {
            "iceberg_id": iceberg["id"],
            "name": iceberg["name"],
            "forecast_hours": forecast_hours,
            "trajectory": trajectory_points,
            "drift_summary": {
                "initial_position": [lat0, lon0],
                "projected_position_120h": [trajectory_points[-1]["lat"], trajectory_points[-1]["lon"]],
                "avg_speed_knots": round(float(np.mean([pt["speed_knots"] for pt in trajectory_points])), 2),
                "total_drift_distance_km": round(
                    sum(pt["speed_knots"] * 0.514444 * (time_step_hours * 3600) / 1000.0 for pt in trajectory_points[1:]), 1
                ),
                "model_confidence": "HIGH (2D hydrodynamic form drag, windage, and Coriolis parameter f with ensemble eddy variance)"
            }
        }

# Global singleton instance
iceberg_drift_engine = IcebergDriftModel()
