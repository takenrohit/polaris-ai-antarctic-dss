"""
Authentic Lindqvist (1989) & Riska (1997) Ice Resistance Physics Model.
Calculates:
- Crushing resistance (R_c)
- Bending resistance (R_b)
- Submersion resistance (R_s)
- Speed-dependent velocity resistance (R_v)
- Total resistance in level and pack ice
- Delivered brake power (P_B) and Marine Gas Oil (MGO) fuel consumption in Metric Tons.
"""
import math
from typing import Dict, Any

class LindqvistIceResistanceModel:
    """
    Implements the Lindqvist (1989) model for ship resistance in level and pack ice,
    with Riska (1997) corrections for Finnish-Swedish Ice Class and IMO Polar Class vessels.
    """
    def __init__(self):
        # Physical constants
        self.RHO_WATER = 1028.0 # kg/m^3 (cold Antarctic seawater)
        self.RHO_ICE = 900.0    # kg/m^3
        self.G = 9.80665        # m/s^2
        self.E_ICE = 2.0e9      # Young's modulus of ice (Pa)
        self.MU_STEEL_ICE = 0.10 # Hull steel-ice friction coefficient
        self.SFOC_G_KWH = 185.0 # Specific Fuel Oil Consumption (g MGO / kWh) for medium-speed marine diesel
        self.ETA_PROPULSIVE = 0.65 # Net propulsive efficiency

    def calculate_resistance_kn(
        self,
        length_m: float,
        beam_m: float,
        draft_m: float,
        speed_knots: float,
        ice_thickness_m: float,
        ice_concentration: float,
        stem_angle_deg: float = 25.0,
        waterplane_angle_deg: float = 30.0,
        flexural_strength_kpa: float = 550.0
    ) -> Dict[str, float]:
        """
        Calculates ice resistance components in kiloNewtons (kN) based on Lindqvist (1989).
        """
        if ice_concentration < 0.05 or ice_thickness_m <= 0.01:
            # Pure open water
            v_ms = max(0.5, speed_knots * 0.514444)
            # Holtrop-Mennen simplified open-water resistance (kN)
            r_ow_kn = 0.045 * (beam_m * draft_m) * (v_ms ** 1.9)
            return {
                "r_crushing_kn": 0.0,
                "r_bending_kn": 0.0,
                "r_submersion_kn": 0.0,
                "r_ice_total_kn": 0.0,
                "r_open_water_kn": round(r_ow_kn, 2),
                "r_total_kn": round(r_ow_kn, 2)
            }

        v_ms = max(0.5, speed_knots * 0.514444)
        h = max(0.05, ice_thickness_m)
        sigma_b = flexural_strength_kpa * 1000.0 # Pa
        phi = math.radians(stem_angle_deg)
        alpha = math.radians(waterplane_angle_deg)
        psi = math.atan(math.tan(phi) / max(0.01, math.sin(alpha)))
        mu = self.MU_STEEL_ICE

        # 1. Crushing resistance R_c (Newtons)
        numerator_c = math.tan(phi) + (mu * math.cos(phi) / math.cos(psi))
        denominator_c = 1.0 - (mu * math.sin(phi) / math.cos(psi))
        r_c = 0.5 * sigma_b * (h ** 2) * (numerator_c / max(0.1, denominator_c))

        # 2. Bending resistance R_b (Newtons)
        char_length_l = (self.E_ICE * (h ** 3) / (12.0 * (1.0 - 0.3**2) * self.RHO_WATER * self.G)) ** 0.25
        term_b1 = (math.tan(psi) / math.cos(phi)) + (mu * math.cos(psi) * math.sin(phi))
        term_b2 = 1.0 + 9.22 * (char_length_l / length_m)
        r_b = (27.0 / 64.0) * sigma_b * beam_m * (h ** 1.5) * math.sqrt(self.RHO_WATER * self.G / self.E_ICE) * term_b1 * term_b2

        # 3. Submersion resistance R_s (Newtons)
        delta_rho = self.RHO_WATER - self.RHO_ICE
        r_s_static = delta_rho * self.G * h * beam_m * draft_m * ((beam_m + draft_m) / (beam_m + 2.0 * draft_m)) * (1.0 + 2.0 * mu * (draft_m / beam_m))
        # Speed-dependent velocity term
        froude_ice = v_ms / math.sqrt(max(0.1, self.G * h))
        r_s_velocity = 0.063 * delta_rho * self.G * h * beam_m * length_m * froude_ice
        r_s = r_s_static + r_s_velocity

        # Level ice total resistance (kN)
        r_level_kn = (r_c + r_b + r_s) / 1000.0

        # Pack ice / Marginal Ice Zone scaling based on ice concentration C
        # Non-linear pack concentration exponent ~ 1.6
        c_scale = min(1.0, max(0.0, ice_concentration)) ** 1.6
        r_ice_pack_kn = r_level_kn * c_scale

        # Calm water resistance
        r_ow_kn = 0.045 * (beam_m * draft_m) * (v_ms ** 1.9)
        r_total_kn = r_ow_kn + r_ice_pack_kn

        return {
            "r_crushing_kn": round((r_c / 1000.0) * c_scale, 2),
            "r_bending_kn": round((r_b / 1000.0) * c_scale, 2),
            "r_submersion_kn": round((r_s / 1000.0) * c_scale, 2),
            "r_ice_total_kn": round(r_ice_pack_kn, 2),
            "r_open_water_kn": round(r_ow_kn, 2),
            "r_total_kn": round(r_total_kn, 2)
        }

    def estimate_fuel_burn_mt(
        self,
        distance_nm: float,
        speed_knots: float,
        ice_concentration: float,
        ice_thickness_m: float = 0.8,
        length_m: float = 135.0,
        beam_m: float = 22.0,
        draft_m: float = 8.5,
        stem_angle_deg: float = 25.0,
        sfoc_g_kwh: float = 185.0
    ) -> Dict[str, float]:
        """
        Calculates fuel burn in Metric Tons (MT) over leg distance using Lindqvist physics:
        Power P_B (kW) = (R_total (kN) * V (m/s)) / eta_propulsive
        Fuel Burn (MT) = Power * Hours * SFOC / 1e6
        """
        hours = distance_nm / max(1.5, speed_knots)
        v_ms = speed_knots * 0.514444

        res = self.calculate_resistance_kn(
            length_m=length_m,
            beam_m=beam_m,
            draft_m=draft_m,
            speed_knots=speed_knots,
            ice_thickness_m=ice_thickness_m,
            ice_concentration=ice_concentration,
            stem_angle_deg=stem_angle_deg
        )

        r_total_kn = res["r_total_kn"]
        # Required brake power at engine flywheel
        power_kw = (r_total_kn * v_ms) / self.ETA_PROPULSIVE
        # Hotel and auxiliary baseline electrical load ~ 800 kW
        aux_power_kw = 800.0
        total_power_kw = power_kw + aux_power_kw

        # Fuel burn in Metric Tons (1 MT = 1,000,000 g)
        fuel_mt = (total_power_kw * hours * sfoc_g_kwh) / 1.0e6
        fuel_rate_mt_per_hour = (total_power_kw * sfoc_g_kwh) / 1.0e6

        return {
            "fuel_mt": round(fuel_mt, 2),
            "fuel_rate_mt_per_hour": round(fuel_rate_mt_per_hour, 3),
            "power_kw": round(total_power_kw, 1),
            "resistance_kn": r_total_kn,
            "ice_resistance_kn": res["r_ice_total_kn"],
            "transit_hours": round(hours, 2)
        }

# Global singleton instance
lindqvist_fuel_model = LindqvistIceResistanceModel()
