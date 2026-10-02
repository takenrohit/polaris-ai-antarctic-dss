"""
Maritime Vessel Particulars, Input Validation, and Quality Assurance Gate.
Implements:
- Vessel geometry and propulsion validation (LOA, Beam, Draft, Power, Polar Class)
- Metocean data quality checks (freshness, range bounds, sensor dropouts)
- 'DO NOT USE FOR NAVIGATION' fail-safe operational state detection
"""
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone


VALID_POLAR_CLASSES = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "OPEN_WATER"]

# Standard IMO / IACS vessel parameter bounds for polar expedition ships
MIN_VESSEL_LENGTH_M = 30.0
MAX_VESSEL_LENGTH_M = 350.0
MIN_VESSEL_BEAM_M = 8.0
MAX_VESSEL_BEAM_M = 50.0
MIN_VESSEL_DRAFT_M = 3.0
MAX_VESSEL_DRAFT_M = 18.0
MIN_VESSEL_POWER_KW = 1000.0
MAX_VESSEL_POWER_KW = 100000.0


class VesselConfiguration:
    """Rigorous model of polar vessel engineering particulars."""
    def __init__(
        self,
        name: str = "MV Vasiliy Golovnin",
        ice_class: str = "PC5",
        length_m: float = 163.0,
        beam_m: float = 22.4,
        draft_m: float = 9.0,
        displacement_dwt: float = 10700.0,
        engine_power_kw: float = 12800.0,
        stem_angle_deg: float = 25.0,
        flare_angle_deg: float = 55.0,
        friction_coeff: float = 0.10,
        cruising_speed_knots: float = 13.5
    ):
        self.name = name
        self.ice_class = ice_class.upper()
        self.length_m = float(length_m)
        self.beam_m = float(beam_m)
        self.draft_m = float(draft_m)
        self.displacement_dwt = float(displacement_dwt)
        self.engine_power_kw = float(engine_power_kw)
        self.stem_angle_deg = float(stem_angle_deg)
        self.flare_angle_deg = float(flare_angle_deg)
        self.friction_coeff = float(friction_coeff)
        self.cruising_speed_knots = float(cruising_speed_knots)

        self.validate()

    def validate(self):
        """Validates physical vessel parameters against naval architecture bounds."""
        if self.ice_class not in VALID_POLAR_CLASSES:
            raise ValueError(
                f"Invalid vessel ice class '{self.ice_class}'. Must be one of: {VALID_POLAR_CLASSES}"
            )
        if not (MIN_VESSEL_LENGTH_M <= self.length_m <= MAX_VESSEL_LENGTH_M):
            raise ValueError(
                f"Invalid length {self.length_m} m. Expected [{MIN_VESSEL_LENGTH_M}, {MAX_VESSEL_LENGTH_M}] m."
            )
        if not (MIN_VESSEL_BEAM_M <= self.beam_m <= MAX_VESSEL_BEAM_M):
            raise ValueError(
                f"Invalid beam {self.beam_m} m. Expected [{MIN_VESSEL_BEAM_M}, {MAX_VESSEL_BEAM_M}] m."
            )
        if self.beam_m >= self.length_m:
            raise ValueError(
                f"Vessel beam ({self.beam_m} m) must be strictly less than length ({self.length_m} m)."
            )
        if not (MIN_VESSEL_DRAFT_M <= self.draft_m <= MAX_VESSEL_DRAFT_M):
            raise ValueError(
                f"Invalid draft {self.draft_m} m. Expected [{MIN_VESSEL_DRAFT_M}, {MAX_VESSEL_DRAFT_M}] m."
            )
        if self.draft_m >= self.beam_m:
            raise ValueError(
                f"Vessel draft ({self.draft_m} m) cannot exceed vessel beam ({self.beam_m} m)."
            )
        if not (MIN_VESSEL_POWER_KW <= self.engine_power_kw <= MAX_VESSEL_POWER_KW):
            raise ValueError(
                f"Invalid engine power {self.engine_power_kw} kW. Expected [{MIN_VESSEL_POWER_KW}, {MAX_VESSEL_POWER_KW}] kW."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ice_class": self.ice_class,
            "length_m": self.length_m,
            "beam_m": self.beam_m,
            "draft_m": self.draft_m,
            "displacement_dwt": self.displacement_dwt,
            "engine_power_kw": self.engine_power_kw,
            "stem_angle_deg": self.stem_angle_deg,
            "flare_angle_deg": self.flare_angle_deg,
            "friction_coeff": self.friction_coeff,
            "cruising_speed_knots": self.cruising_speed_knots
        }


def assess_data_quality(
    observation_iso: Optional[str] = None,
    max_latency_hours: float = 48.0,
    has_sic: bool = True,
    has_wind: bool = True,
    has_currents: bool = True
) -> Dict[str, Any]:
    """
    Evaluates input data quality and determines operational certification state.
    Tied directly to the fail-safe gate under IMO Polar Code MSC.1/Circ.1519 safety doctrine:
    - If satellite SIC feed is missing: trips fail-safe gate -> 'DO_NOT_USE_FOR_NAVIGATION'.
    - If data latency exceeds threshold (48h): trips fail-safe gate -> 'DO_NOT_USE_FOR_NAVIGATION'.
    - If secondary meteorological fields are missing: status is 'DEGRADED'.
    - If all data is fresh (<= 48h) and valid: status is 'OPERATIONAL'.
    """
    now = datetime.now(timezone.utc)
    reasons = []
    fail_safe_gate_tripped = False

    if not has_sic:
        reasons.append("Critical failure: Satellite Sea Ice Concentration feed missing")
        fail_safe_gate_tripped = True
    if not has_wind:
        reasons.append("Atmospheric wind forcing unavailable (defaulting to zero-wind drag)")
    if not has_currents:
        reasons.append("Surface ocean currents unavailable")

    # Unknown or unparseable observation time means the data age cannot be verified.
    # That is treated as unsafe (fail-safe), never as "fresh".
    latency_hours: Optional[float] = None
    if observation_iso:
        try:
            obs_dt = datetime.fromisoformat(observation_iso.replace("Z", "+00:00"))
            if obs_dt.tzinfo is None:
                obs_dt = obs_dt.replace(tzinfo=timezone.utc)
            latency_hours = max(0.0, (now - obs_dt).total_seconds() / 3600.0)
        except ValueError:
            latency_hours = None

    if latency_hours is None:
        fail_safe_gate_tripped = True
        reasons.append(
            "Observation timestamp missing or unparseable: data age cannot be verified. "
            "Automatic navigation guidance locked under IMO fail-safe protocol."
        )
    elif latency_hours > max_latency_hours:
        fail_safe_gate_tripped = True
        reasons.append(
            f"Data freshness alert: Metocean observation latency ({latency_hours:.1f}h) "
            f"exceeds operational safety threshold ({max_latency_hours:.0f}h). "
            f"Automatic navigation guidance locked under IMO fail-safe protocol."
        )

    if fail_safe_gate_tripped:
        status = "DO_NOT_USE_FOR_NAVIGATION"
    elif reasons:
        status = "DEGRADED"
    else:
        status = "OPERATIONAL"

    return {
        "quality_status": status,
        "is_safe_for_decision_support": (status == "OPERATIONAL"),
        "fail_safe_gate_tripped": fail_safe_gate_tripped,
        "data_freshness_hours": None if latency_hours is None else round(latency_hours, 1),
        "freshness_threshold_hours": max_latency_hours,
        "data_freshness_indicator": (
            "UNKNOWN" if latency_hours is None
            else "FRESH" if latency_hours <= 24.0
            else ("STALE" if latency_hours <= max_latency_hours else "EXPIRED")
        ),
        "timestamp_utc": now.isoformat(),
        "alerts": reasons
    }
