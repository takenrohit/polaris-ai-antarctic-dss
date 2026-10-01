"""
Robustness, Input Validation, and Safety Assurance Tests.
Verifies:
- Detection and rejection of land / ice shelf collisions
- Handling of missing data / sensor dropouts with fail-safe degradation
- Enforcement of physical vessel engineering constraints
- Generation of the 'DO NOT USE FOR NAVIGATION' state under critical failure
"""
import sys
import os
from datetime import datetime, timezone, timedelta
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.validators import VesselConfiguration, assess_data_quality
from app.data.ingestion import environmental_data_provider, AntarcticCoastlineMask
from app.models.route_optimizer import polar_route_optimizer


class TestVesselValidation:
    """Verifies physical validation of vessel particulars."""

    def test_valid_vessel_config(self):
        vessel = VesselConfiguration(
            name="MV Vasiliy Golovnin",
            ice_class="PC5",
            length_m=163.0,
            beam_m=22.4,
            draft_m=9.0,
            engine_power_kw=12800.0
        )
        assert vessel.ice_class == "PC5"
        assert vessel.length_m == 163.0

    def test_invalid_ice_class_rejection(self):
        with pytest.raises(ValueError, match="Invalid vessel ice class"):
            VesselConfiguration(ice_class="INVALID_CLASS")

    def test_beam_exceeds_length_rejection(self):
        with pytest.raises(ValueError, match="must be strictly less than length"):
            VesselConfiguration(length_m=30.0, beam_m=35.0)

    def test_draft_exceeds_beam_rejection(self):
        with pytest.raises(ValueError, match="cannot exceed vessel beam"):
            VesselConfiguration(beam_m=10.0, draft_m=12.0)

    def test_negative_or_extreme_power_rejection(self):
        with pytest.raises(ValueError, match="Invalid engine power"):
            VesselConfiguration(engine_power_kw=-500.0)


class TestSafetyAndCollisionGates:
    """Verifies land / shelf collision avoidance and fail-safe behaviors."""

    def test_continental_interior_collision_detection(self):
        mask = AntarcticCoastlineMask()
        # South Pole
        assert mask.is_land_or_shelf(-90.0, 0.0) is True
        # Queen Maud Land interior
        assert mask.is_land_or_shelf(-78.0, 15.0) is True
        # Amery Ice Shelf interior
        assert mask.is_land_or_shelf(-71.5, 71.0) is True

    def test_missing_data_quality_assessment(self):
        # Scenario 1: Normal operational data (< 24h latency)
        recent_iso = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        q_normal = assess_data_quality(
            observation_iso=recent_iso,
            has_sic=True,
            has_wind=True,
            has_currents=True
        )
        assert q_normal["quality_status"] == "OPERATIONAL"
        assert q_normal["fail_safe_gate_tripped"] is False
        assert q_normal["is_safe_for_decision_support"] is True

        # Scenario 2: Critical sea-ice dropout -> DO NOT USE FOR NAVIGATION
        q_critical = assess_data_quality(has_sic=False)
        assert q_critical["quality_status"] == "DO_NOT_USE_FOR_NAVIGATION"
        assert q_critical["is_safe_for_decision_support"] is False
        assert q_critical["fail_safe_gate_tripped"] is True

        # Scenario 3: Stale data (> 48h) trips fail-safe gate -> DO_NOT_USE_FOR_NAVIGATION
        q_stale = assess_data_quality(
            observation_iso="2024-01-01T00:00:00Z",
            max_latency_hours=48.0
        )
        assert q_stale["quality_status"] == "DO_NOT_USE_FOR_NAVIGATION"
        assert q_stale["fail_safe_gate_tripped"] is True

    def test_route_optimizer_rejection_explanation(self):
        # Request route from Cape Town to Bharati
        res = polar_route_optimizer.find_pareto_routes(
            origin_lat=-33.918,
            origin_lon=18.423,
            dest_lat=-69.407,
            dest_lon=76.187,
            vessel_ice_class="PC5"
        )
        assert "decision_brief" in res
        # Must include clear compliance explanation and iceberg warnings
        brief = res["decision_brief"]
        assert "summary" in brief
        assert "ice_risk_alert" in brief
        assert "polaris_compliance" in brief
