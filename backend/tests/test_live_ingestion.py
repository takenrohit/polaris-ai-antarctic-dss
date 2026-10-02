"""
Unit tests for live data ingestion:
- Real-time BYU/ASCAT satellite scatterometer iceberg feed
- Open-Meteo Antarctic in-situ & global metocean weather ingestion
- Dynamic live NAVAREA alerts & vessel weather enrichment
"""
import sys
import os
import pytest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.ingestion import environmental_data_provider
from app.services.iceberg_service import iceberg_service
from app.services.alert_service import alert_service
from app.services.vessel_service import vessel_service

def test_live_weather_ingestion():
    """Verify live weather fetching returns required metocean variables with offline fallback."""
    # Bharati coordinates
    weather = environmental_data_provider.get_live_weather(lat=-69.407, lon=76.187, timeout_s=4.0)
    assert "temperature_2m_c" in weather
    assert "wind_speed_knots" in weather
    assert "wind_direction_deg" in weather
    assert "surface_pressure_hpa" in weather
    assert "wave_height_m" in weather
    assert "data_source" in weather
    assert isinstance(weather["temperature_2m_c"], (int, float))

def test_live_byu_feed_sync():
    """Verify BYU scatterometer live satellite feed parses icebergs."""
    result = iceberg_service.sync_live_byu_feed(timeout_s=5.0)
    assert "status" in result
    assert "total_icebergs_tracked" in result
    # Check that icebergs in the service have valid coordinates and drift metadata
    bergs = iceberg_service.list_icebergs()
    assert len(bergs) >= 5
    for b in bergs:
        assert -90.0 <= b["lat"] <= -40.0

        assert -180.0 <= b["lon"] <= 180.0
        assert b["area_km2"] > 0

def test_vessel_weather_enrichment():
    """Verify vessel fleet positions are enriched with ambient weather."""
    vessels = vessel_service.list_vessels(enrich_live_weather=True)
    assert len(vessels) >= 3
    for v in vessels:
        pos = v["current_position"]
        assert "ambient_temp_c" in pos
        assert "wind_speed_knots" in pos
        assert "significant_wave_height_m" in pos

def test_dynamic_live_alerts():
    """Verify dynamic alert bulletins generate live UTC timestamps and metocean data."""
    alerts = alert_service.list_alerts()
    assert len(alerts) >= 3
    for a in alerts:
        assert a["is_live"] is True
        assert "timestamp" in a
        assert "coordinates" in a
        assert a["severity"] in ["CRITICAL", "HIGH", "WARNING", "INFO"]
