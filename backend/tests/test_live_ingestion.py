"""
Unit and integration tests for live data ingestion and fail-safes:
- Mocked Open-Meteo live metocean weather responses (asserting is_live: True)
- Offline fallback path (asserting is_live: False, nulls for unmeasured constants)
- Mocked BYU/ASCAT live table HTML parsing (asserting is_live: True, no invented dimensions, exact ID matching)
- Rule-triggered advisories (illustrative scenario disclaimer, correct is_live status)
- Non-blocking vessel telemetry loops and caching
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest.mock as mock
import urllib.error
import pytest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.ingestion import environmental_data_provider
from app.services.iceberg_service import iceberg_service
from app.services.alert_service import alert_service
from app.services.vessel_service import vessel_service


SAMPLE_BYU_HTML = """
<!DOCTYPE html>
<html>
<head><title>Current Icebergs</title></head>
<body>
<table>
<tr><th>Iceberg</th><th>Longitude</th><th>Latitude</th><th>DOY</th></tr>
<tr><td>b29</td><td>123 45 W</td><td>65 30 S</td><td>250</td></tr>
<tr><td>a76a</td><td>054 12 W</td><td>62 10 S</td><td>251</td></tr>
<tr><td>z99</td><td>020 00 E</td><td>66 00 S</td><td>252</td></tr>
</table>
</body>
</html>
"""

SAMPLE_OPEN_METEO_JSON = {
    "latitude": -69.4,
    "longitude": 76.2,
    "current": {
        "time": "2026-10-02T12:00",
        "temperature_2m": -11.5,
        "wind_speed_10m": 12.3,
        "wind_direction_10m": 145.0,
        "surface_pressure": 988.4,
        "relative_humidity_2m": 82.0
    }
}

SAMPLE_MARINE_JSON = {
    "latitude": -69.4,
    "longitude": 76.2,
    "current": {
        "wave_height": 2.1,
        "wave_direction": 180.0,
        "wave_period": 7.5
    }
}


def test_mocked_open_meteo_live_weather_success():
    """Verify live weather returns is_live=True and exact parsed values when HTTP succeeds."""
    def mock_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "marine" in url:
            body = json.dumps(SAMPLE_MARINE_JSON).encode("utf-8")
        else:
            body = json.dumps(SAMPLE_OPEN_METEO_JSON).encode("utf-8")
        return io.BytesIO(body)

    # Invalidate cache
    if hasattr(environmental_data_provider, "_live_weather_cache"):
        environmental_data_provider._live_weather_cache.clear()

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        weather = environmental_data_provider.get_live_weather(lat=-69.407, lon=76.187, timeout_s=3.0)

    assert weather["is_live"] is True
    assert weather["temperature_2m_c"] == -11.5
    assert weather["temperature_c"] == -11.5
    assert weather["wind_speed_ms"] == 12.3
    assert weather["wind_speed_knots"] == round(12.3 * 1.94384, 1)
    assert weather["surface_pressure_hpa"] == 988.4
    assert weather["relative_humidity_pct"] == 82.0
    assert weather["wave_height_m"] == 2.1
    assert "Open-Meteo" in weather["data_source"]


def test_open_meteo_offline_fallback_null_constants():
    """Verify offline fallback returns is_live=False, newest index data, and nulls for unmeasured constants."""
    if hasattr(environmental_data_provider, "_live_weather_cache"):
        environmental_data_provider._live_weather_cache.clear()

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Simulated Network Down")):
        weather = environmental_data_provider.get_live_weather(lat=-69.407, lon=76.187, timeout_s=1.0)

    assert weather["is_live"] is False
    assert "offline fallback" in weather["data_source"].lower()
    # Unmeasured parameters must be None (never invented 985 hPa or 75% humidity)
    assert weather["surface_pressure_hpa"] is None
    assert weather["relative_humidity_pct"] is None
    assert weather["wave_height_m"] is None
    # Wind and SST from NetCDF are present
    assert isinstance(weather["wind_speed_knots"], (int, float))
    assert isinstance(weather["temperature_c"], (int, float))


def test_mocked_byu_live_feed_parsing_and_no_invented_constants():
    """Verify BYU table parser sets is_live=True and leaves unmeasured attributes as null/unknown."""
    def mock_urlopen(req, timeout=None):
        return io.BytesIO(SAMPLE_BYU_HTML.encode("utf-8"))

    # Seed B-29a in archive to test exact matching
    iceberg_service.icebergs["B-29a"] = {
        "id": "B-29a",
        "name": "Iceberg B-29a",
        "calving_source": "Amundsen Shelf",
        "lat": -70.0,
        "lon": -110.0,
        "area_km2": 320.0,
        "is_live": False
    }

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = iceberg_service.sync_live_byu_feed(timeout_s=3.0)

    assert res["status"] == "LIVE_FEED_SYNCED"
    assert res["total_icebergs_tracked"] >= 3

    # 1. Verify B-29a was NOT overwritten by row 'b29'
    b29a = iceberg_service.icebergs.get("B-29a")
    assert b29a is not None
    # B-29a original lat must remain untouched
    assert b29a["lat"] == -70.0

    # 2. Check newly registered berg 'Z-99'
    z99 = iceberg_service.icebergs.get("Z-99")
    assert z99 is not None
    assert z99["is_live"] is True
    assert z99["observation_doy"] == "252"
    # Positions must be parsed
    assert z99["lat"] == -66.0
    assert z99["lon"] == 20.0
    # Crucial: NO invented constants!
    assert z99["area_km2"] is None
    assert z99["length_km"] is None
    assert z99["width_km"] is None
    assert z99["mass_gt"] is None
    assert z99["drift_speed_knots"] is None
    assert z99["drift_bearing_deg"] is None
    assert z99["hazard_level"] == "UNASSESSED"
    assert z99["geometry_measured"] is False


def test_rule_triggered_alerts_disclaimer_and_provenance():
    """Verify alerts fire only when thresholds are met, display 'no data' instead of defaults, and reflect provenance."""
    # Test offline alert generation
    if hasattr(environmental_data_provider, "_live_weather_cache"):
        environmental_data_provider._live_weather_cache.clear()

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Offline")):
        alerts = alert_service.list_alerts(force_refresh=True)

    # In offline store, Bharati meets freezing threshold (-12.8°C <= -10°C) and A-23a is tracked
    # Maitri wind is 18.1 kts (< 22 kts threshold) so it does NOT fire without meeting threshold
    alert_ids = [a["id"] for a in alerts]
    assert "ALERT-POLARIS-APPROACH-BHARATI" in alert_ids
    assert any(aid.startswith("ALERT-POLARIS-BERG-") for aid in alert_ids)
    assert "ALERT-POLARIS-WIND-MAITRI" not in alert_ids  # Below 22 kts threshold!

    for a in alerts:
        # Must clearly state illustrative scenario advisory
        assert a["is_official_bulletin"] is False
        assert "illustrative scenario alert" in a["source"].lower()
        assert "disclaimer" in a
        assert "not an official NAVAREA/WMO bulletin" in a["disclaimer"]

        # When offline, is_live must be False and text must NOT say "Live metocean"
        if a["id"] in ["ALERT-POLARIS-WIND-MAITRI", "ALERT-POLARIS-APPROACH-BHARATI"]:
            assert a["is_live"] is False
            assert "Offline metocean reference store" in a["description"]
            assert "Live metocean feed" not in a["description"]

        if "A-23a" in a["id"]:
            # A-23a advisory must clarify it is from historical archive
            assert "not currently listed on the active BYU" in a["description"]

    # Test that wind alert FIRES when threshold is met (e.g. wind >= 22 kts)
    with mock.patch.object(environmental_data_provider, "get_live_weather") as mock_wx:
        mock_wx.side_effect = lambda lat, lon, **kw: {
            "is_live": True,
            "data_source": "Open-Meteo",
            "temperature_2m_c": None,  # Test missing temperature -> must show 'no data'
            "wind_speed_knots": 28.5 if lat < -70.0 else 10.0,
            "sea_ice_concentration_pct": None,
        }
        alerts_high_wind = alert_service.list_alerts(force_refresh=True)
        m_alert = next((a for a in alerts_high_wind if a["id"] == "ALERT-POLARIS-WIND-MAITRI"), None)
        assert m_alert is not None
        assert m_alert["severity"] == "WARNING"
        assert "Temp no data" in m_alert["description"]
        assert m_alert["trigger_metrics"]["temperature_c"] == "no data"
        assert m_alert["trigger_metrics"]["wind_speed_knots"] == 28.5


def test_vessel_telemetry_caching_and_nonblocking():
    """Verify list_vessels(enrich_live_weather=False) makes zero network calls and runs synchronously."""
    with mock.patch("urllib.request.urlopen") as mock_http:
        vessels = vessel_service.list_vessels(enrich_live_weather=False)
        assert len(vessels) == 3
        # Must not have attempted any HTTP call
        mock_http.assert_not_called()
