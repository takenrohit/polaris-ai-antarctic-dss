"""
Offline tests for the live-data pipeline (no network, no torch).

Synthetic NSIDC polar-stereographic GeoTIFFs and a fake Open-Meteo response are served
through the injectable ``fetch`` callable of ``live_fetch``.
"""
import io
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.validators import assess_data_quality
from app.data import live_fetch as lf
from app.data.ingestion import DEFAULT_NC_PATH, EnvironmentalDataProvider
from app.models import live_forecast as lfc

H, W = len(lf.GRID_LATS), len(lf.GRID_LONS)


# --------------------------------------------------------------------------- #
# Synthetic sources
# --------------------------------------------------------------------------- #
def make_tif_bytes(ring_scale: float = 1.0) -> bytes:
    """NSIDC-like south grid (316 x 332, 25 km, EPSG:3412): ice ring, land core, uint16."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    width, height = 316, 332
    transform = from_origin(-3950000.0, 4350000.0, 25000.0, 25000.0)
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x = -3950000.0 + (cols + 0.5) * 25000.0
    y = 4350000.0 - (rows + 0.5) * 25000.0
    r_km = np.hypot(x, y) / 1000.0
    data = np.zeros((height, width), dtype=np.uint16)
    data[(r_km > 1500) & (r_km < 2400 * ring_scale)] = 1000      # 100 % ice ring
    data[r_km <= 1300] = 2540                                     # land flag
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=height, width=width, count=1,
                      dtype="uint16", crs="EPSG:3412", transform=transform) as dst:
            dst.write(data, 1)
        return mem.read()


class FakeNet:
    """Serves NSIDC files for ``available`` dates and a synthetic Open-Meteo payload."""

    def __init__(self, available, today, tif=None, fail_weather=False):
        self.available = set(available)
        self.today = today
        self.tif = tif or make_tif_bytes()
        self.fail_weather = fail_weather
        self.calls = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url.startswith(lf.NSIDC_BASE):
            stamp = url.rsplit("S_", 1)[1][:8]
            d = datetime.strptime(stamp, "%Y%m%d").date()
            if d in self.available:
                return self.tif
            raise lf.NotFound(url)
        if url.startswith(lf.OPEN_METEO_URL):
            if self.fail_weather:
                raise lf.LiveFetchError("weather down")
            return self._weather(url)
        raise AssertionError(f"unexpected URL {url}")

    def _weather(self, url: str) -> bytes:
        q = parse_qs(urlparse(url).query)
        lats = [float(v) for v in q["latitude"][0].split(",")]
        lons = [float(v) for v in q["longitude"][0].split(",")]
        past, fut = int(q["past_days"][0]), int(q["forecast_days"][0])
        days = [self.today - timedelta(days=past - i) for i in range(past + fut)]
        out = []
        for i, (la, lo) in enumerate(zip(lats, lons)):
            n = len(days)
            speed = [30.0 + (i % 5) + 0.5 * k for k in range(n)]       # km/h
            out.append({
                "latitude": la, "longitude": lo,
                "daily": {
                    "time": [d.isoformat() for d in days],
                    "wind_speed_10m_max": speed,
                    "wind_direction_10m_dominant": [270.0] * n,        # westerly
                    "temperature_2m_mean": [-5.0 + 0.1 * k for k in range(n)],
                },
            })
        return json.dumps(out).encode()


def _days(end: date, n: int):
    return [end - timedelta(days=i) for i in range(n)]


@pytest.fixture(scope="module")
def tif_bytes():
    return make_tif_bytes()


def build(tmp_path, available, today, **kw):
    net = FakeNet(available, today, kw.pop("tif", None), kw.pop("fail_weather", False))
    out = tmp_path / "live.nc"
    summary = lf.build_live_store(out, tmp_path / "cache", today=today,
                                  now=datetime(today.year, today.month, today.day, 12, tzinfo=timezone.utc),
                                  fetch=net, **kw)
    return out, summary, net


# --------------------------------------------------------------------------- #
# NSIDC
# --------------------------------------------------------------------------- #
def test_nsidc_url_matches_published_layout():
    assert lf.nsidc_url(date(2026, 9, 24)) == (
        "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/"
        "2026/09_Sep/S_20260924_concentration_v4.0.tif")


def test_find_latest_walks_back_over_unpublished_days(tif_bytes):
    today = date(2026, 10, 2)
    net = FakeNet(_days(date(2026, 9, 30), 30), today, tif_bytes)
    latest, payload = lf.find_latest_available(net, today, max_back=14)
    assert latest == date(2026, 9, 30) and payload == tif_bytes


def test_find_latest_raises_when_nothing_published(tif_bytes):
    net = FakeNet([], date(2026, 10, 2), tif_bytes)
    with pytest.raises(lf.DataUnavailable):
        lf.find_latest_available(net, date(2026, 10, 2), max_back=5)


def test_html_error_page_is_not_accepted_as_geotiff():
    def fetch(url):
        return b"<html>" + b"x" * 20000 + b"</html>"
    with pytest.raises(lf.DataUnavailable):
        lf.find_latest_available(fetch, date(2026, 10, 2), max_back=1)


def test_read_nsidc_sic_values(tmp_path, tif_bytes):
    p = tmp_path / "a.tif"
    p.write_bytes(tif_bytes)
    sic = lf.read_nsidc_sic(p)
    assert sic.shape == (H, W) and sic.dtype == np.float32
    assert 0.0 <= sic.min() and sic.max() <= 1.0
    i72 = int(np.argmin(np.abs(lf.GRID_LATS - (-72.0))))          # r ~ 2000 km -> inside ring
    i56 = int(np.argmin(np.abs(lf.GRID_LATS - (-56.0))))          # r ~ 3770 km -> open water
    assert sic[i72].mean() > 0.9
    assert sic[i56].max() == 0.0


# --------------------------------------------------------------------------- #
# Store build + provider
# --------------------------------------------------------------------------- #
def test_build_store_schema_and_provider_live_mode(tmp_path):
    today = date(2026, 10, 2)
    out, summary, _ = build(tmp_path, _days(today, 40), today)
    assert summary["latest_observation_date"] == "2026-10-02"
    assert summary["gap_filled_dates"] == []

    import xarray as xr
    ds = xr.open_dataset(out)
    assert dict(ds.sizes) == {"time": 28, "latitude": H, "longitude": W}
    for v in ("sic", "u10", "v10", "u_curr", "v_curr", "sst"):
        assert v in ds.data_vars
    assert np.isfinite(ds["sic"].values[:21]).all()
    assert np.isnan(ds["sic"].values[21:]).all()               # forecast days: wind only
    assert np.isfinite(ds["u10"].values).all()
    assert ds.attrs["data_mode"] == "live" and int(ds.attrs["n_observed_days"]) == 21
    assert "NOT CMEMS" in ds.attrs["currents_source"]
    u10 = ds["u10"].values
    ds.close()

    prov = EnvironmentalDataProvider(nc_path=str(out))
    assert prov.mode == "live" and prov.n_obs == 21 and prov.n_time == 28
    assert prov.latest_observation_iso() == "2026-10-02T00:00:00Z"

    # hour_offset 0 -> newest observed day; +48 h -> two forecast-wind days later
    lat, lon = -62.0, -50.0
    i = int(np.argmin(np.abs(lf.GRID_LATS - lat)))
    j = int(np.argmin(np.abs(lf.GRID_LONS - lon)))
    assert prov.get_wind(lat, lon, 0)[0] == pytest.approx(float(u10[20, i, j]), abs=1e-5)
    assert prov.get_wind(lat, lon, 48)[0] == pytest.approx(float(u10[22, i, j]), abs=1e-5)
    assert prov.get_wind(lat, lon, 24 * 365)[0] == pytest.approx(float(u10[27, i, j]), abs=1e-5)

    seq = prov.get_gridded_sequence(lf.GRID_LATS, lf.GRID_LONS, num_days=21)
    assert seq.shape == (21, 5, H, W) and np.isfinite(seq).all()

    meta = prov.metadata()
    assert meta["is_live"] and meta["observed_days"] == 21 and meta["forecast_wind_days"] == 7


def test_live_store_is_fresh_for_the_gate(tmp_path):
    today = datetime.now(timezone.utc).date()
    out, _, _ = build(tmp_path, _days(today, 40), today)
    prov = EnvironmentalDataProvider(nc_path=str(out))
    q = assess_data_quality(observation_iso=prov.latest_observation_iso())
    assert q["quality_status"] == "OPERATIONAL" and q["fail_safe_gate_tripped"] is False
    assert q["data_freshness_hours"] < 48.0


def test_snapshot_store_is_honestly_stale():
    prov = EnvironmentalDataProvider(nc_path=str(DEFAULT_NC_PATH))
    assert prov.mode == "snapshot"
    assert prov.latest_observation_iso() == "2026-01-21T00:00:00Z"
    q = assess_data_quality(observation_iso=prov.latest_observation_iso())
    assert q["quality_status"] == "DO_NOT_USE_FOR_NAVIGATION"
    assert q["fail_safe_gate_tripped"] is True


def test_snapshot_indexing_unchanged():
    """Legacy semantics: wind index = hour_offset // 24 from day 0, clamped at day 20."""
    prov = EnvironmentalDataProvider(nc_path=str(DEFAULT_NC_PATH))
    u = prov._reader.arrays["u10"]
    i = int(np.argmin(np.abs(prov._reader.lats - (-68.5))))
    j = int(np.argmin(np.abs(prov._reader.lons - 75.0)))
    assert prov.get_wind(-68.5, 75.0, 0)[0] == pytest.approx(float(u[0, i, j]), abs=0.5)
    assert prov._wind_time_index(24 * 400) == 20 and prov._wind_time_index(48) == 2


# --------------------------------------------------------------------------- #
# Gaps and failure handling
# --------------------------------------------------------------------------- #
def test_single_missing_day_is_gap_filled_and_recorded(tmp_path):
    today = date(2026, 10, 2)
    missing = today - timedelta(days=9)
    avail = [d for d in _days(today, 40) if d != missing]
    out, summary, _ = build(tmp_path, avail, today)
    assert summary["gap_filled_dates"] == [missing.isoformat()]


def test_too_many_gaps_refuses_to_build(tmp_path):
    today = date(2026, 10, 2)
    avail = [d for d in _days(today, 40) if not (today - timedelta(days=15)) <= d <= (today - timedelta(days=8))]
    with pytest.raises(lf.DataUnavailable):
        build(tmp_path, avail, today)
    assert not (tmp_path / "live.nc").exists()


def test_failed_build_keeps_previous_store_and_leaves_no_tmp(tmp_path):
    today = date(2026, 10, 2)
    out = tmp_path / "live.nc"
    out.write_bytes(b"previous-good-store")
    net = FakeNet(_days(today, 40), today, fail_weather=True)
    with pytest.raises(lf.LiveFetchError):
        lf.build_live_store(out, tmp_path / "cache", today=today, fetch=net)
    assert out.read_bytes() == b"previous-good-store"
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_ice_cube_fails_sanity_check(tmp_path):
    today = date(2026, 10, 2)
    with pytest.raises(lf.LiveFetchError):
        build(tmp_path, _days(today, 40), today, tif=make_tif_bytes(ring_scale=0.01))


def test_open_meteo_parse_handles_single_dict_and_missing_values():
    d0 = date(2026, 10, 1)
    dates = [d0 + timedelta(days=i) for i in range(4)]
    payload = json.dumps({
        "latitude": -60.0, "longitude": 0.0,
        "daily": {"time": [d.isoformat() for d in dates],
                  "wind_speed_10m_max": [None, 36.0, 40.0, None],
                  "wind_direction_10m_dominant": [270.0] * 4,
                  "temperature_2m_mean": [-3.0, None, -4.0, -5.0]},
    }).encode()
    r = lf.parse_open_meteo(payload, dates)
    assert r["speed_kmh"].shape == (1, 4)
    assert np.isfinite(r["speed_kmh"]).all() and r["speed_kmh"][0, 0] == 36.0   # back-filled
    assert r["speed_kmh"][0, 3] == 40.0                                          # forward-filled
    u, v = lf.wind_components(np.array([[36.0]]), np.array([[270.0]]))
    assert u[0, 0] == pytest.approx(10.0, abs=1e-6) and v[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_open_meteo_error_payload_raises():
    with pytest.raises(lf.LiveFetchError):
        lf.parse_open_meteo(json.dumps({"error": True, "reason": "limit"}).encode(), [date(2026, 10, 1)])


# --------------------------------------------------------------------------- #
# Freshness gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, "", "not-a-date"])
def test_gate_trips_when_observation_time_unknown(bad):
    q = assess_data_quality(observation_iso=bad)
    assert q["quality_status"] == "DO_NOT_USE_FOR_NAVIGATION"
    assert q["data_freshness_hours"] is None and q["data_freshness_indicator"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# Live forecast math
# --------------------------------------------------------------------------- #
def _window(T=21, wind=(0.0, 0.0)):
    rng = np.random.default_rng(0)
    sic = np.clip(rng.random((H, W)) * 0.3, 0, 1)
    seq = np.zeros((T, 5, H, W), dtype=np.float32)
    for t in range(T):
        seq[t, 0] = np.clip(sic + 0.002 * t, 0, 1)
        seq[t, 2], seq[t, 3] = wind
    return seq


def test_live_forecast_day1_is_persistence_and_bounded():
    seq = _window(wind=(8.0, -3.0))
    r = lfc.hybrid_forecast_from_latest(seq, lf.GRID_LATS, lf.GRID_LONS, days_ahead=7)
    assert r["model"].shape == (7, H, W)
    assert 0.0 <= r["model"].min() and r["model"].max() <= 1.0
    np.testing.assert_allclose(r["model"][0], r["persistence"][0], atol=1e-6)   # alpha(1) = 0
    cap, base, exp = r["alpha_params"]
    assert cap in lfc.ALPHA_CAPS and base in lfc.ALPHA_BASES and exp in lfc.ALPHA_EXPS
    np.testing.assert_array_equal(r["baseline"], seq[-1, 0])                    # origin = newest day


def test_live_forecast_rejects_short_window():
    with pytest.raises(ValueError):
        lfc.hybrid_forecast_from_latest(_window(T=10), lf.GRID_LATS, lf.GRID_LONS)


def test_live_forecast_uses_latest_day_not_start_of_window():
    seq = _window()
    seq[-1, 0] = np.clip(seq[-1, 0] + 0.2, 0, 1)
    r = lfc.hybrid_forecast_from_latest(seq, lf.GRID_LATS, lf.GRID_LONS, days_ahead=3)
    np.testing.assert_allclose(r["persistence"][2], seq[-1, 0], atol=1e-6)


def test_live_math_reproduces_committed_january_hindcast():
    """
    Regression check: run the live forecaster on days 0-13 of the frozen snapshot and compare
    against the committed evaluation. Persistence is deterministic (must match exactly);
    the hybrid differs only by the 2 % neural-residual term, so allow a small tolerance.
    """
    metrics_path = Path(__file__).resolve().parents[2] / "evaluation" / "results" / "metrics.json"
    committed = json.loads(metrics_path.read_text())["sea_ice_forecasting"]["lead_time_metrics"]

    prov = EnvironmentalDataProvider(nc_path=str(DEFAULT_NC_PATH))
    # same regional grid the committed evaluation uses (evaluation/run_evaluation.py)
    lats, lons = np.linspace(-78.0, -56.0, 30), np.linspace(-60.0, 90.0, 45)
    seq = prov.get_gridded_sequence(lats, lons, num_days=21)
    r = lfc.hybrid_forecast_from_latest(seq[:14], lats, lons, days_ahead=7)
    truth = seq[14:21, 0]

    for t in range(7):
        persist = float(np.sqrt(np.mean((r["persistence"][t] - truth[t]) ** 2)))
        hybrid = float(np.sqrt(np.mean((r["model"][t] - truth[t]) ** 2)))
        assert persist == pytest.approx(committed[t]["persistence_rmse"], abs=2e-4)
        assert hybrid == pytest.approx(committed[t]["hybrid_rmse"], abs=2.5e-3)


# --------------------------------------------------------------------------- #
# API: ingestion status + guarded refresh (uses the app's global provider)
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_ingestion_status_reports_real_age_and_gate(client):
    r = client.get("/api/forecast/ingestion-status").json()
    assert r["data_mode"] in ("live", "snapshot")
    feed = r["primary_feed"]
    assert feed["latency_hours"] is not None and feed["latest_observation_iso"]
    assert r["sentinel1_sar"]["status"] == "NOT_INGESTED_STUB"        # no invented latency
    assert "latency_hours" not in r["sentinel1_sar"]
    if r["data_mode"] == "snapshot":                                   # frozen Jan-2026 data
        assert r["status"] == "DO_NOT_USE_FOR_NAVIGATION" and r["fail_safe_gate_tripped"] is True
        assert feed["status"] == "FROZEN_SNAPSHOT"


def test_refresh_disabled_without_token(client, monkeypatch):
    monkeypatch.delenv("POLARIS_REFRESH_TOKEN", raising=False)
    assert client.post("/api/forecast/refresh").status_code == 403


def test_refresh_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv("POLARIS_REFRESH_TOKEN", "s3cret")
    assert client.post("/api/forecast/refresh").status_code == 401
    assert client.post("/api/forecast/refresh", headers={"X-Refresh-Token": "nope"}).status_code == 401


def test_refresh_failure_keeps_previous_store(client, monkeypatch):
    from app.api import routes_forecast as rf
    monkeypatch.setenv("POLARIS_REFRESH_TOKEN", "s3cret")
    before = rf.environmental_data_provider.metadata()

    def boom(*a, **k):
        raise lf.DataUnavailable("nsidc down")
    monkeypatch.setattr(lf, "build_live_store", boom)
    r = client.post("/api/forecast/refresh", headers={"X-Refresh-Token": "s3cret"})
    assert r.status_code == 502 and "previous store kept" in r.json()["detail"]
    assert rf.environmental_data_provider.metadata() == before


def test_refresh_success_reloads_provider_and_clears_cache(client, monkeypatch, tmp_path):
    from app.api import routes_forecast as rf
    today = datetime.now(timezone.utc).date()
    out = tmp_path / "live.nc"
    net = FakeNet(_days(today, 40), today)
    real_build = lf.build_live_store

    def fake_build(path, cache, **k):
        return real_build(out, tmp_path / "cache", fetch=net, today=today)
    monkeypatch.setattr(lf, "build_live_store", fake_build)
    monkeypatch.setattr(rf, "LIVE_NC_PATH", out)
    monkeypatch.setenv("POLARIS_REFRESH_TOKEN", "s3cret")

    prov = rf.environmental_data_provider
    original_path, original_state = prov.nc_path, (prov._reader, prov.mode, prov.n_obs, prov.base_idx, prov.n_time)
    try:
        r = client.post("/api/forecast/refresh", headers={"X-Refresh-Token": "s3cret"})
        assert r.status_code == 200 and r.json()["refreshed"] is True
        assert prov.mode == "live" and prov.nc_path == out
        assert prov.data_age_hours() < 48.0
    finally:  # restore the global provider for other tests
        prov.nc_path = original_path
        prov._reader, prov.mode, prov.n_obs, prov.base_idx, prov.n_time = original_state
