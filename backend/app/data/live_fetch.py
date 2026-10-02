"""
Live data acquisition for POLARIS-AI.

Builds a rolling NetCDF metocean store (same variables and grid as the frozen
January-2026 reference store) from keyless public sources:

  * Sea-ice concentration : NOAA@NSIDC G02135 (Sea Ice Index v4.0) daily GeoTIFFs
                            https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff/
  * 10 m wind + 2 m temp. : Open-Meteo forecast API (past_days + forecast_days),
                            https://api.open-meteo.com/v1/forecast  (CC BY 4.0,
                            free for non-commercial use; attribute Open-Meteo)

Honest limits (also written into the store's global attributes):
  * Ocean currents are still the analytic ACC / coastal-current PROXY, not CMEMS.
  * "sst" is still 2 m air temperature (proxy), exactly as in the reference store.
  * Winds are Open-Meteo best-match NWP analysis/forecast, not ERA5 reanalysis.
  * Observation time is stamped at 00:00 UTC of the NSIDC valid day, which is the
    conservative choice for the freshness fail-safe gate.

This module is torch-free so it can be tested and run without the ML stack.
All network access goes through an injectable ``fetch(url) -> bytes`` callable.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (grid matches build_real_antarctic_dataset.py exactly)
# --------------------------------------------------------------------------- #
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/geotiff"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
USER_AGENT = "POLARIS-AI/1.0 (+https://github.com/takenrohit/polaris-ai-antarctic-dss)"

GRID_LATS = np.linspace(-78.0, -54.0, 49)
GRID_LONS = np.linspace(-180.0, 180.0, 73)

SAMPLE_LATS = [-56.0, -62.0, -68.0, -72.0, -76.0]
SAMPLE_LONS = [-150.0, -100.0, -50.0, 0.0, 50.0, 90.0, 130.0, 170.0]

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MIN_OBS_DAYS = 14          # forecast_live needs a 14-day window ending at the baseline
MAX_GAP_FILL_DAYS = 3      # more missing NSIDC days than this -> refuse to build
RECHECK_RECENT_DAYS = 3    # re-download the newest days (NRT files can be revised)

Fetcher = Callable[[str], bytes]


# --------------------------------------------------------------------------- #
# Errors and HTTP
# --------------------------------------------------------------------------- #
class LiveFetchError(RuntimeError):
    """Network, parsing or validation failure while building the live store."""


class NotFound(LiveFetchError):
    """The remote file does not exist (HTTP 404) - e.g. not yet published."""


class DataUnavailable(LiveFetchError):
    """No usable observation found within the search window."""


def http_get(url: str, timeout: float = 30.0, retries: int = 3, backoff: float = 2.0) -> bytes:
    """GET with retries. 404 -> NotFound immediately; other failures are retried."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFound(f"404 Not Found: {url}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise LiveFetchError(f"GET failed after {retries} attempts: {url} ({last})")


# --------------------------------------------------------------------------- #
# NSIDC G02135
# --------------------------------------------------------------------------- #
def nsidc_url(d: date) -> str:
    return (f"{NSIDC_BASE}/{d.year}/{d.month:02d}_{_MONTHS[d.month - 1]}/"
            f"S_{d:%Y%m%d}_concentration_v4.0.tif")


def _looks_like_tiff(payload: bytes) -> bool:
    return len(payload) > 10_000 and payload[:4] in (b"II*\x00", b"MM\x00*")


def _fetch_nsidc_bytes(fetch: Fetcher, d: date) -> bytes:
    payload = fetch(nsidc_url(d))
    if not _looks_like_tiff(payload):
        # e.g. an HTML error page served with HTTP 200
        raise NotFound(f"NSIDC response for {d} is not a GeoTIFF ({len(payload)} bytes)")
    return payload


def find_latest_available(fetch: Fetcher, today: date, max_back: int = 14) -> Tuple[date, bytes]:
    """Walk back from ``today`` and return the newest published daily GeoTIFF."""
    for back in range(max_back + 1):
        d = today - timedelta(days=back)
        try:
            return d, _fetch_nsidc_bytes(fetch, d)
        except NotFound:
            continue
    raise DataUnavailable(
        f"No NSIDC G02135 daily GeoTIFF found in the last {max_back + 1} days before {today}")


def _cache_path(cache_dir: Path, d: date) -> Path:
    return cache_dir / f"S_{d:%Y%m%d}_concentration_v4.0.tif"


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def download_window(
    fetch: Fetcher,
    latest: date,
    latest_bytes: bytes,
    n_days: int,
    cache_dir: Path,
) -> Tuple[List[date], List[Path], List[date]]:
    """
    Ensure GeoTIFFs for the n_days ending at ``latest`` are cached.

    Returns (dates, paths, gap_filled_dates). A missing day is filled with the
    nearest earlier available file (or the nearest later one at the start of the
    window); more than MAX_GAP_FILL_DAYS gaps raises DataUnavailable.
    """
    dates = [latest - timedelta(days=n_days - 1 - i) for i in range(n_days)]
    found: Dict[date, Path] = {}
    _write_atomic(_cache_path(cache_dir, latest), latest_bytes)
    found[latest] = _cache_path(cache_dir, latest)

    for d in dates:
        if d in found:
            continue
        path = _cache_path(cache_dir, d)
        recent = (latest - d).days < RECHECK_RECENT_DAYS
        if path.exists() and not recent:
            found[d] = path
            continue
        try:
            _write_atomic(path, _fetch_nsidc_bytes(fetch, d))
            found[d] = path
        except NotFound:
            if path.exists():          # keep the older cached copy rather than lose the day
                found[d] = path

    missing = [d for d in dates if d not in found]
    if len(missing) > MAX_GAP_FILL_DAYS:
        raise DataUnavailable(
            f"{len(missing)} of {n_days} NSIDC days missing (limit {MAX_GAP_FILL_DAYS}): "
            f"{[m.isoformat() for m in missing]}")

    paths: List[Path] = []
    for i, d in enumerate(dates):
        if d in found:
            paths.append(found[d])
            continue
        prev = next((found[x] for x in reversed(dates[:i]) if x in found), None)
        nxt = next((found[x] for x in dates[i + 1:] if x in found), None)
        paths.append(prev or nxt)  # type: ignore[arg-type]
    return dates, paths, missing


def read_nsidc_sic(path: Path, lats: np.ndarray = GRID_LATS, lons: np.ndarray = GRID_LONS) -> np.ndarray:
    """
    Sample an NSIDC polar-stereographic GeoTIFF onto the (lat, lon) grid.

    Same convention as the reference-store builder: raw 0..1000 = concentration x10,
    anything else (land 2540, coast 2530, pole hole 2510, missing 2550) -> 0.0.
    Returns float32 (H, W) in [0, 1].
    """
    import rasterio
    from rasterio.warp import transform

    lons_2d, lats_2d = np.meshgrid(lons, lats)
    with rasterio.open(path) as src:
        xs, ys = transform("EPSG:4326", src.crs,
                           lons_2d.ravel().tolist(), lats_2d.ravel().tolist())
        raw = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=np.float32)
    ice = np.where((raw >= 0) & (raw <= 1000), raw / 1000.0, 0.0)
    return ice.reshape(len(lats), len(lons)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Open-Meteo winds / temperature
# --------------------------------------------------------------------------- #
def open_meteo_url(start: date, today: date, forecast_days: int = 10) -> str:
    coords = [(la, lo) for la in SAMPLE_LATS for lo in SAMPLE_LONS]
    past_days = min(92, max(1, (today - start).days + 1))
    params = {
        "latitude": ",".join(str(c[0]) for c in coords),
        "longitude": ",".join(str(c[1]) for c in coords),
        "daily": "wind_speed_10m_max,wind_direction_10m_dominant,temperature_2m_mean",
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
        "timezone": "GMT",
    }
    return OPEN_METEO_URL + "?" + urllib.parse.urlencode(params, safe=",")


def _fill_nan_along_time(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs along axis 1 of an (L, T) array."""
    out = arr.copy()
    for row in out:
        valid = np.isfinite(row)
        if not valid.any():
            raise LiveFetchError("Open-Meteo returned no valid values for a sample point")
        idx = np.where(valid, np.arange(len(row)), 0)
        np.maximum.accumulate(idx, out=idx)
        row[:] = row[idx]
        first = int(np.argmax(valid))
        row[:first] = row[first]
    return out


def parse_open_meteo(payload: bytes, dates: Sequence[date]) -> Dict[str, np.ndarray]:
    """
    Parse the multi-location response into arrays aligned to ``dates``.
    Returns dict with loc_lats (L,), loc_lons (L,), speed_kmh, dir_deg, temp_c (L, T).
    """
    data = json.loads(payload.decode("utf-8"))
    if isinstance(data, dict):
        if data.get("error"):
            raise LiveFetchError(f"Open-Meteo error: {data.get('reason')}")
        data = [data]
    if not data:
        raise LiveFetchError("Open-Meteo returned an empty response")

    want = [d.isoformat() for d in dates]
    n_loc = len(data)
    speed = np.full((n_loc, len(want)), np.nan, dtype=np.float64)
    wdir = np.full_like(speed, np.nan)
    temp = np.full_like(speed, np.nan)
    loc_lats = np.zeros(n_loc)
    loc_lons = np.zeros(n_loc)

    for i, item in enumerate(data):
        loc_lats[i], loc_lons[i] = item["latitude"], item["longitude"]
        daily = item["daily"]
        pos = {t: j for j, t in enumerate(daily["time"])}
        for k, key, dest in (("wind_speed_10m_max", "s", speed),
                             ("wind_direction_10m_dominant", "d", wdir),
                             ("temperature_2m_mean", "t", temp)):
            series = daily[k]
            for col, t in enumerate(want):
                j = pos.get(t)
                if j is not None and series[j] is not None:
                    dest[i, col] = float(series[j])

    return {
        "loc_lats": loc_lats, "loc_lons": loc_lons,
        "speed_kmh": _fill_nan_along_time(speed),
        "dir_deg": _fill_nan_along_time(wdir),
        "temp_c": _fill_nan_along_time(temp),
    }


def wind_components(speed_kmh: np.ndarray, dir_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Meteorological (direction wind blows FROM) -> u, v in m/s; same as the builder."""
    speed_ms = speed_kmh / 3.6
    rad = np.radians(dir_deg)
    return -speed_ms * np.sin(rad), -speed_ms * np.cos(rad)


def idw_to_grid(loc_lats: np.ndarray, loc_lons: np.ndarray, values: np.ndarray,
                lats: np.ndarray = GRID_LATS, lons: np.ndarray = GRID_LONS,
                power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting of (L, T) sample values -> (T, H, W) grid."""
    la, lo = np.meshgrid(lats, lons, indexing="ij")                    # (H, W)
    dlat = loc_lats[:, None, None] - la[None]
    dlon = (loc_lons[:, None, None] - lo[None] + 180.0) % 360.0 - 180.0
    dist = np.sqrt(dlat ** 2 + dlon ** 2) + 1e-4
    w = 1.0 / dist ** power
    w /= w.sum(axis=0, keepdims=True)
    return np.einsum("lhw,lt->thw", w, values).astype(np.float32)


def proxy_currents(lats: np.ndarray, u10: np.ndarray, v10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic ACC / coastal-current PROXY (identical to the reference-store builder)."""
    u_c = np.zeros_like(u10)
    v_c = np.zeros_like(v10)
    for i, lat in enumerate(lats):
        if lat > -65.0:
            u_c[:, i, :] = 0.28 + 0.02 * u10[:, i, :]
            v_c[:, i, :] = 0.06 + 0.02 * v10[:, i, :]
        else:
            u_c[:, i, :] = -0.16 + 0.02 * u10[:, i, :]
            v_c[:, i, :] = -0.04 + 0.02 * v10[:, i, :]
    return u_c, v_c


# --------------------------------------------------------------------------- #
# Store builder
# --------------------------------------------------------------------------- #
def build_live_store(
    out_path: Path,
    cache_dir: Path,
    n_obs_days: int = 21,
    n_forecast_days: int = 7,
    max_back: int = 14,
    today: Optional[date] = None,
    now: Optional[datetime] = None,
    fetch: Optional[Fetcher] = None,
) -> Dict[str, object]:
    """
    Fetch the latest data and write the rolling NetCDF store atomically.

    Time axis = n_obs_days observed days ending at the newest NSIDC file, followed by
    n_forecast_days of forecast wind/temperature (sic is NaN there). The reader and
    ``EnvironmentalDataProvider`` use the ``n_observed_days`` attribute to tell them apart.
    """
    import netCDF4 as nc

    if n_obs_days < MIN_OBS_DAYS:
        raise ValueError(f"n_obs_days must be >= {MIN_OBS_DAYS}")
    fetch = fetch or http_get
    now = now or datetime.now(timezone.utc)
    today = today or now.date()
    out_path, cache_dir = Path(out_path), Path(cache_dir)

    # 1. Sea ice
    latest, latest_bytes = find_latest_available(fetch, today, max_back)
    obs_dates, paths, gap_filled = download_window(fetch, latest, latest_bytes, n_obs_days, cache_dir)
    sic_obs = np.stack([read_nsidc_sic(p) for p in paths]).astype(np.float32)   # (T_obs, H, W)
    if not np.isfinite(sic_obs).all() or sic_obs.max() < 0.05:
        raise LiveFetchError("Sea-ice cube failed sanity check (non-finite or implausibly empty)")

    # 2. Winds / temperature (observed + forecast days)
    all_dates = obs_dates + [latest + timedelta(days=i + 1) for i in range(n_forecast_days)]
    wx = parse_open_meteo(fetch(open_meteo_url(obs_dates[0], today)), all_dates)
    u_loc, v_loc = wind_components(wx["speed_kmh"], wx["dir_deg"])
    u10 = idw_to_grid(wx["loc_lats"], wx["loc_lons"], u_loc)
    v10 = idw_to_grid(wx["loc_lats"], wx["loc_lons"], v_loc)
    sst = idw_to_grid(wx["loc_lats"], wx["loc_lons"], wx["temp_c"])
    u_curr, v_curr = proxy_currents(GRID_LATS, u10, v10)

    n_time = len(all_dates)
    sic = np.full((n_time, len(GRID_LATS), len(GRID_LONS)), np.nan, dtype=np.float32)
    sic[:n_obs_days] = sic_obs

    # 3. Write NetCDF (atomic)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    latest_iso = datetime(latest.year, latest.month, latest.day, tzinfo=timezone.utc) \
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    with nc.Dataset(str(tmp), "w", format="NETCDF4") as ds:
        ds.title = "POLARIS-AI live metocean store (rolling): NSIDC G02135 SIC + Open-Meteo wind"
        ds.conventions = "CF-1.8"
        ds.geospatial_bounds = "54.0S to 78.0S, 180.0W to 180.0E"
        ds.data_mode = "live"
        ds.n_observed_days = np.int32(n_obs_days)
        ds.n_forecast_days = np.int32(n_forecast_days)
        ds.latest_observation_date = latest.isoformat()
        ds.latest_observation_iso = latest_iso
        ds.observation_time_convention = "00:00 UTC of NSIDC valid day (conservative for freshness gate)"
        ds.fetched_at_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        ds.gap_filled_dates = ",".join(d.isoformat() for d in gap_filled)
        ds.sic_source = "NOAA@NSIDC G02135 v4.0 daily GeoTIFF (noaadata.apps.nsidc.org)"
        ds.wind_source = "Open-Meteo forecast API (best-match NWP; past_days + forecast_days)"
        ds.currents_source = "climatological_proxy (analytic ACC/coastal formula; NOT CMEMS)"
        ds.sst_source = "proxy: Open-Meteo 2 m air temperature (NOT satellite SST)"
        ds.history = "Created by app/data/live_fetch.py"

        ds.createDimension("time", n_time)
        ds.createDimension("latitude", len(GRID_LATS))
        ds.createDimension("longitude", len(GRID_LONS))

        v = ds.createVariable("time", "f4", ("time",))
        v.units = f"days since {obs_dates[0]:%Y-%m-%d} 00:00:00"
        v.long_name = "time"
        v[:] = np.arange(n_time, dtype=np.float32)

        v = ds.createVariable("latitude", "f4", ("latitude",))
        v.units, v.long_name = "degrees_north", "latitude"
        v[:] = GRID_LATS.astype(np.float32)
        v = ds.createVariable("longitude", "f4", ("longitude",))
        v.units, v.long_name = "degrees_east", "longitude"
        v[:] = GRID_LONS.astype(np.float32)

        for name, arr, units, long_name in (
            ("sic", sic, "1", "sea_ice_area_fraction"),
            ("u10", u10, "m s-1", "10m_eastward_wind"),
            ("v10", v10, "m s-1", "10m_northward_wind"),
            ("u_curr", u_curr, "m s-1", "surface_eastward_sea_water_velocity"),
            ("v_curr", v_curr, "m s-1", "surface_northward_sea_water_velocity"),
            ("sst", sst, "degC", "sea_surface_temperature"),
        ):
            var = ds.createVariable(name, "f4", ("time", "latitude", "longitude"),
                                    zlib=True, fill_value=np.float32(np.nan))
            var.units, var.long_name = units, long_name
            var[:] = arr
    os.replace(tmp, out_path)

    return {
        "path": str(out_path),
        "latest_observation_date": latest.isoformat(),
        "latest_observation_iso": latest_iso,
        "observed_days": n_obs_days,
        "forecast_wind_days": n_forecast_days,
        "gap_filled_dates": [d.isoformat() for d in gap_filled],
        "fetched_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_sic": round(float(sic_obs.max()), 3),
    }
