"""
Operational (live) sea-ice forecast math - numpy/scipy only, no torch.

The existing ``SeaIcePredictor.forecast`` is a *hindcast*: it always starts from a fixed
day and scores against later observed days. Here the baseline is the **newest observed
day** and there is no ground truth, so no skill score is computed.

Method (identical blend to the validated hybrid forecaster):

    pred(tau) = (1 - alpha(tau)) * persistence
              + alpha(tau) * (wind-advected + melt-trend state)
              + 0.02 * nn_residual(tau)

alpha(tau) = min(cap, base * (tau-1)**exp) is re-fitted on every call by grid search on
the older part of the rolling window (baseline = day 6, targets = days 7..13), i.e. only
on data that is strictly earlier than the forecast origin.

Skill caveat: the hybrid was only shown to be comparable to persistence on one January
window (see EVALUATION.md). Nothing here changes that; live mode makes the forecast
current, not better.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

ALPHA_CAPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
ALPHA_BASES = (0.002, 0.005, 0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.035, 0.045, 0.060)
ALPHA_EXPS = (0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5)
DEFAULT_ALPHA = (0.35, 0.018, 1.5)
NN_WEIGHT = 0.02
MIN_WINDOW_DAYS = 14        # 14-day window: calibration + melt trend (matches the hindcast)

# channel order in the gridded sequence: [SIC, SST, U10, V10, current_speed]
_SIC, _U10, _V10 = 0, 2, 3


def _drift_per_day(lat_grid: np.ndarray, u10: np.ndarray, v10: np.ndarray):
    """Free-drift velocity in degrees/day (same constants as the hindcast forecaster)."""
    dlat = (0.012 * v10 * 86.4) / 111.0
    dlon = (0.012 * u10 * 86.4) / (111.0 * np.cos(np.radians(lat_grid))[:, None] + 1e-6)
    return dlat, dlon


def _grid_steps(lat_grid: np.ndarray, lon_grid: np.ndarray) -> Tuple[float, float]:
    d_lat = (lat_grid[-1] - lat_grid[0]) / max(1, len(lat_grid) - 1)
    d_lon = (lon_grid[-1] - lon_grid[0]) / max(1, len(lon_grid) - 1)
    return float(d_lat), float(d_lon)


def _blend(day0: np.ndarray, melt_field: np.ndarray, dlat: np.ndarray, dlon: np.ndarray,
           tau: int, alpha: float, d_lat: float, d_lon: float,
           nn_residual: float = 0.0) -> np.ndarray:
    """One lead of the hybrid: persistence blended with wind-advected + melt-trend state."""
    H, W = day0.shape
    yy, xx = np.mgrid[0:H, 0:W]
    sy = (dlat * tau) / (d_lat + 1e-9)
    sx = (dlon * tau) / (d_lon + 1e-9)
    advected = map_coordinates(day0, np.array([yy - sy, xx - sx]), order=1, mode="nearest")
    thermo = np.clip(advected + melt_field * tau * 0.7, 0.0, 1.0)
    return np.clip((1.0 - alpha) * day0 + alpha * thermo + NN_WEIGHT * nn_residual, 0.0, 1.0)


def fit_alpha_params(window: np.ndarray, lat_grid: np.ndarray, lon_grid: np.ndarray
                     ) -> Tuple[Tuple[float, float, float], float]:
    """
    Grid-search (cap, base, exp) across widened parameter bounds over multiple baseline days.

    ``window`` is (T, 5, H, W) with T >= 14: fits over multiple antecedent baseline days
    (e.g. days 4, 5, 6) projecting 7-day targets against ground truth within the 14-day history.
    Precomputes advected fields to achieve fast evaluation across the widened search grid.
    Returns (params, summed_rmse_over_targets).
    """
    if window.shape[0] < 14:
        return DEFAULT_ALPHA, float("nan")
    d_lat, d_lon = _grid_steps(lat_grid, lon_grid)
    H, W = window.shape[2], window.shape[3]
    yy, xx = np.mgrid[0:H, 0:W]

    # Precompute advected & thermo states across multiple baseline days (days 4, 5, 6)
    # to eliminate single-day noise and stabilize horizon parameter calibration.
    precomputed = []
    baseline_days = (4, 5, 6)
    for b in baseline_days:
        day0 = window[b, _SIC]
        truth = window[b + 1 : b + 8, _SIC]
        melt = (window[b, _SIC] - window[0, _SIC]) / float(b)
        dlat, dlon = _drift_per_day(lat_grid, window[b, _U10], window[b, _V10])
        leads_data = []
        for t in range(7):
            tau = t + 1
            sy = (dlat * tau) / (d_lat + 1e-9)
            sx = (dlon * tau) / (d_lon + 1e-9)
            advected = map_coordinates(day0, np.array([yy - sy, xx - sx]), order=1, mode="nearest")
            thermo = np.clip(advected + melt * tau * 0.7, 0.0, 1.0)
            leads_data.append((day0, thermo, truth[t]))
        precomputed.append(leads_data)

    best, best_params = float("inf"), DEFAULT_ALPHA
    for cap in ALPHA_CAPS:
        for base in ALPHA_BASES:
            for exp in ALPHA_EXPS:
                total = 0.0
                for b_idx in range(len(precomputed)):
                    for t in range(7):
                        tau = t + 1
                        alpha = min(cap, base * max(0.0, (tau - 1) ** exp))
                        day0, thermo, tr = precomputed[b_idx][t]
                        pred = (1.0 - alpha) * day0 + alpha * thermo
                        total += float(np.sqrt(np.mean((pred - tr) ** 2)))
                if total < best:
                    best, best_params = total, (cap, base, exp)
    return best_params, best


def hybrid_forecast_from_latest(
    window: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    days_ahead: int = 7,
    nn_deltas: Optional[np.ndarray] = None,
    alpha_params: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, object]:
    """
    Forecast ``days_ahead`` days from the last day of ``window`` (T, 5, H, W).

    Returns dict(model (D,H,W), persistence (D,H,W), alpha_params, calibration_rmse, baseline).
    """
    if window.ndim != 4 or window.shape[0] < MIN_WINDOW_DAYS:
        raise ValueError(f"window must be (T>={MIN_WINDOW_DAYS}, 5, H, W); got {window.shape}")

    params, cal = (alpha_params, float("nan")) if alpha_params else fit_alpha_params(window, lat_grid, lon_grid)
    cap, base, exp = params

    day0 = window[-1, _SIC]
    melt_field = (window[-1, _SIC] - window[-14, _SIC]) / 13.0       # last 14 days, as in hindcast
    dlat, dlon = _drift_per_day(lat_grid, window[-1, _U10], window[-1, _V10])
    d_lat, d_lon = _grid_steps(lat_grid, lon_grid)

    preds = []
    for t in range(days_ahead):
        tau = t + 1
        alpha = min(cap, base * max(0.0, (tau - 1) ** exp))
        nn = 0.0 if nn_deltas is None or t >= len(nn_deltas) else nn_deltas[t]
        preds.append(_blend(day0, melt_field, dlat, dlon, tau, alpha, d_lat, d_lon, nn))

    return {
        "model": np.array(preds, dtype=np.float32),
        "persistence": np.tile(day0[None], (days_ahead, 1, 1)).astype(np.float32),
        "alpha_params": tuple(float(x) for x in params),
        "calibration_rmse_sum": cal,
        "baseline": day0.astype(np.float32),
    }
