"""
Polar Geodesics and Coordinate Transformation Utilities for Antarctic Waters.
Handles:
- EPSG:3412 (Antarctic Polar Stereographic South, WGS84) to EPSG:4326 (WGS84 Lat/Lon) transformations
- Strict longitude wrapping [-180, 180] and antimeridian crossing handling
- High-latitude polar singularity stability (approaching South Pole -90°)
- Admissible spherical/ellipsoidal geodesic distance calculations
"""
import math
from typing import Tuple, List, Dict, Any


R_EARTH_KM = 6371.0088
R_EARTH_NM = 3440.065


def wrap_longitude(lon: float) -> float:
    """
    Normalizes longitude to the standard range [-180.0, 180.0] degrees.
    Handles multiple 360-degree wraps cleanly.
    """
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    # Avoid -180.0 vs +180.0 edge anomaly
    if wrapped == -180.0 and lon > 0:
        return 180.0
    return round(wrapped, 6)


def clamp_latitude(lat: float) -> float:
    """Clamps latitude to valid geographic range [-90.0, 90.0] degrees."""
    return max(-90.0, min(90.0, lat))


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance in Nautical Miles using high-precision Haversine formula.
    Robust against identical points, longitude wrap-around, and polar singularities.
    """
    lat1, lat2 = clamp_latitude(lat1), clamp_latitude(lat2)
    lon1, lon2 = wrap_longitude(lon1), wrap_longitude(lon2)

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R_EARTH_NM * c


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    return haversine_distance_nm(lat1, lon1, lat2, lon2) * 1.852


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates initial true compass bearing in degrees [0, 360) from point 1 to point 2.
    """
    lat1, lat2 = clamp_latitude(lat1), clamp_latitude(lat2)
    lon1, lon2 = wrap_longitude(lon1), wrap_longitude(lon2)

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return round(bearing, 2)


def intermediate_point(lat1: float, lon1: float, lat2: float, lon2: float, fraction: float) -> Tuple[float, float]:
    """
    Calculates great-circle intermediate waypoint along transit track at fraction f in [0, 1].
    """
    fraction = max(0.0, min(1.0, fraction))
    if fraction == 0.0:
        return (lat1, wrap_longitude(lon1))
    if fraction == 1.0:
        return (lat2, wrap_longitude(lon2))

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lambda1, lambda2 = math.radians(lon1), math.radians(lon2)

    d = 2.0 * math.asin(math.sqrt(
        math.sin((phi2 - phi1) / 2.0)**2 +
        math.cos(phi1) * math.cos(phi2) * math.sin((lambda2 - lambda1) / 2.0)**2
    ))

    if d < 1e-9:
        return (round(lat1, 4), round(wrap_longitude(lon1), 4))

    a = math.sin((1.0 - fraction) * d) / math.sin(d)
    b = math.sin(fraction * d) / math.sin(d)

    x = a * math.cos(phi1) * math.cos(lambda1) + b * math.cos(phi2) * math.cos(lambda2)
    y = a * math.cos(phi1) * math.sin(lambda1) + b * math.cos(phi2) * math.sin(lambda2)
    z = a * math.sin(phi1) + b * math.sin(phi2)

    res_phi = math.atan2(z, math.sqrt(x**2 + y**2))
    res_lambda = math.atan2(y, x)

    return (round(math.degrees(res_phi), 4), round(wrap_longitude(math.degrees(res_lambda)), 4))


def wgs84_to_epsg3412_approx(lat: float, lon: float) -> Tuple[float, float]:
    """
    Analytical projection from WGS84 (Lat, Lon) to Antarctic Polar Stereographic South (EPSG:3412).
    Standard parallel: -71.0°S, Central meridian: 0.0°E, True scale at 71°S.
    Returns (x_meters, y_meters).
    """
    lat = clamp_latitude(lat)
    if lat > -50.0:
        # Polar stereographic is not defined/appropriate outside polar regions
        pass
    phi = math.radians(abs(lat))
    lam = math.radians(wrap_longitude(lon))
    phi_std = math.radians(71.0)

    # Standard conformal polar stereographic radius
    k0 = math.cos(phi_std) * (1.0 + math.sin(phi_std))
    t = math.tan(math.pi / 4.0 - phi / 2.0)
    rho = 2.0 * 6378137.0 * t / (1.0 + math.sin(phi_std))

    x = rho * math.sin(lam)
    y = -rho * math.cos(lam)
    return (round(x, 2), round(y, 2))
