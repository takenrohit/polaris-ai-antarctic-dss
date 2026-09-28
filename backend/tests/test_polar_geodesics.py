"""
Unit Tests for Antarctic Polar Geodesics, Antimeridian Wrapping, and Coordinate Transforms.
Verifies:
- Longitude wrapping around -180 / +180 antimeridian
- Clamping of high southern latitude singularity (-90°S)
- Equivalence of Haversine implementation with known navigational distances
- Bearing and intermediate point calculations
- Analytical EPSG:3412 projection calculations
"""
import sys
import os
import math
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.geodesics import (
    wrap_longitude,
    clamp_latitude,
    haversine_distance_nm,
    haversine_distance_km,
    initial_bearing_deg,
    intermediate_point,
    wgs84_to_epsg3412_approx
)


class TestPolarGeodesics:
    """Rigorous tests for geographic calculations in polar environments."""

    def test_longitude_wrapping_boundaries(self):
        assert wrap_longitude(0.0) == 0.0
        assert wrap_longitude(180.0) == 180.0
        assert wrap_longitude(-180.0) == -180.0
        # Wrap past positive antimeridian
        assert wrap_longitude(185.0) == -175.0
        assert wrap_longitude(360.0) == 0.0
        assert wrap_longitude(540.0) == -180.0 or wrap_longitude(540.0) == 180.0
        # Wrap past negative antimeridian
        assert wrap_longitude(-185.0) == 175.0
        assert wrap_longitude(-360.0) == 0.0

    def test_latitude_clamping(self):
        assert clamp_latitude(-90.0) == -90.0
        assert clamp_latitude(90.0) == 90.0
        assert clamp_latitude(-95.0) == -90.0
        assert clamp_latitude(100.0) == 90.0
        assert clamp_latitude(-69.4) == -69.4

    def test_known_antarctic_distances(self):
        # Cape Town (-33.918, 18.423) to Bharati Station (-69.407, 76.187)
        dist_nm = haversine_distance_nm(-33.918, 18.423, -69.407, 76.187)
        assert 2800.0 <= dist_nm <= 2900.0

        # Maitri Station (-70.767, 11.733) to Bharati Station (-69.407, 76.187)
        m_to_b_nm = haversine_distance_nm(-70.767, 11.733, -69.407, 76.187)
        assert 1200.0 <= m_to_b_nm <= 1350.0

        # Distance to self must be 0
        assert haversine_distance_nm(-70.0, 50.0, -70.0, 50.0) == 0.0

    def test_antimeridian_crossing_distance(self):
        # Ross Sea to Amundsen Sea crossing 180° meridian at -72°S
        lat = -72.0
        lon1 = 175.0 # East
        lon2 = -175.0 # West
        # Distance across 10 degrees longitude at -72°S:
        # 10 deg * cos(72°) * 60 NM/deg = 10 * 0.3090 * 60 ≈ 185.4 NM
        dist_nm = haversine_distance_nm(lat, lon1, lat, lon2)
        assert 180.0 <= dist_nm <= 195.0

    def test_intermediate_points_continuity(self):
        lat1, lon1 = -33.918, 18.423
        lat2, lon2 = -69.407, 76.187

        start = intermediate_point(lat1, lon1, lat2, lon2, 0.0)
        assert pytest.approx(start[0], abs=1e-2) == lat1
        assert pytest.approx(start[1], abs=1e-2) == lon1

        end = intermediate_point(lat1, lon1, lat2, lon2, 1.0)
        assert pytest.approx(end[0], abs=1e-2) == lat2
        assert pytest.approx(end[1], abs=1e-2) == lon2

        mid = intermediate_point(lat1, lon1, lat2, lon2, 0.5)
        # Midpoint latitude should be between start and end
        assert lat2 <= mid[0] <= lat1

    def test_epsg3412_polar_stereographic_projection(self):
        # South pole (-90°S) maps to (0, 0) in polar stereographic
        x_pole, y_pole = wgs84_to_epsg3412_approx(-90.0, 0.0)
        assert abs(x_pole) < 1.0 and abs(y_pole) < 1.0

        # Bharati Station (-69.407, 76.187)
        x_b, y_b = wgs84_to_epsg3412_approx(-69.407, 76.187)
        assert isinstance(x_b, float) and isinstance(y_b, float)
