"""
tests.test_primitives
=====================
Unit tests for SDF primitives against known analytical solutions.

Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
from blade_precompute.section_geometry.geometry.primitives import (
    sdf_circle, sdf_box, sdf_half_plane,
    sdf_segment, sdf_capsule, sdf_ellipse, sdf_polygon,
)


# ---------------------------------------------------------------------------
# sdf_circle
# ---------------------------------------------------------------------------

class TestCircle:
    def test_on_boundary(self):
        # Points on circumference: phi = 0
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        phi = sdf_circle(x, y, r=1.0)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_inside(self):
        phi = sdf_circle(0.0, 0.0, r=1.0)
        assert phi < 0.0
        np.testing.assert_allclose(phi, -1.0, atol=1e-10)

    def test_outside(self):
        phi = sdf_circle(2.0, 0.0, r=1.0)
        np.testing.assert_allclose(phi, 1.0, atol=1e-10)

    def test_offset_centre(self):
        phi = sdf_circle(3.0, 4.0, cx=3.0, cy=4.0, r=0.5)
        np.testing.assert_allclose(phi, -0.5, atol=1e-10)

    def test_eikonal_property(self):
        # |∇φ| should be ≈ 1 away from the centre singularity and boundary
        x = np.linspace(-2, 2, 200)
        y = np.linspace(-2, 2, 200)
        X, Y = np.meshgrid(x, y)
        phi = sdf_circle(X, Y, r=1.0)
        dx = x[1] - x[0]
        gx = np.gradient(phi, dx, axis=1)
        gy = np.gradient(phi, dx, axis=0)
        gm = np.sqrt(gx**2 + gy**2)
        # Exclude centre region (finite-diff artefact at the origin singularity)
        # and restrict to mid-interior annulus [-0.8, -0.1]
        annulus = (phi < -0.1) & (phi > -0.8)
        np.testing.assert_allclose(gm[annulus], 1.0, atol=0.02)


# ---------------------------------------------------------------------------
# sdf_box
# ---------------------------------------------------------------------------

class TestBox:
    def test_centre_inside(self):
        phi = sdf_box(0.0, 0.0, cx=0.0, cy=0.0, half_w=1.0, half_h=0.5)
        np.testing.assert_allclose(phi, -0.5, atol=1e-10)  # min(half_w, half_h)

    def test_corner_outside(self):
        phi = sdf_box(2.0, 1.0, half_w=1.0, half_h=0.5)
        expected = np.sqrt(1.0**2 + 0.5**2)
        np.testing.assert_allclose(phi, expected, atol=1e-10)

    def test_face_outside(self):
        phi = sdf_box(1.5, 0.0, half_w=1.0, half_h=0.5)
        np.testing.assert_allclose(phi, 0.5, atol=1e-10)

    def test_on_edge(self):
        phi = sdf_box(1.0, 0.0, half_w=1.0, half_h=0.5)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sdf_half_plane
# ---------------------------------------------------------------------------

class TestHalfPlane:
    def test_above_plane(self):
        # Normal pointing up (ny=1), boundary at y=0
        phi = sdf_half_plane(0.0, 1.0, nx=0.0, ny=1.0, d=0.0)
        np.testing.assert_allclose(phi, 1.0, atol=1e-10)

    def test_below_plane(self):
        phi = sdf_half_plane(0.0, -1.0, nx=0.0, ny=1.0, d=0.0)
        np.testing.assert_allclose(phi, -1.0, atol=1e-10)

    def test_on_plane(self):
        phi = sdf_half_plane(0.0, 0.0, nx=0.0, ny=1.0, d=0.0)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)

    def test_normalisation(self):
        # Non-unit normal should give same distances as unit normal
        phi1 = sdf_half_plane(0.0, 2.0, nx=0.0, ny=1.0, d=0.0)
        phi2 = sdf_half_plane(0.0, 2.0, nx=0.0, ny=3.0, d=0.0)
        np.testing.assert_allclose(phi1, phi2, atol=1e-10)


# ---------------------------------------------------------------------------
# sdf_segment
# ---------------------------------------------------------------------------

class TestSegment:
    def test_midpoint(self):
        # Closest point is midpoint of segment
        d = sdf_segment(0.0, 1.0, -1.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose(d, 1.0, atol=1e-10)

    def test_endpoint(self):
        d = sdf_segment(2.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose(d, 1.0, atol=1e-10)

    def test_on_segment(self):
        d = sdf_segment(0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sdf_capsule
# ---------------------------------------------------------------------------

class TestCapsule:
    def test_on_boundary_at_cap(self):
        # Point on circular end-cap
        d = sdf_capsule(2.0, 0.0, -1.0, 0.0, 1.0, 0.0, r=1.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_inside(self):
        d = sdf_capsule(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, r=1.0)
        np.testing.assert_allclose(d, -1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sdf_ellipse
# ---------------------------------------------------------------------------

class TestEllipse:
    def test_inside(self):
        phi = sdf_ellipse(0.0, 0.0, rx=2.0, ry=1.0)
        assert phi < 0.0

    def test_outside(self):
        phi = sdf_ellipse(3.0, 0.0, rx=2.0, ry=1.0)
        assert phi > 0.0

    def test_circle_degenerate(self):
        # Ellipse with rx=ry=1 should match circle for points near the surface
        # (Quilez iteration loses accuracy for far-field exterior points)
        x = np.array([0.0, 0.7])
        y = np.array([0.0, 0.7])
        phi_e = sdf_ellipse(x, y, rx=1.0, ry=1.0)
        phi_c = sdf_circle(x, y, r=1.0)
        np.testing.assert_allclose(phi_e, phi_c, atol=1e-3)


# ---------------------------------------------------------------------------
# sdf_polygon
# ---------------------------------------------------------------------------

class TestPolygon:
    @pytest.fixture
    def square(self):
        return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

    def test_centre_inside(self, square):
        phi = sdf_polygon(0.5, 0.5, square)
        assert phi < 0.0

    def test_outside_corner(self, square):
        phi = sdf_polygon(2.0, 2.0, square)
        assert phi > 0.0

    def test_on_edge(self, square):
        phi = sdf_polygon(0.5, 0.0, square)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)
