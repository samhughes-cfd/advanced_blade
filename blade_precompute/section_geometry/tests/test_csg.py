"""
tests.test_csg
==============
Unit tests for CSG operations on known SDF pairs.
"""

import numpy as np
import pytest
from blade_precompute.section_geometry.geometry.csg import intersect, offset, shell, subtract, union
from blade_precompute.section_geometry.geometry.primitives import sdf_circle


def c1(x, y): return sdf_circle(x, y, cx=0.0, cy=0.0, r=1.0)
def c2(x, y): return sdf_circle(x, y, cx=1.0, cy=0.0, r=1.0)


class TestUnion:
    def test_inside_either(self):
        f = union(c1, c2)
        # Point inside c1 only
        phi = f(-0.5, 0.0)
        assert phi < 0.0
        # Point inside c2 only
        phi = f(1.5, 0.0)
        assert phi < 0.0

    def test_outside_both(self):
        f = union(c1, c2)
        phi = f(10.0, 0.0)
        assert phi > 0.0

    def test_commutativity(self):
        f_ab = union(c1, c2)
        f_ba = union(c2, c1)
        x, y = np.linspace(-2, 3, 50), np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        np.testing.assert_allclose(f_ab(X, Y), f_ba(X, Y), atol=1e-12)


class TestIntersect:
    def test_inside_overlap(self):
        f = intersect(c1, c2)
        # Point in overlap region (between 0 and 1 on x-axis)
        phi = f(0.5, 0.0)
        assert phi < 0.0

    def test_outside_overlap(self):
        f = intersect(c1, c2)
        # Point inside c1 but outside c2
        phi = f(-0.5, 0.0)
        assert phi > 0.0


class TestSubtract:
    def test_inside_base_outside_cutter(self):
        f = subtract(c1, c2)
        # Far left of c1, away from c2
        phi = f(-0.5, 0.0)
        assert phi < 0.0

    def test_in_overlap_region_removed(self):
        f = subtract(c1, c2)
        # Overlap region should be exterior after subtraction
        phi = f(0.7, 0.0)
        assert phi > 0.0


class TestOffset:
    def test_dilate(self):
        f = offset(c1, 0.5)   # shrinks by 0.5 (phi - 0.5)
        # Point at r=1.2 should now be inside the dilated circle (r=1.5)
        phi_orig = c1(1.2, 0.0)
        phi_off  = f(1.2, 0.0)
        assert phi_orig > 0.0  # outside original
        assert phi_off  < 0.0  # inside dilated

    def test_erode(self):
        f = offset(c1, -0.3)  # erodes (phi + 0.3)
        # Point at r=0.8 is inside original but outside eroded (r=0.7)
        phi_orig = c1(0.8, 0.0)
        phi_off  = f(0.8, 0.0)
        assert phi_orig < 0.0
        assert phi_off  > 0.0


class TestShell:
    def test_on_surface_inside_shell(self):
        f = shell(c1, thickness=0.2)
        # Point on original boundary (phi=0) → inside shell
        phi = f(1.0, 0.0)
        np.testing.assert_allclose(phi, -0.1, atol=1e-10)

    def test_well_inside_outside_shell(self):
        f = shell(c1, thickness=0.2)
        phi = f(0.0, 0.0)  # centre: phi_original = -1
        np.testing.assert_allclose(phi, 0.9, atol=1e-10)


class TestGrid:
    """Integration: evaluate on a numpy grid."""
    def test_array_eval(self):
        f = union(c1, c2)
        x = np.linspace(-2, 3, 100)
        y = np.linspace(-2, 2, 80)
        X, Y = np.meshgrid(x, y)
        phi = f(X, Y)
        assert phi.shape == (80, 100)
        # Interior of union has negative values
        assert phi.min() < 0.0
