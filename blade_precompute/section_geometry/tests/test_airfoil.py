"""
tests.test_airfoil
==================
Tests for AirfoilSDF: NACA generation, SDF sign, and geometry queries.
"""

import numpy as np
import pytest
import tempfile
import os

from blade_precompute.section_geometry.engine.implicit_section_geometry import AirfoilSDF


@pytest.fixture
def naca0012():
    return AirfoilSDF.from_naca("0012", n_points=200, chord=1.0)


@pytest.fixture
def naca2412():
    return AirfoilSDF.from_naca("2412", n_points=200, chord=1.0)


class TestNACAGeneration:
    def test_vertex_count(self, naca0012):
        # n_points=200 → ~199 vertices (half each side minus duplicates)
        assert 190 <= len(naca0012.vertices) <= 210

    def test_chord_length(self, naca0012):
        verts = naca0012.vertices
        chord_actual = verts[:, 0].max() - verts[:, 0].min()
        np.testing.assert_allclose(chord_actual, 1.0, atol=0.01)

    def test_symmetric_airfoil_zero_camber(self, naca0012):
        xc, yc = naca0012.camber_line(n_points=50)
        # For symmetric airfoil the camber line should be ≈ 0
        np.testing.assert_allclose(yc, 0.0, atol=1e-3)

    def test_cambered_airfoil_positive_camber(self, naca2412):
        xc, yc = naca2412.camber_line(n_points=50)
        # NACA 2412 has positive camber
        assert yc.max() > 0.005

    def test_thickness_positive(self, naca0012):
        xc, t = naca0012.thickness_distribution()
        assert (t >= 0).all()


class TestSDFSign:
    def test_interior_negative(self, naca0012):
        # Point well inside a symmetric NACA 0012 at mid-chord
        phi = naca0012(0.5, 0.0)
        assert phi < 0.0

    def test_exterior_positive(self, naca0012):
        phi = naca0012(-0.5, 0.0)   # upstream of LE
        assert phi > 0.0
        phi = naca0012(1.5, 0.0)    # downstream of TE
        assert phi > 0.0
        phi = naca0012(0.5, 0.5)    # far above
        assert phi > 0.0

    def test_leading_edge_near_zero(self, naca0012):
        le = naca0012.leading_edge
        phi = naca0012(le[0], le[1])
        assert abs(phi) < 0.02   # close to boundary


class TestGeometryQueries:
    def test_leading_edge_approx(self, naca0012):
        le = naca0012.leading_edge
        np.testing.assert_allclose(le[0], 0.0, atol=0.01)

    def test_trailing_edge_approx(self, naca0012):
        te = naca0012.trailing_edge
        np.testing.assert_allclose(te[0], 1.0, atol=0.01)

    def test_normalise(self, naca2412):
        af_norm = naca2412.normalise()
        np.testing.assert_allclose(af_norm.chord, 1.0, atol=1e-10)
        le = af_norm.leading_edge
        np.testing.assert_allclose(le[0], 0.0, atol=1e-10)

    def test_scale(self, naca0012):
        af2 = naca0012.scale(2.0)
        np.testing.assert_allclose(af2.chord, 2.0)
        verts = af2.vertices
        chord_actual = verts[:, 0].max() - verts[:, 0].min()
        np.testing.assert_allclose(chord_actual, 2.0, atol=0.02)

    def test_translate(self, naca0012):
        af2 = naca0012.translate(5.0, 1.0)
        le = af2.leading_edge
        np.testing.assert_allclose(le[0], 5.0, atol=0.01)
        np.testing.assert_allclose(le[1], 1.0, atol=0.01)


class TestGridEval:
    def test_array_output_shape(self, naca0012):
        x = np.linspace(-0.1, 1.1, 50)
        y = np.linspace(-0.2, 0.2, 30)
        X, Y = np.meshgrid(x, y)
        phi = naca0012(X, Y)
        assert phi.shape == (30, 50)


class TestInputValidation:
    def test_empty_vertices_raises(self):
        with pytest.raises(ValueError, match="At least 3 vertices"):
            AirfoilSDF.from_array(np.empty((0, 2)))

    def test_invalid_vertices_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            AirfoilSDF.from_array(np.array([0.0, 1.0, 2.0]))

    def test_non_positive_chord_raises(self):
        verts = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]])
        with pytest.raises(ValueError, match="positive finite"):
            AirfoilSDF.from_array(verts, chord=0.0)

    def test_empty_dat_file_raises_value_error(self):
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as fh:
            path = fh.name
            fh.write("header only\n")
        try:
            with pytest.raises(ValueError, match="No valid airfoil coordinate rows found"):
                AirfoilSDF.from_dat(path)
        finally:
            os.unlink(path)
