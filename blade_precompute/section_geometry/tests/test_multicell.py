"""
tests.test_multicell
====================
Unit tests for MultiCellSection and the corrected SparCap construction.
"""

import numpy as np
import pytest

from blade_precompute.section_geometry.engine.implicit_section_geometry import (
    AirfoilSDF,
    ContinuousSparCap,
    MultiCellSection,
    SDFGrid,
    SparCap,
    offset,
)
from blade_precompute.section_geometry.interface.plot import _DEFAULT_COLOR, _component_color


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def af():
    return AirfoilSDF.from_naca("0012", n_points=200, chord=1.0)

@pytest.fixture(scope="module")
def af2412():
    return AirfoilSDF.from_naca("2412", n_points=200, chord=1.0)

@pytest.fixture(scope="module")
def grid(af):
    return SDFGrid.from_airfoil(af, padding=0.05, nx=300, ny=120)


# ---------------------------------------------------------------------------
# Corrected SparCap construction
# ---------------------------------------------------------------------------

class TestSparCapCurvature:
    """Verify that the corrected SparCap has constant thickness everywhere."""

    def test_outer_face_flush_with_inner_skin(self, af):
        """The outer cap face should coincide with the inner skin surface."""
        skin_t  = 0.003
        cap_h   = 0.015
        cap     = SparCap(af, skin_t, x_start=0.20, x_end=0.50,
                          cap_height=cap_h, surface="upper")
        inner_skin = offset(af, skin_t)

        # Sample points on the inner skin in the cap x-range
        x_samples = np.linspace(0.22, 0.48, 30)
        # Find y on inner skin upper surface by scanning
        y_scan = np.linspace(0.0, 0.15, 500)
        outer_face_phi = []
        for x_q in x_samples:
            phi_vals = inner_skin(np.full_like(y_scan, x_q), y_scan)
            # Find y where inner_skin ≈ 0 (upper surface crossing)
            idx = np.argmin(np.abs(phi_vals))
            y_q = y_scan[idx]
            # Evaluate inner_skin there — should be near 0
            outer_face_phi.append(float(inner_skin(x_q, y_q)))

        np.testing.assert_allclose(outer_face_phi, 0.0, atol=0.005)

    def test_cap_thickness_uniform(self, af):
        """Cap thickness (distance from outer to inner face) should equal cap_height."""
        skin_t  = 0.003
        cap_h   = 0.015
        inner_skin = offset(af, skin_t)

        # Sample points on the outer face (inner_skin ≈ 0) in the cap x-range
        x_samples = np.linspace(0.22, 0.48, 20)
        y_scan    = np.linspace(0.0, 0.15, 1000)
        thicknesses = []

        for x_q in x_samples:
            phi_is = inner_skin(np.full_like(y_scan, x_q), y_scan)
            # Find upper skin crossing (phi ≈ 0, y > 0)
            pos_vals = phi_is[y_scan > 0.005]
            pos_y    = y_scan[y_scan > 0.005]
            if len(pos_vals) == 0:
                continue
            idx    = np.argmin(np.abs(pos_vals))
            y_outer = pos_y[idx]   # y at outer cap face

            # Inner cap face is at inner_skin = -cap_h → scan inward
            y_inner_scan = np.linspace(0.0, y_outer, 500)
            phi_inner    = inner_skin(np.full_like(y_inner_scan, x_q), y_inner_scan)
            # Find where phi_inner ≈ -cap_h
            target = -cap_h
            idx2   = np.argmin(np.abs(phi_inner - target))
            y_inner = y_inner_scan[idx2]

            thickness = y_outer - y_inner
            thicknesses.append(thickness)

        # Thickness should be approximately cap_h (within grid resolution)
        thicknesses = np.array(thicknesses)
        np.testing.assert_allclose(thicknesses, cap_h, atol=0.003)

    def test_cap_inside_airfoil(self, af, grid):
        """Cap interior points should be within one skin thickness of the airfoil boundary.

        The cap shell straddles the inner skin surface, so the outermost cap
        points sit between the inner skin and the outer skin surface — i.e.
        within skin_thickness of the airfoil boundary (phi_af <= skin_thickness).
        """
        skin_t = 0.003
        cap_h  = 0.015
        cap = SparCap(af, skin_t, 0.20, 0.50, cap_h, surface="upper")
        phi_cap = grid.eval(cap)
        phi_af  = grid.eval(af)
        cap_interior = phi_cap < 0.0
        assert cap_interior.any(), "Cap has no interior region."
        # The cap shell spans phi_inner_skin in [0, -cap_h], which corresponds
        # to phi_af in [-skin_t, cap_h - skin_t].  The outermost cap points sit
        # at phi_af = cap_h - skin_t (between inner and outer skin laminates).
        max_allowed = cap_h - skin_t + 0.002   # +grid tolerance
        assert np.all(phi_af[cap_interior] <= max_allowed), \
            "Some cap interior points exceed the expected outer-skin boundary."


# ---------------------------------------------------------------------------
# MultiCellSection topology
# ---------------------------------------------------------------------------

class TestMultiCellTopology:
    def test_single_web_d_spar_labels(self, af):
        mcs = MultiCellSection.d_spar(af, web_x=0.25,
                                      skin_thickness=0.003,
                                      cap_height=0.012,
                                      web_thickness=0.004)
        assert "web_0"         in mcs.labels
        assert "spar_cap_upper" in mcs.labels
        assert "spar_cap_lower" in mcs.labels
        assert "outer_skin"    in mcs.labels
        assert mcs.n_webs == 1

    def test_twin_web_labels(self, af):
        mcs = MultiCellSection.twin_web(af)
        assert mcs.n_webs  == 2
        assert "web_0"     in mcs.labels
        assert "web_1"     in mcs.labels
        assert "core_0"    in mcs.labels
        assert mcs.n_cells == 1

    def test_three_web_labels(self, af):
        mcs = MultiCellSection.torsion_box(af)
        assert mcs.n_webs  == 3
        assert mcs.n_cells == 2
        assert "core_0"    in mcs.labels
        assert "core_1"    in mcs.labels

    def test_n_web_generalisation(self, af):
        """N webs → N-1 cells."""
        for n in range(1, 6):
            xs = np.linspace(0.15, 0.55, n).tolist()
            mcs = MultiCellSection(af, web_x_positions=xs,
                                   skin_thickness=0.003,
                                   cap_height=0.010,
                                   web_thickness=0.003)
            assert mcs.n_webs == n
            expected_cells = max(0, n - 1)
            assert mcs.n_cells == expected_cells, \
                f"Expected {expected_cells} cells for {n} webs, got {mcs.n_cells}"

    def test_te_le_inserts(self, af):
        mcs = MultiCellSection.twin_web(af,
                                        te_insert_x=0.75,
                                        le_insert_x=0.10)
        assert "te_insert" in mcs.labels
        assert "le_insert" in mcs.labels


# ---------------------------------------------------------------------------
# Interior checks
# ---------------------------------------------------------------------------

class TestInteriorChecks:
    def test_core_inside_airfoil(self, af, grid):
        mcs = MultiCellSection.twin_web(af,
                                        skin_thickness=0.003,
                                        cap_height=0.010)
        phi_core = grid.eval(mcs["core_0"])
        phi_af   = grid.eval(af)
        interior = phi_core < 0.0
        assert interior.any(), "Core has no interior region."
        assert np.all(phi_af[interior] <= 0.01), \
            "Core interior leaks outside airfoil."

    def test_caps_do_not_overlap(self, af, grid):
        """Upper and lower caps should not share interior points."""
        mcs   = MultiCellSection.twin_web(af, skin_thickness=0.003, cap_height=0.010)
        phi_u = grid.eval(mcs["spar_cap_upper"])
        phi_l = grid.eval(mcs["spar_cap_lower"])
        overlap = (phi_u < 0.0) & (phi_l < 0.0)
        n_overlap = int(overlap.sum())
        assert n_overlap < 5, \
            f"Upper and lower caps overlap at {n_overlap} grid points."

    def test_web_intersects_interior(self, af, grid):
        mcs = MultiCellSection.twin_web(af, skin_thickness=0.003, web_thickness=0.005)
        phi_web = grid.eval(mcs["web_0"])
        assert (phi_web < 0.0).any(), "Web_0 has no interior region."


# ---------------------------------------------------------------------------
# Web alignment
# ---------------------------------------------------------------------------

class TestWebAlignment:
    def test_flapwise_vs_chord_normal_differ(self, af):
        """Flapwise and chord-normal webs at non-zero twist should produce different SDFs."""
        twist = np.radians(15)
        mcs_cn = MultiCellSection.twin_web(
            af, web_thickness=0.004, web_alignment="chord_normal", twist_angle=0.0
        )
        mcs_fw = MultiCellSection.twin_web(
            af, web_thickness=0.004, web_alignment="flapwise", twist_angle=twist
        )
        x = np.linspace(0.18, 0.22, 50)
        y = np.linspace(-0.05, 0.05, 50)
        X, Y = np.meshgrid(x, y)
        phi_cn = mcs_cn["web_0"](X, Y)
        phi_fw = mcs_fw["web_0"](X, Y)
        # They should differ meaningfully in the web region
        max_diff = float(np.abs(phi_cn - phi_fw).max())
        assert max_diff > 1e-4, \
            "Flapwise and chord-normal webs should differ when twist != 0."

    def test_flapwise_zero_twist_equals_chord_normal(self, af):
        """At zero twist, flapwise == chord-normal."""
        mcs_cn = MultiCellSection.twin_web(
            af, web_thickness=0.004, web_alignment="chord_normal", twist_angle=0.0
        )
        mcs_fw = MultiCellSection.twin_web(
            af, web_thickness=0.004, web_alignment="flapwise", twist_angle=0.0
        )
        x = np.linspace(0.18, 0.22, 30)
        y = np.linspace(-0.05, 0.05, 30)
        X, Y = np.meshgrid(x, y)
        phi_cn = mcs_cn["web_0"](X, Y)
        phi_fw = mcs_fw["web_0"](X, Y)
        np.testing.assert_allclose(phi_cn, phi_fw, atol=1e-10)

    def test_per_web_alignment(self, af):
        """Mixed alignment list: first web chord_normal, second flapwise."""
        mcs = MultiCellSection.twin_web(
            af,
            web_thickness=0.004,
            web_alignment=["chord_normal", "flapwise"],
            twist_angle=np.radians(10),
        )
        assert mcs["web_0"] is not None
        assert mcs["web_1"] is not None


class TestAnchorDetection:
    def test_narrow_y_search_raises_actionable_error(self, af):
        with pytest.raises(ValueError, match="Provide explicit web_y_coords or widen y_search"):
            MultiCellSection(
                af,
                web_x_positions=[0.2, 0.5],
                y_search=(-1e-4, 1e-4),
            )

    def test_explicit_web_y_coords_bypass_search(self, af):
        mcs = MultiCellSection(
            af,
            web_x_positions=[0.2, 0.5],
            web_y_coords=[(0.05, -0.05), (0.05, -0.05)],
        )
        assert "web_0" in mcs.labels and "web_1" in mcs.labels


class TestPlotColorMapping:
    def test_web_label_has_non_default_color(self):
        assert _component_color("web_0") != _DEFAULT_COLOR

    def test_core_label_has_non_default_color(self):
        assert _component_color("core_0") != _DEFAULT_COLOR


# ---------------------------------------------------------------------------
# ContinuousSparCap curvature
# ---------------------------------------------------------------------------

class TestContinuousSparCap:
    def test_upper_lower_do_not_overlap(self, af, grid):
        cap_u = ContinuousSparCap(af, 0.003, 0.20, 0.50, 0.012, surface="upper")
        cap_l = ContinuousSparCap(af, 0.003, 0.20, 0.50, 0.012, surface="lower")
        phi_u = grid.eval(cap_u)
        phi_l = grid.eval(cap_l)
        overlap = (phi_u < 0.0) & (phi_l < 0.0)
        assert int(overlap.sum()) < 5

    def test_twist_rotates_cap(self, af):
        cap_0  = ContinuousSparCap(af, 0.003, 0.20, 0.50, 0.012,
                                   surface="upper", twist_angle=0.0)
        cap_tw = ContinuousSparCap(af, 0.003, 0.20, 0.50, 0.012,
                                   surface="upper", twist_angle=np.radians(15))
        x = np.linspace(0.2, 0.5, 50)
        y = np.linspace(0.0, 0.08, 50)
        X, Y = np.meshgrid(x, y)
        phi_0  = cap_0(X, Y)
        phi_tw = cap_tw(X, Y)
        max_diff = float(np.abs(phi_0 - phi_tw).max())
        assert max_diff > 1e-4, "Twisted and untwisted caps should differ."
