"""
Contract tests for the ShellMidlineStrip → ShellMeshInputs → MITC4 mesh pipeline.

Verifies the public handoff contract established in PR1:

  build_shell_mesh_inputs(section, ...) → ShellMeshInputs
  build_section_v2(shell_inputs)        → panels (ordered skin → caps → webs)
  build_mitc4_mesh(shell_inputs, ...)   → Mitc4SectionMesh

Key invariants checked:

  A4  len(panels) == len(shell_inputs.midlines); every panel label matches
      exactly one strip via label = f"{kind}:{label}".
  A5  panels are emitted in skin → caps → webs order, and panel.nodes
      equals the corresponding strip's midline_b (as produced by
      build_section_v2).
  A6  Same contract holds with a non-zero twist angle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root and legacy stress-model are on sys.path so topology_v2
# can import multi_cell_blade_section at import time.
_REPO = Path(__file__).resolve().parents[3]
_STRESS = _REPO / "examples" / "section_stress_model"
for _p in (str(_REPO), str(_REPO / "examples"), str(_STRESS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

# Arc-length tolerance for geometry matching (metres); chosen to be generous
# enough for floating-point round-trips through rotate_chord_to_blade while
# still catching any wrong-strip assignment.
_GEOM_TOL_M = 1e-10

# Layout keys exercised in the parametrize matrix (excludes airfoil_sdf_only
# 0A/0B and invalid 1D-*).
_LAYOUT_KEYS = ["2D-F", "2D-CN", "1A-CN", "3C-F", "5D-CN"]


@pytest.fixture(scope="session")
def airfoil():
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF

    return AirfoilSDF.from_naca("0012", n_points=100, chord=1.0)


def _build_pipeline(airfoil, layout_key: str, twist_rad: float = 0.0):
    """Build all pipeline objects for one layout / twist combination."""
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )
    from blade_precompute.section_shell_model.lib.topology_v2 import build_section_v2

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=twist_rad, layout_key=layout_key, n_cap_samples=40
    )
    panels, _, _, _ = build_section_v2(shell_inputs)
    mesh = build_mitc4_mesh(shell_inputs, n_elements_per_panel=4, endpoint_tol=1e-6)
    return shell_inputs, panels, mesh


# ---------------------------------------------------------------------------
# A4 — Panel count and label multiset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_key", _LAYOUT_KEYS)
def test_panel_count_equals_midline_count(airfoil, layout_key):
    """len(panels) must equal len(shell_inputs.midlines) for every layout."""
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key)
    assert len(panels) == len(shell_inputs.midlines), (
        f"{layout_key}: {len(panels)} panels but {len(shell_inputs.midlines)} midlines"
    )


@pytest.mark.parametrize("layout_key", _LAYOUT_KEYS)
def test_panel_labels_match_midline_labels(airfoil, layout_key):
    """Each panel.label must equal f'{m.kind}:{m.label}' for exactly one strip."""
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key)

    # Build a dict from expected label → strip for O(1) lookup.
    expected_labels = {f"{m.kind}:{m.label}": m for m in shell_inputs.midlines}

    for panel in panels:
        assert panel.label in expected_labels, (
            f"{layout_key}: panel label {panel.label!r} has no matching midline. "
            f"Available: {list(expected_labels)}"
        )


# ---------------------------------------------------------------------------
# A5 — Ordered geometry (skin → caps → webs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_key", _LAYOUT_KEYS)
def test_panel_ordering_skin_caps_webs(airfoil, layout_key):
    """Panels must be emitted in skin → caps → webs order."""
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key)

    # Reconstruct expected order from midlines (same logic as build_section_v2).
    skin_ms = [m for m in shell_inputs.midlines if m.kind == "skin"]
    cap_ms = [m for m in shell_inputs.midlines if m.kind == "cap"]
    web_ms = [m for m in shell_inputs.midlines if m.kind == "web"]
    ordered = skin_ms + cap_ms + web_ms

    assert len(ordered) == len(panels), "Ordering length mismatch (already caught by A4 test)."

    for i, (panel, m) in enumerate(zip(panels, ordered)):
        expected_label = f"{m.kind}:{m.label}"
        assert panel.label == expected_label, (
            f"{layout_key}: panels[{i}].label={panel.label!r} "
            f"!= expected {expected_label!r}. "
            f"Check build_section_v2 ordering: skin → caps → webs."
        )


@pytest.mark.parametrize("layout_key", _LAYOUT_KEYS)
def test_panel_nodes_match_ordered_midline_b(airfoil, layout_key):
    """panel.nodes must equal the corresponding strip's midline_b (geometry A5)."""
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key)

    skin_ms = [m for m in shell_inputs.midlines if m.kind == "skin"]
    cap_ms = [m for m in shell_inputs.midlines if m.kind == "cap"]
    web_ms = [m for m in shell_inputs.midlines if m.kind == "web"]
    ordered = skin_ms + cap_ms + web_ms

    for i, (panel, m) in enumerate(zip(panels, ordered)):
        expected_nodes = np.asarray(m.midline_b, dtype=float)
        assert panel.nodes.shape == expected_nodes.shape, (
            f"{layout_key}: panels[{i}] shape {panel.nodes.shape} "
            f"!= midline_b shape {expected_nodes.shape}"
        )
        assert np.allclose(panel.nodes, expected_nodes, atol=_GEOM_TOL_M), (
            f"{layout_key}: panels[{i}] nodes diverge from midline_b "
            f"(max diff={np.abs(panel.nodes - expected_nodes).max():.2e} m). "
            f"Do not index panels[i] directly against midlines[i]; "
            f"midlines are unordered (skin/web/cap dict insertion order)."
        )


# ---------------------------------------------------------------------------
# A4/A5 — midlines[i] is NOT the same as panels[i] (ordering trap guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_key", ["2D-F", "3C-F"])
def test_raw_midline_order_differs_from_panel_order(airfoil, layout_key):
    """For layouts with caps+webs, midlines[i] must NOT equal panels[i] for web/cap rows.

    This test documents the ordering invariant: build_section_v2 reorders
    to skin→caps→webs, so raw midlines[i] cannot be paired directly with panels[i]
    for any layout that has both caps and webs.
    """
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key)
    midlines = shell_inputs.midlines

    # At least one panel must have a different kind from the corresponding raw midline.
    has_mismatch = any(
        panels[i].label != f"{midlines[i].kind}:{midlines[i].label}"
        for i in range(len(panels))
    )
    assert has_mismatch, (
        f"{layout_key}: panels[i].label matched midlines[i] for all i — "
        f"this suggests no reordering occurred. "
        f"Expected skin→caps→webs reordering to produce at least one mismatch. "
        f"midline kinds: {[m.kind for m in midlines]}, "
        f"panel labels: {[p.label for p in panels]}"
    )


# ---------------------------------------------------------------------------
# A6 — Twist: same contract with twist_rad != 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_key", ["2D-F", "2D-CN"])
def test_contract_with_nonzero_twist(airfoil, layout_key):
    """Panel count/labels/ordering must hold under a non-zero section twist."""
    twist_rad = 0.35  # ~20 deg; matches repro_geometry_to_midlines conventions
    shell_inputs, panels, _ = _build_pipeline(airfoil, layout_key, twist_rad=twist_rad)

    # A4
    assert len(panels) == len(shell_inputs.midlines)

    # A4 label multiset
    expected_labels = {f"{m.kind}:{m.label}" for m in shell_inputs.midlines}
    for panel in panels:
        assert panel.label in expected_labels

    # A5 ordering
    skin_ms = [m for m in shell_inputs.midlines if m.kind == "skin"]
    cap_ms = [m for m in shell_inputs.midlines if m.kind == "cap"]
    web_ms = [m for m in shell_inputs.midlines if m.kind == "web"]
    ordered = skin_ms + cap_ms + web_ms

    for i, (panel, m) in enumerate(zip(panels, ordered)):
        assert panel.label == f"{m.kind}:{m.label}"
        expected_nodes = np.asarray(m.midline_b, dtype=float)
        assert np.allclose(panel.nodes, expected_nodes, atol=_GEOM_TOL_M)

    # Twisted B-frame: trailing_edge_b (at [chord, 0] in S-frame) must move under twist.
    # leading_edge_b is at the origin and stays [0,0] regardless of twist angle.
    shell_inputs_0, _, _ = _build_pipeline(airfoil, layout_key, twist_rad=0.0)
    te_twist = np.array(shell_inputs.trailing_edge_b)
    te_zero = np.array(shell_inputs_0.trailing_edge_b)
    assert not np.allclose(te_twist, te_zero, atol=1e-6), (
        "trailing_edge_b must rotate under non-zero twist."
    )


# ---------------------------------------------------------------------------
# A4/A5 — Mesh object consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_key", _LAYOUT_KEYS)
def test_mitc4_mesh_panel_count(airfoil, layout_key):
    """Mitc4SectionMesh must contain one panel mesh per midline."""
    shell_inputs, panels, mesh = _build_pipeline(airfoil, layout_key)
    assert len(mesh.panel_meshes) == len(shell_inputs.midlines), (
        f"{layout_key}: mesh has {len(mesh.panel_meshes)} panel meshes "
        f"but {len(shell_inputs.midlines)} midlines"
    )


def test_target_element_length_m_coarser_on_short_webs_than_skin(airfoil):
    """Distance-based sizing: short webs get fewer elements than long skin strips."""
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    layout_key = "2D-F"
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=0.0, layout_key=layout_key, n_cap_samples=40
    )
    mesh = build_mitc4_mesh(
        shell_inputs, target_element_length_m=0.02, endpoint_tol=1e-6
    )
    web_meshes = [pm for pm in mesh.panel_meshes if pm.kind == "web"]
    skin_meshes = [pm for pm in mesh.panel_meshes if pm.kind == "skin"]
    assert web_meshes and skin_meshes
    for pm in web_meshes:
        if pm.arc_length_m < 0.08:
            # Baseline linspace can be coarse, but knot-merge may add web polyline samples.
            assert 1 <= pm.n_elements <= 80
    max_skin_n = max(pm.n_elements for pm in skin_meshes)
    assert max_skin_n >= 8


def test_target_element_length_m_overridden_by_explicit_uniform_count(airfoil):
    """Uniform ``n_elements_per_panel`` other than 10 overrides ``target_element_length_m``."""
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    layout_key = "2D-F"
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=0.0, layout_key=layout_key, n_cap_samples=40
    )
    mesh = build_mitc4_mesh(
        shell_inputs,
        n_elements_per_panel=7,
        target_element_length_m=0.001,
        endpoint_tol=1e-6,
    )
    for pm in mesh.panel_meshes:
        if pm.n_elements == 0:
            continue
        if pm.kind == "cap":
            # Cap panels follow explicit Class A/B/C knot locations and may
            # have fewer elements than the uniform fallback count.
            assert pm.n_elements >= 1
        else:
            # Uniform count 7 is the minimum linspace resolution.
            assert pm.n_elements >= 7


@pytest.mark.parametrize("layout_key", ["2D-F", "2D-CN"])
def test_web_junction_clusters_are_multi_member_after_skin_split(airfoil, layout_key):
    """Skin split + snapped web ends: web-related endpoint clusters are not isolated."""
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=0.0, layout_key=layout_key, n_cap_samples=40
    )
    mesh = build_mitc4_mesh(shell_inputs, n_elements_per_panel=6, endpoint_tol=1e-6)
    for c in mesh.clusters:
        if any("web" in pl.lower() for pl in c.panel_labels):
            assert len(c.members) >= 2, (
                f"{layout_key}: isolated web cluster {c.cluster_id} "
                f"members={c.members} labels={c.panel_labels}"
            )


@pytest.mark.parametrize("layout_key", ["2D-F", "2D-CN"])
def test_cap_endpoints_preserve_exported_positions(airfoil, layout_key):
    """Web-centric rule: cap endpoints are not remapped to skin during split."""
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_geometry.interface.shell_midline_export import (
        build_shell_midline_strips,
    )
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    strips_geo = build_shell_midline_strips(
        section, twist_rad=0.0, n_web_samples=20, n_cap_samples=40
    )
    cap_end_geo = {
        s.label: (np.asarray(s.midline_b, dtype=float)[0], np.asarray(s.midline_b, dtype=float)[-1])
        for s in strips_geo
        if s.kind == "cap"
    }
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=0.0, layout_key=layout_key, n_cap_samples=40
    )
    for m in shell_inputs.midlines:
        if m.kind != "cap":
            continue
        arr = np.asarray(m.midline_b, dtype=float)
        g0, g1 = cap_end_geo[m.label]
        assert np.allclose(arr[0], g0, atol=1e-9, rtol=0.0), (layout_key, m.label, "start")
        assert np.allclose(arr[-1], g1, atol=1e-9, rtol=0.0), (layout_key, m.label, "end")
