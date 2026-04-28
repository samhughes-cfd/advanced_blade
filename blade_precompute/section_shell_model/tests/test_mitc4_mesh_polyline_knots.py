"""MITC4 strip nodes include polyline knots (cap/web; sparse skin) — junction snap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_STRESS = _REPO / "examples" / "section_stress_model"
for _p in (str(_REPO), str(_REPO / "examples"), str(_STRESS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def airfoil():
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF

    return AirfoilSDF.from_naca("0012", n_points=100, chord=1.0)


def _target_element_length_demo_m(chord_m: float) -> float:
    return float(max(0.015, 0.012 * chord_m))


def _web_polylines_from_shell(shell: object) -> list[np.ndarray]:
    webs: list[np.ndarray] = []
    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind != "web":
            continue
        arr = np.asarray(m.midline_b, dtype=float)
        if arr.shape[0] >= 2:
            webs.append(arr)
    return webs


@pytest.mark.parametrize("layout_key", ["2C-CN", "2D-CN", "3C-CN"])
def test_cap_mesh_nodes_intersect_web_midlines(layout_key: str):
    """Cap mesh carries web-centric shared nodes at cap-web intersections."""
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.orchestration.system_layout import build_section_view, resolve_system_type
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    chord_m = 1.655959896
    twist_rad = float(np.deg2rad(34.110405083))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord_m, closed_te=True
    )
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell = build_shell_mesh_inputs(
        section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )
    webs = _web_polylines_from_shell(shell)
    assert webs, layout_key

    mesh = build_mitc4_mesh(
        shell,
        target_element_length_m=_target_element_length_demo_m(chord_m),
        endpoint_tol=1e-6,
    )
    cap_yz = np.vstack(
        [np.asarray(pm.yz_nodes, dtype=float) for pm in mesh.panel_meshes if pm.kind == "cap"]
    )
    assert cap_yz.size > 0, layout_key
    best = float("inf")
    for w in webs:
        for j in range(w.shape[0] - 1):
            a = w[j]
            b = w[j + 1]
            ab = b - a
            l2 = float(np.dot(ab, ab)) + 1e-30
            rel = cap_yz - a.reshape(1, 2)
            t = np.clip((rel @ ab.reshape(2, 1)).ravel() / l2, 0.0, 1.0)
            proj = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)
            d = float(np.linalg.norm(cap_yz - proj, axis=1).min())
            best = min(best, d)
    assert best < 1e-4, (layout_key, best)


@pytest.mark.parametrize("layout_key", ["2C-CN", "2D-CN"])
def test_mitc4_cap_nodes_include_resampled_cap_midline_knots(layout_key: str):
    """Class C: MITC4 cap nodes include all cap midline knots from shell inputs."""
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.orchestration.system_layout import build_section_view, resolve_system_type
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    chord_m = 1.655959896
    twist_rad = float(np.deg2rad(34.110405083))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord_m, closed_te=True
    )
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell = build_shell_mesh_inputs(
        section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )
    mesh = build_mitc4_mesh(
        shell,
        target_element_length_m=_target_element_length_demo_m(chord_m),
        endpoint_tol=1e-6,
    )
    cap_mesh = {
        pm.panel_label.split(":", 1)[1]: np.asarray(pm.yz_nodes, dtype=float)
        for pm in mesh.panel_meshes
        if pm.kind == "cap"
    }
    assert cap_mesh
    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind != "cap":
            continue
        arr = np.asarray(m.midline_b, dtype=float)
        yz = cap_mesh[m.label]
        for q in arr:
            d = float(np.linalg.norm(yz - q.reshape(1, 2), axis=1).min())
            assert d < 1e-8, (layout_key, m.label, d)


def test_1a_cn_skin_mesh_endpoints_match_panel_polyline(airfoil):
    """Skin strip MITC4 endpoints match underlying panel polyline ends (no drift)."""
    from blade_precompute.orchestration.system_layout import build_section_view, resolve_system_type
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )
    from blade_precompute.section_shell_model.lib.topology_v2 import build_section_v2

    layout_key = "1A-CN"
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell = build_shell_mesh_inputs(
        section, twist_rad=0.0, layout_key=layout_key, n_web_samples=20, n_cap_samples=40
    )
    panels, _, _, _ = build_section_v2(shell)
    mesh = build_mitc4_mesh(shell, n_elements_per_panel=5, endpoint_tol=1e-6)

    for pi, pm in enumerate(mesh.panel_meshes):
        if pm.kind != "skin" or pm.yz_nodes.size == 0:
            continue
        p = panels[pi]
        nodes = np.asarray(p.nodes, dtype=float)
        assert np.allclose(pm.yz_nodes[0], nodes[0], atol=1e-9, rtol=0.0)
        assert np.allclose(pm.yz_nodes[-1], nodes[-1], atol=1e-9, rtol=0.0)
