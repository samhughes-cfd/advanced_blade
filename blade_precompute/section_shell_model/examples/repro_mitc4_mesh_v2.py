"""
Smoke repro: ``build_shell_midline_strips`` + ``ShellMidlineStrip``
→ ``build_shell_mesh_inputs`` → ``build_mitc4_mesh``.

Verifies that ``build_mitc4_mesh`` produces a wire-compatible MITC4 strip
mesh from the per-subcomponent midlines (skin, webs, caps) emitted by
``build_shell_mesh_inputs``, and that the resulting endpoint clustering
correctly identifies the section's junctions (web/skin and cap/skin
intersections).

Also exercises the explicit geometry-export path via
``build_shell_midline_strips`` (from
``blade_precompute.section_geometry.interface.shell_midline_export``) and
asserts that non-skin strips match ``ShellMeshInputs`` (skin may be split into
``seg*`` sub-strips at web/cap junctions; web/cap polylines may be snapped).

Reuses the same run007 / station i000 / NACA63-415 / 2D-F inputs as
``repro_geometry_to_midlines.py`` so the mesh can be visually compared
against the section_geometry baseline.

Outputs (PNG):
    blade_precompute/section_shell_model/examples/output/repro_mitc4_mesh_v2/
    mitc4_mesh_<layout>_i<idx>.png

Each figure has **two** equal-aspect B-frame panels:

1. **section_geometry** SDF (reference).
2. **MITC4 mesh** — ``build_mitc4_mesh(..., target_element_length_m=…)``
   from :mod:`blade_precompute.section_shell_model.lib.mitc4_mesh` (~1.2 % of
   section ``chord_m``, minimum 15 mm), using the package default uniform
   override so per-panel ``n`` scales with polyline arc length.     On this panel, **cyan quivers** show the Class C **ray** direction
   (perpendicular to skin-ring tangent, sign toward cap centroid) at skin
   vertices inside each cap’s AABB, matching
   ``_resample_one_cap_interior_from_skin_rays`` — not a fixed airfoil “material
   inward” normal.

Each figure suptitle includes **X** (web count), **Y** = ``structural_family``
(taxonomy spar-cap letter), **Z** (chord-normal vs flapwise), matching
``system_type_xyz_taxonomy.md`` (registry middle letter = **Y**).

By default writes one PNG for **every registered layout key** in
``SYSTEM_TYPE_KEYS``: the 38 multicell keys (**X** = 1..5, **Y** = A/B/C/D,
**Z** = CN or F, excluding **``1D-*``** — a single web cannot form a
continuous box spar) plus **``0A``** (X=0, Y=A: skin-only, no webs) and
**``0B``** (X=0, Y=B: skin + max-thickness cap band, cap SDF deferred so
currently skin-only too).  Total: 40 keys.

Also writes ``taxonomy_coverage.json`` in the same output directory (expected
**XYZ** set vs what was rendered, and ``excluded_invalid_xyz``).

Optional CLI: pass layout keys to render a subset, e.g.
``python -m blade_precompute.section_shell_model.examples.repro_mitc4_mesh_v2 0A 0B 2D-F 2A-CN``.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_DEBUG_LOG_PATH = Path(__file__).resolve().parents[3] / "debug-caf7ba.log"


def _agent_debug_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    # region agent log
    payload = {
        "sessionId": "caf7ba",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass
    # endregion agent log


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _all_layout_keys() -> tuple[str, ...]:
    """All registered layout keys: ``0A``, ``0B``, and all multicell ``XY-Z`` keys."""
    from blade_precompute.orchestration.system_layout import SYSTEM_TYPE_KEYS

    return SYSTEM_TYPE_KEYS


def _target_element_length_demo_m(chord_m: float) -> float:
    """~uniform physical spacing for MITC4 demo (metres), scaled from chord."""
    return float(max(0.015, 0.012 * chord_m))


def _smoke_geo_export_vs_shell_inputs(
    strips_geo: tuple[Any, ...],
    shell_inputs: Any,
    layout_key: str,
    np: Any,
) -> None:
    """``build_shell_mesh_inputs`` may split skin and snap web/cap ends to the skin midline.

    Cap strips may additionally run
    :func:`~blade_precompute.section_shell_model.lib.shell_inputs_from_section._resample_cap_interiors_from_skin_rays`,
    which replaces interior vertices (different ``N`` than ``n_cap_samples``) while
    keeping endpoints; those caps are only checked for endpoint agreement here.
    """
    geo_map = {(s.label, s.kind): s for s in strips_geo}
    shell_rows = list(shell_inputs.midlines)
    shell_map = {(m.label, m.kind): m for m in shell_rows}

    for key, s in geo_map.items():
        lab, kind = key
        if kind == "skin":
            segs = [
                m
                for m in shell_rows
                if m.kind == "skin" and str(m.label).startswith(f"{lab}:")
            ]
            if not segs:
                exact = [m for m in shell_rows if m.kind == "skin" and m.label == lab]
                assert exact and np.allclose(
                    s.midline_b, exact[0].midline_b, atol=1e-9, rtol=0.0
                ), f"[{layout_key}] missing or mismatched skin strip for {lab!r}"
            continue
        assert key in shell_map, f"[{layout_key}] missing shell midline for {key!r}"
        m = shell_map[key]
        ps = np.asarray(s.midline_b, dtype=float)
        pm = np.asarray(m.midline_b, dtype=float)
        # Webs: endpoints move onto the skin midline (~mm). Caps: ends can move
        # to the nearest snapped web foot within shell_inputs (up to tens of mm
        # under twist / real chord) while interiors may be ray-resampled.
        ep_tol = 0.025
        cap_ep_tol = 0.055
        if kind == "cap":
            assert ps.shape[1] == pm.shape[1] == 2, f"[{layout_key}] cap midline must be Nx2"
            assert float(np.linalg.norm(ps[0] - pm[0])) <= cap_ep_tol, (
                f"[{layout_key}] start endpoint drift for {key!r}"
            )
            assert float(np.linalg.norm(ps[-1] - pm[-1])) <= cap_ep_tol, (
                f"[{layout_key}] end endpoint drift for {key!r}"
            )
            continue
        if ps.shape != pm.shape:
            # region agent log
            _agent_debug_log(
                hypothesis_id="A",
                location="repro_mitc4_mesh_v2.py:_smoke_geo_export_vs_shell_inputs",
                message="midline_b shape mismatch geo vs shell_inputs",
                data={
                    "layout_key": layout_key,
                    "key": [lab, kind],
                    "geo_shape": list(ps.shape),
                    "shell_shape": list(pm.shape),
                    "kind": kind,
                },
            )
            # endregion agent log
        assert ps.shape == pm.shape, f"[{layout_key}] shape mismatch for {key!r}"
        if ps.shape[0] > 2:
            assert np.allclose(ps[1:-1], pm[1:-1], atol=1e-9, rtol=0.0), (
                f"[{layout_key}] interior midline_b mismatch for {key!r}"
            )
        assert float(np.linalg.norm(ps[0] - pm[0])) <= ep_tol, (
            f"[{layout_key}] start endpoint drift for {key!r}"
        )
        assert float(np.linalg.norm(ps[-1] - pm[-1])) <= ep_tol, (
            f"[{layout_key}] end endpoint drift for {key!r}"
        )


def _junction_display_valence(mesh: Any, c: Any) -> int:
    """Legend / title valence: 3 = skin+web+cap, 2 = skin–web or skin–cap only.

    Raw cluster size counts each panel end separately, so split skin can show
    ``n=4`` at one geometric corner; this returns **3** only when all three strip
    kinds participate, otherwise **2** for binary skin–web or skin–cap interfaces.
    """
    kinds: set[str] = set()
    for pi, _ in c.members:
        kinds.add(str(mesh.panel_meshes[int(pi)].kind))
    if kinds == {"skin", "web", "cap"}:
        return 3
    if kinds == {"skin", "web"} or kinds == {"skin", "cap"}:
        return 2
    return int(len(c.members))


def _add_skin_inward_rays_class_c_near_caps(
    ax_mesh: Any,
    shell_inputs: Any,
    *,
    chord_m: float,
) -> None:
    """Draw the Class C ray **directions** the resampler uses (same culling and math).

    This is **not** a constant “material inward” normal: it is the unit vector
    ``v = pick_one(±(perp skin tangent))`` that best aligns with
    ``cap_centroid - q`` for one spar cap, where the skin tangent is the
    *closed-ring* chord (next−prev) at each skin vertex. Arrows are only drawn at
    skin ring vertices for which
    :func:`blade_precompute.section_shell_model.lib.shell_inputs_from_section._resample_one_cap_interior_from_skin_rays`
    would also consider ``q`` — i.e. inside that cap’s axis-aligned box plus
    ``_SKIN_RAY_CAP_BBOX_MARGIN_M``. If several caps’ boxes apply, the cap
    whose open midline is **closest** to ``q`` supplies ``centroid_for_pick``.
    """
    import numpy as np
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        _SKIN_RAY_CAP_BBOX_MARGIN_M,
        _concat_skin_ring_vertices,
        _normal_2d_ccw,
        _pick_inward_ray_direction,
        _point_to_open_polyline_nearest,
        _skin_ring_vertex_tangents,
    )

    strips = shell_inputs.midlines
    cap_polys: list[Any] = []
    cap_centroids: list[Any] = []
    for st in strips:
        if st.kind != "cap":
            continue
        pts = np.asarray(st.midline_b, dtype=float)
        if pts.shape[0] < 2:
            continue
        cap_polys.append(pts)
        cap_centroids.append(np.mean(pts, axis=0).reshape(2))
    if not cap_polys:
        return

    skin_ring = _concat_skin_ring_vertices(strips)
    if skin_ring.shape[0] < 2:
        return
    tans = _skin_ring_vertex_tangents(skin_ring)
    mrg = float(_SKIN_RAY_CAP_BBOX_MARGIN_M)
    arrow_m = max(0.008, 0.012 * float(chord_m))

    xs: list[float] = []
    ys: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    for i in range(skin_ring.shape[0]):
        q = skin_ring[i]
        # Same shortlist as the resampler: only caps whose (expanded) AABB
        # contains q; if several, the cap with smallest distance to its midline
        # wins (picks c_mid for _pick_inward_ray_direction).
        best_d = float("inf")
        best_cmid = cap_centroids[0]
        found = False
        for cpts, c_mid in zip(cap_polys, cap_centroids, strict=True):
            xmin, xmax = float(cpts[:, 0].min()), float(cpts[:, 0].max())
            ymin, ymax = float(cpts[:, 1].min()), float(cpts[:, 1].max())
            if (
                q[0] < xmin - mrg
                or q[0] > xmax + mrg
                or q[1] < ymin - mrg
                or q[1] > ymax + mrg
            ):
                continue
            d, _ = _point_to_open_polyline_nearest(q, cpts)
            if d < best_d:
                best_d = float(d)
                best_cmid = c_mid
                found = True
        if not found:
            continue
        n_ccw = _normal_2d_ccw(tans[i])
        v = _pick_inward_ray_direction(q, n_ccw, best_cmid)
        xs.append(float(q[0]))
        ys.append(float(q[1]))
        us.append(float(v[0] * arrow_m))
        vs.append(float(v[1] * arrow_m))

    if not xs:
        return
    ax_mesh.quiver(
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(us, dtype=float),
        np.asarray(vs, dtype=float),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0024,
        color="#66ffcc",
        alpha=0.9,
        zorder=16,
        label="Class C ray dir (perp tangent, in cap bbox)",
    )


def _draw_mitc4_mesh_on_ax(
    ax_mesh: Any,
    mesh: Any,
    *,
    le_b: tuple[float, float],
    te_b: tuple[float, float],
    kind_colour: dict[str, str],
    title: str,
    shell_inputs: Any | None = None,
    chord_m: float | None = None,
) -> None:
    """Plot one :class:`Mitc4SectionMesh` (midlines, element edges, junctions)."""
    drawn_kinds: set[str] = set()
    for pm in mesh.panel_meshes:
        if pm.yz_nodes.size == 0:
            continue
        col = kind_colour.get(pm.kind, "#bbbbbb")
        legend = None
        if pm.kind not in drawn_kinds:
            drawn_kinds.add(pm.kind)
            legend = f"{pm.kind} midline"
        ax_mesh.plot(
            pm.yz_nodes[:, 0],
            pm.yz_nodes[:, 1],
            color=col,
            lw=1.4,
            alpha=0.55,
            zorder=3,
            label=legend,
        )

    elem_drawn: set[str] = set()
    for pm in mesh.panel_meshes:
        if pm.yz_nodes.size == 0:
            continue
        col = kind_colour.get(pm.kind, "#bbbbbb")
        for i in range(pm.yz_nodes.shape[0] - 1):
            x0, y0 = pm.yz_nodes[i]
            x1, y1 = pm.yz_nodes[i + 1]
            tag = f"{pm.kind}_elem"
            legend = None
            if tag not in elem_drawn:
                elem_drawn.add(tag)
                legend = f"{pm.kind} element"
            ax_mesh.plot(
                [x0, x1],
                [y0, y1],
                color=col,
                lw=2.4,
                alpha=0.95,
                zorder=5,
                solid_capstyle="round",
                label=legend,
            )
        ax_mesh.scatter(
            pm.yz_nodes[:, 0],
            pm.yz_nodes[:, 1],
            color=col,
            s=14,
            alpha=0.95,
            zorder=8,
            edgecolors="white",
            linewidths=0.4,
        )

    junction_colour_by_size = {2: "#2ecc71", 3: "#f39c12", 4: "#e67e22"}
    cluster_drawn: set[int] = set()
    for c in mesh.clusters:
        if len(c.members) < 2:
            continue
        size = _junction_display_valence(mesh, c)
        col = junction_colour_by_size.get(size, "#9b59b6")
        legend = None
        if size not in cluster_drawn:
            cluster_drawn.add(size)
            legend = f"junction (n={size})"
        ax_mesh.scatter(
            [c.point_yz[0]],
            [c.point_yz[1]],
            facecolors="none",
            edgecolors=col,
            s=180,
            lw=1.6,
            zorder=20,
            label=legend,
        )

    ax_mesh.scatter(
        [le_b[0], te_b[0]],
        [le_b[1], te_b[1]],
        color="#ffffff",
        s=60,
        marker="x",
        zorder=30,
        label="LE / TE",
    )

    if shell_inputs is not None and chord_m is not None and chord_m > 0.0:
        _add_skin_inward_rays_class_c_near_caps(
            ax_mesh, shell_inputs, chord_m=chord_m
        )

    summ = mesh.summary()
    junction_valence = Counter()
    for cl in mesh.clusters:
        if len(cl.members) < 2:
            continue
        junction_valence[_junction_display_valence(mesh, cl)] += 1
    ax_mesh.set_title(
        f"{title}\n{summ['n_panels']} panels · {summ['n_total_elements']} elements · "
        f"{summ['n_total_nodes']} nodes · {summ['n_clusters']} junctions "
        f"(by label n) {dict(sorted(junction_valence.items()))}",
        color="#e0e0e0",
        fontsize=10,
    )
    ax_mesh.legend(
        loc="lower right",
        fontsize=7,
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=2,
    )


def _taxonomy_plot_caption(layout_key: str, layout: Any) -> str:
    """Caption: taxonomy **X** / **Y** / **Z** (``layout_key`` middle letter matches **Y** = ``structural_family``)."""
    y = str(layout.structural_family)
    y_desc = {
        "A": "no spar caps",
        "B": "single cap band (pitch / max-t)",
        "C": "discrete caps per web",
        "D": "continuous box (first web to last web)",
    }.get(y, y)
    z = (
        "flapwise (B-frame vertical webs)"
        if layout.web_orientation == "flapwise"
        else "chord-normal (S-frame vertical webs)"
    )
    parts = [
        f"X = {layout.n_webs} web(s)",
        f"Y = {y} ({y_desc})",
        f"Z = {z}",
    ]
    return " | ".join(parts)


def _run_single_layout(
    *,
    layout_key: str,
    airfoil,
    twist_rad: float,
    twist_deg: float,
    out_dir: Path,
) -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from blade_precompute.contract.shell_midline_strip import ShellMidlineStrip
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_geometry.interface.shell_midline_export import (
        build_shell_midline_strips,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        ShellMeshInputs,
        SubcomponentMidline,
        build_shell_mesh_inputs,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)

    # Smoke: geometry-export path produces the same strips as the adapter.
    strips_geo = build_shell_midline_strips(
        section, twist_rad=twist_rad, n_web_samples=20, n_cap_samples=80
    )
    shell_inputs = build_shell_mesh_inputs(
        section=section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )

    _smoke_geo_export_vs_shell_inputs(strips_geo, shell_inputs, layout_key, np)
    assert isinstance(shell_inputs, ShellMeshInputs)
    assert all(isinstance(m, ShellMidlineStrip) for m in shell_inputs.midlines)
    assert SubcomponentMidline is ShellMidlineStrip

    endpoint_tol = 1e-2
    chord_m = float(shell_inputs.chord_m)
    target_len_m = _target_element_length_demo_m(chord_m)

    mesh = build_mitc4_mesh(
        shell_inputs,
        target_element_length_m=target_len_m,
        endpoint_tol=endpoint_tol,
    )

    fig_path = out_dir / f"mitc4_mesh_{layout_key}_i000.png"

    plt.style.use("dark_background")
    fig, (ax_geom, ax_mesh) = plt.subplots(
        1, 2, figsize=(18, 8), sharex=True, sharey=True
    )
    fig.patch.set_facecolor("#0f1117")
    for _ax in (ax_geom, ax_mesh):
        _ax.set_facecolor("#0f1117")

    kind_colour = {"skin": "#7eb8da", "web": "#e74c3c", "cap": "#f1c40f"}

    # ---- LEFT: section_geometry SDF (B-frame) ----
    airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil
    verts_b = np.asarray(airfoil_b.vertices, dtype=float)
    pad = 0.05 * float(max(np.ptp(verts_b[:, 0]), np.ptp(verts_b[:, 1])))
    x_min = float(verts_b[:, 0].min()) - pad
    x_max = float(verts_b[:, 0].max()) + pad
    y_min = float(verts_b[:, 1].min()) - pad
    y_max = float(verts_b[:, 1].max()) + pad
    nx_grid, ny_grid = 600, 360
    gx = np.linspace(x_min, x_max, nx_grid)
    gy = np.linspace(y_min, y_max, ny_grid)
    GX, GY = np.meshgrid(gx, gy)

    label_to_kind: dict[str, str] = {}
    for label in section.labels:
        s = str(label).lower()
        if "skin" in s:
            label_to_kind[label] = "skin"
        elif "web" in s:
            label_to_kind[label] = "web"
        elif "cap" in s:
            label_to_kind[label] = "cap"
    kind_alpha = {"skin": 0.30, "web": 0.55, "cap": 0.55}
    for label, kind in label_to_kind.items():
        try:
            phi = np.asarray(section[label](GX, GY), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        col = kind_colour[kind]
        ax_geom.contourf(
            GX,
            GY,
            phi,
            levels=[-1e9, 0.0],
            colors=[col],
            alpha=kind_alpha[kind],
            zorder=1,
        )
        ax_geom.contour(
            GX,
            GY,
            phi,
            levels=[0.0],
            colors=[col],
            linewidths=0.8,
            alpha=0.9,
            zorder=2,
        )

    le_b = shell_inputs.leading_edge_b
    te_b = shell_inputs.trailing_edge_b
    ax_geom.scatter(
        [le_b[0], te_b[0]],
        [le_b[1], te_b[1]],
        color="#ffffff",
        s=60,
        marker="x",
        zorder=30,
    )

    sdf_legend = [
        Patch(
            facecolor=kind_colour["skin"],
            alpha=kind_alpha["skin"],
            label="skin SDF",
        ),
        Patch(
            facecolor=kind_colour["cap"],
            alpha=kind_alpha["cap"],
            label="cap SDF",
        ),
        Patch(
            facecolor=kind_colour["web"],
            alpha=kind_alpha["web"],
            label="web SDF",
        ),
    ]
    ax_geom.legend(
        handles=sdf_legend,
        loc="lower right",
        fontsize=8,
        facecolor="#1a1a2e",
        edgecolor="#2a2a3a",
        labelcolor="#e0e0e0",
        ncol=1,
    )
    ax_geom.set_title(
        "section_geometry SDF (B-frame)",
        color="#e0e0e0",
        fontsize=11,
    )

    _draw_mitc4_mesh_on_ax(
        ax_mesh,
        mesh,
        le_b=le_b,
        te_b=te_b,
        kind_colour=kind_colour,
        title=f"MITC4 — target_element_length_m={target_len_m:.4g} m",
        shell_inputs=shell_inputs,
        chord_m=chord_m,
    )

    for _ax in (ax_geom, ax_mesh):
        _ax.set_xlabel("y_B [m]", color="#e0e0e0")
        _ax.set_ylabel("z_B [m]", color="#e0e0e0")
        _ax.tick_params(colors="#aaaaaa")
        _ax.set_aspect("equal")
        _ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)

    tax = _taxonomy_plot_caption(layout_key, layout)
    summ = mesh.summary()
    fig.suptitle(
        f"{layout_key} — section_geometry SDF vs MITC4 (target length)\n"
        f"{tax}\n"
        f"NACA63-415 · twist = {twist_deg:.2f}° · "
        f"{summ['n_total_elements']} elements",
        color="#e0e0e0",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(fig_path, dpi=160, facecolor="#0f1117")
    plt.close(fig)

    print(f"[mitc4_mesh_v2] wrote {fig_path}")
    print(
        f"[mitc4_mesh_v2] {layout_key}: target_len={target_len_m:.4g} m -> "
        f"{summ['n_total_elements']} elements -- {tax}"
    )


def main() -> None:
    _bootstrap_path()

    import sys

    import numpy as np

    from blade_precompute.orchestration.system_layout import resolve_system_type
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF

    chord_m = 1.655959896
    twist_deg = 34.110405083
    naca_series, naca_m, naca_p, naca_xx = 6, 63.0, 4.0, 15.0

    airfoil = AirfoilSDF.from_naca_series(
        naca_series,
        naca_m,
        naca_p,
        naca_xx,
        n_points=200,
        chord=chord_m,
        closed_te=True,
    )
    twist_rad = float(np.deg2rad(twist_deg))

    out_dir = Path(__file__).resolve().parent / "output" / "repro_mitc4_mesh_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = sys.argv[1:]
    if argv:
        layout_keys = [resolve_system_type(k.strip()).key for k in argv if k.strip()]
    else:
        layout_keys = list(_all_layout_keys())

    if not layout_keys:
        raise SystemExit("No layout keys to render.")

    rendered_png_names = {f"mitc4_mesh_{k}_i000.png" for k in layout_keys}
    stale_pngs = sorted(
        p.name for p in out_dir.glob("mitc4_mesh_*_i000.png") if p.name not in rendered_png_names
    )
    for name in stale_pngs:
        try:
            (out_dir / name).unlink()
        except Exception:
            pass

    # Expected full registry: 0A, 0B, plus X=1..5 × Y=A/B/C/D × Z=CN/F minus 1D-*.
    expected_keys = {"0A", "0B"} | {
        f"{x}{y}-{z}"
        for x in range(1, 6)
        for y in ("A", "B", "C", "D")
        for z in ("CN", "F")
        if not (x == 1 and y == "D")
    }
    rendered_set = set(layout_keys)
    coverage_manifest = {
        "rendered_keys": list(layout_keys),
        "missing_vs_full_registry": sorted(expected_keys - rendered_set),
        "extra_vs_full_registry": sorted(rendered_set - expected_keys),
        "excluded_invalid_keys": ["1D-CN", "1D-F"],
        "stale_pngs_removed": stale_pngs,
    }
    with open(out_dir / "taxonomy_coverage.json", "w", encoding="utf-8") as f:
        json.dump(coverage_manifest, f, indent=2)

    print(
        f"[mitc4_mesh_v2] rendering {len(layout_keys)} layout(s): "
        f"{', '.join(layout_keys)}"
    )
    for layout_key in layout_keys:
        _run_single_layout(
            layout_key=layout_key,
            airfoil=airfoil,
            twist_rad=twist_rad,
            twist_deg=twist_deg,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
