"""
Evidence-gathering repro for the section_shell_model refactor (option C).

Mirrors the inputs used by ``section_geometry_impl`` for run007 station i000
(z=0, chord=1.66 m, twist=34.11 deg, NACA63-415, layout 2D-F — flapwise webs)
and runs::

    AirfoilSDF.from_naca_series → build_section_view (with twist) →
    SDFGrid.from_airfoil(airfoil_b) → MedialAxisExtractor.extract_for_section

For each labelled subcomponent (outer_skin, web_0, web_1, spar_cap_*, core_*)
we log: the polyline count, total arc length, x-range, y-range, first/last
points, and (when applicable) per-polyline orientation. We also save a single
PNG plotting all midlines on top of the rotated airfoil contour, so we can
visually compare against
``outputs/.../section_geometry/station_i000_z0.000/section_i000_rz0.000.png``.

This gives the runtime ground-truth needed before refactoring
``section_shell_model_impl`` to consume midlines instead of rebuilding NACA
panels from scratch.

Usage (from repo root)::

    python blade_precompute/section_shell_model/examples/repro_geometry_to_midlines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


_LOG_PATH = Path(__file__).resolve().parents[3] / "debug-55cddb.log"
_SESSION_ID = "55cddb"


def _log(payload: dict) -> None:
    entry = {
        "sessionId": _SESSION_ID,
        "timestamp": int(time.time() * 1000),
        **payload,
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    _bootstrap_path()

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.section_geometry.geometry.grid import SDFGrid
    from blade_precompute.section_geometry.medial.extractor import MedialAxisExtractor
    from blade_precompute.section_geometry.sections.subcomponents import ShearWeb
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )

    def _rot(pts, angle):
        c, s = np.cos(angle), np.sin(angle)
        a = np.asarray(pts, dtype=float).reshape(-1, 2)
        out = np.column_stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1]])
        return out

    def _analytical_web_midline_b(web: "ShearWeb", twist_rad: float, n: int = 32) -> np.ndarray:
        x_top_S = float(web.x_top); y_top_S = float(web.y_top)
        x_bot_S = float(web.x_bot); y_bot_S = float(web.y_bot)
        if web.alignment == "flapwise" and abs(twist_rad) > 1e-10:
            cx = 0.5 * (x_top_S + x_bot_S); cy = 0.5 * (y_top_S + y_bot_S)
            top_S = np.array([x_top_S, y_top_S]) - np.array([cx, cy])
            bot_S = np.array([x_bot_S, y_bot_S]) - np.array([cx, cy])
            top_S = _rot(top_S, -twist_rad)[0] + np.array([cx, cy])
            bot_S = _rot(bot_S, -twist_rad)[0] + np.array([cx, cy])
        else:
            top_S = np.array([x_top_S, y_top_S]); bot_S = np.array([x_bot_S, y_bot_S])
        line_S = np.linspace(top_S, bot_S, n)
        return _rot(line_S, twist_rad) if abs(twist_rad) > 1e-10 else line_S

    # ---- inputs mirroring run007 station i000 (z=0) ----
    chord_m = 1.655959896
    twist_deg = 34.110405083
    naca_series, naca_m, naca_p, naca_xx = 6, 63.0, 4.0, 15.0
    layout_key = "2D-F"
    layout = resolve_system_type(layout_key)
    web_chord_fracs = tuple(layout.web_chord_fracs)

    # ---- build airfoil + section view (same as section_geometry stage) ----
    airfoil = AirfoilSDF.from_naca_series(
        naca_series, naca_m, naca_p, naca_xx, n_points=200, chord=chord_m, closed_te=True
    )
    twist_rad = float(np.deg2rad(twist_deg))
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil

    # Use the same grid resolution as section_geometry_impl (props_sdf_*).
    grid = SDFGrid.from_airfoil(airfoil_b, nx=384, ny=180)

    labels = list(getattr(section, "labels", list(section)))
    _log(
        {
            "runId": "geom-midline-evidence",
            "hypothesisId": "H7",
            "location": "repro_geometry_to_midlines.py:setup",
            "message": "section_geometry replica setup complete",
            "data": {
                "layout_key": layout_key,
                "n_webs": int(layout.n_webs),
                "web_orientation": str(layout.web_orientation),
                "geometry_mode": str(layout.geometry_mode),
                "web_chord_fracs": list(web_chord_fracs),
                "chord_m": chord_m,
                "twist_deg": twist_deg,
                "naca_series": naca_series,
                "labels": labels,
                "grid_nx": int(grid.X.shape[1]),
                "grid_ny": int(grid.X.shape[0]),
                "grid_dx_m": float(grid.dx),
                "grid_dy_m": float(grid.dy),
                "grid_x_min": float(grid.X[0, 0]),
                "grid_x_max": float(grid.X[0, -1]),
                "grid_y_min": float(grid.Y[0, 0]),
                "grid_y_max": float(grid.Y[-1, 0]),
            },
        }
    )

    # ---- H8: production-resolution medial extraction (current behaviour) ----
    extractor = MedialAxisExtractor(grid, grad_threshold=0.95, min_branch_pixels=10)
    midlines = extractor.extract_for_section(section, labels=labels)

    # ---- H8b: high-resolution medial extraction (does cranking grid help?) ----
    grid_hi = SDFGrid.from_airfoil(airfoil_b, nx=1500, ny=600)
    extractor_hi = MedialAxisExtractor(grid_hi, grad_threshold=0.92, min_branch_pixels=4)
    midlines_hi = extractor_hi.extract_for_section(section, labels=labels)
    _log(
        {
            "runId": "geom-midline-evidence",
            "hypothesisId": "H8b",
            "location": "repro_geometry_to_midlines.py:hi_res_medial",
            "message": "High-resolution medial extraction summary",
            "data": {
                "grid_nx": int(grid_hi.X.shape[1]),
                "grid_ny": int(grid_hi.X.shape[0]),
                "dx_m": float(grid_hi.dx),
                "dy_m": float(grid_hi.dy),
                "per_label_n_polylines": {
                    lbl: len(midlines_hi.get(lbl, [])) for lbl in labels
                },
                "per_label_total_pts": {
                    lbl: int(sum(len(p) for p in midlines_hi.get(lbl, [])))
                    for lbl in labels
                },
            },
        }
    )

    # ---- H9: analytical reconstruction from MultiCellSection internals ----
    components_S = getattr(section, "_components_unrotated", None)
    web_axes_b: dict[str, np.ndarray] = {}
    if components_S is not None:
        for lbl, comp in components_S.items():
            if isinstance(comp, ShearWeb):
                web_axes_b[lbl] = _analytical_web_midline_b(comp, twist_rad)

    skin_outer_b = np.asarray(getattr(airfoil_b, "vertices"), dtype=float)
    skin_outer_b_closed = np.vstack([skin_outer_b, skin_outer_b[:1]])
    inner_offset_m = float(0.5 * 0.012)  # nominal skin midline offset; refined later
    skin_inner_S = np.asarray(getattr(airfoil, "vertices"), dtype=float)
    centroid_S = skin_inner_S.mean(axis=0)
    inward = centroid_S - skin_inner_S
    inward /= np.linalg.norm(inward, axis=1, keepdims=True) + 1e-30
    skin_mid_S = skin_inner_S + inward * inner_offset_m
    skin_mid_b = _rot(skin_mid_S, twist_rad) if abs(twist_rad) > 1e-10 else skin_mid_S

    _log(
        {
            "runId": "geom-midline-evidence",
            "hypothesisId": "H9",
            "location": "repro_geometry_to_midlines.py:analytical",
            "message": "Analytical midline reconstruction summary",
            "data": {
                "n_webs_analytical": len(web_axes_b),
                "web_endpoints_b": {
                    lbl: {
                        "top_b": [float(arr[0, 0]), float(arr[0, 1])],
                        "bot_b": [float(arr[-1, 0]), float(arr[-1, 1])],
                        "axis_b": [
                            float(arr[-1, 0] - arr[0, 0]),
                            float(arr[-1, 1] - arr[0, 1]),
                        ],
                    }
                    for lbl, arr in web_axes_b.items()
                },
                "skin_outer_npts_b": int(skin_outer_b.shape[0]),
                "skin_mid_offset_m_used": inner_offset_m,
            },
        }
    )

    # ---- H10: validate the adapter (build_shell_mesh_inputs) ----
    # ShellMeshInputs now exposes midlines: list[ShellMidlineStrip] only.
    # Helper accessors derived from midlines:
    shell_inputs = build_shell_mesh_inputs(
        section, twist_rad=twist_rad, layout_key=layout_key, n_cap_samples=80
    )

    web_strips = [m for m in shell_inputs.midlines if m.kind == "web"]
    cap_strips = [m for m in shell_inputs.midlines if m.kind == "cap"]
    skin_strips = [m for m in shell_inputs.midlines if m.kind == "skin"]

    # Adapter web endpoints from ShellMidlineStrip (top = midline_b[0], bot = midline_b[-1]).
    adapter_web_endpoints = {
        w.label: {
            "top_b": [float(w.midline_b[0, 0]), float(w.midline_b[0, 1])],
            "bot_b": [float(w.midline_b[-1, 0]), float(w.midline_b[-1, 1])],
        }
        for w in web_strips
    }
    h9_web_endpoints = {
        lbl: {
            "top_b": [float(arr[0, 0]), float(arr[0, 1])],
            "bot_b": [float(arr[-1, 0]), float(arr[-1, 1])],
        }
        for lbl, arr in web_axes_b.items()
    }
    web_diff_max_m = 0.0
    for lbl, ep in h9_web_endpoints.items():
        if lbl in adapter_web_endpoints:
            ad = adapter_web_endpoints[lbl]
            web_diff_max_m = max(
                web_diff_max_m,
                float(np.linalg.norm(np.array(ep["top_b"]) - np.array(ad["top_b"]))),
                float(np.linalg.norm(np.array(ep["bot_b"]) - np.array(ad["bot_b"]))),
            )

    # Skin midline from ShellMidlineStrip.
    skin_midline_b = skin_strips[0].midline_b if skin_strips else np.zeros((0, 2))
    skin_diff_rms = float(
        np.sqrt(
            np.mean(
                np.sum((skin_midline_b - skin_mid_b) ** 2, axis=1)
            )
        )
    ) if skin_midline_b.shape[0] > 0 else float("nan")

    _log(
        {
            "runId": "geom-midline-evidence",
            "hypothesisId": "H10",
            "location": "repro_geometry_to_midlines.py:adapter_validation",
            "message": "build_shell_mesh_inputs adapter validation",
            "data": {
                "adapter_chord_m": float(shell_inputs.chord_m),
                "adapter_twist_rad": float(shell_inputs.twist_rad),
                "adapter_layout_key": shell_inputs.layout_key,
                "adapter_skin_thickness_m": float(skin_strips[0].thickness_m) if skin_strips else None,
                "adapter_skin_mid_npts": int(skin_midline_b.shape[0]),
                "adapter_n_webs": len(web_strips),
                "adapter_n_cap_segments": len(cap_strips),
                "adapter_web_endpoints": adapter_web_endpoints,
                "h9_web_endpoints": h9_web_endpoints,
                "web_endpoint_max_diff_m": web_diff_max_m,
                "skin_midline_vs_h9_rms_m": skin_diff_rms,
                "cap_segments": [
                    {
                        "label": cs.label,
                        "surface": cs.surface,
                        "n_pts": int(cs.midline_b.shape[0]),
                        "cap_height_m": float(cs.thickness_m),
                        "alignment": str(cs.alignment),
                        "x_b_range": [
                            float(cs.midline_b[:, 0].min()),
                            float(cs.midline_b[:, 0].max()),
                        ],
                        "y_b_range": [
                            float(cs.midline_b[:, 1].min()),
                            float(cs.midline_b[:, 1].max()),
                        ],
                    }
                    for cs in cap_strips
                ],
            },
        }
    )

    # ---- H11: post-fix cap alignment check ----
    # The flapwise cap should align with bracketing flapwise web verticals
    # in the B-frame (x_B(cap_lo) == x_B(web_first), x_B(cap_hi) == x_B(web_last)).
    expected_lo = float(web_strips[0].midline_b[0, 0]) if web_strips else None
    expected_hi = float(web_strips[-1].midline_b[0, 0]) if web_strips else None
    cap_alignment_diag = []
    for cs in cap_strips:
        x_lo = float(cs.midline_b[:, 0].min())
        x_hi = float(cs.midline_b[:, 0].max())
        diag = {
            "label": cs.label,
            "surface": cs.surface,
            "alignment": str(cs.alignment),
            "x_b_lo_actual": x_lo,
            "x_b_hi_actual": x_hi,
            "x_b_lo_expected": expected_lo,
            "x_b_hi_expected": expected_hi,
        }
        if expected_lo is not None and expected_hi is not None:
            diag["x_b_lo_err_m"] = float(abs(x_lo - expected_lo))
            diag["x_b_hi_err_m"] = float(abs(x_hi - expected_hi))
            diag["max_extent_err_m"] = float(
                max(abs(x_lo - expected_lo), abs(x_hi - expected_hi))
            )
        cap_alignment_diag.append(diag)
    _log(
        {
            "runId": "post-fix",
            "hypothesisId": "H11",
            "location": "repro_geometry_to_midlines.py:cap_alignment_check",
            "message": "Post-fix flapwise cap alignment vs bracketing webs",
            "data": {
                "section_alignment": web_strips[0].alignment if web_strips else "n/a",
                "expected_x_b_lo": expected_lo,
                "expected_x_b_hi": expected_hi,
                "cap_segments": cap_alignment_diag,
            },
        }
    )

    # ---- log per-label diagnostics ----
    summary: dict[str, dict] = {}
    for lbl in labels:
        polys = midlines.get(lbl, [])
        plies = []
        for k, poly in enumerate(polys):
            arr = np.asarray(poly, dtype=float)
            if arr.size == 0:
                plies.append({"k": k, "n_pts": 0})
                continue
            diffs = np.diff(arr, axis=0)
            length = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
            plies.append(
                {
                    "k": k,
                    "n_pts": int(arr.shape[0]),
                    "length_m": length,
                    "x_min": float(arr[:, 0].min()),
                    "x_max": float(arr[:, 0].max()),
                    "y_min": float(arr[:, 1].min()),
                    "y_max": float(arr[:, 1].max()),
                    "first_pt": [float(arr[0, 0]), float(arr[0, 1])],
                    "last_pt": [float(arr[-1, 0]), float(arr[-1, 1])],
                }
            )
        summary[lbl] = {"n_polylines": len(polys), "polylines": plies}

    _log(
        {
            "runId": "geom-midline-evidence",
            "hypothesisId": "H7",
            "location": "repro_geometry_to_midlines.py:midline_summary",
            "message": "MedialAxisExtractor results per label",
            "data": {"layout_key": layout_key, "per_label": summary},
        }
    )

    # ---- visualise side-by-side: production medial | high-res medial | analytical ----
    out_dir = Path(__file__).resolve().parent / "output" / "repro_geometry_to_midlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"midlines_{layout_key}_i000.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0f1117")
    titles = [
        f"H8: medial @ ({grid.X.shape[1]}x{grid.X.shape[0]})",
        f"H8b: medial @ ({grid_hi.X.shape[1]}x{grid_hi.X.shape[0]})",
        "H9: analytical (ShearWeb attrs + airfoil)",
    ]
    cmap = plt.get_cmap("tab20")

    def _draw_outline(ax):
        ax.set_facecolor("#0f1117")
        ax.plot(skin_outer_b_closed[:, 0], skin_outer_b_closed[:, 1],
                color="#5dade2", lw=1.0, alpha=0.8, label="rotated airfoil contour")
        ax.set_aspect("equal")
        ax.set_xlabel("x_B [m]", color="#e0e0e0")
        ax.set_ylabel("y_B [m]", color="#e0e0e0")
        ax.tick_params(colors="#bbbbbb")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")
        ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)

    def _draw_medial(ax, src):
        for li, lbl in enumerate(labels):
            polys = src.get(lbl, [])
            col = cmap(li % 20)
            for poly in polys:
                arr = np.asarray(poly, dtype=float)
                if arr.size == 0:
                    continue
                ax.plot(arr[:, 0], arr[:, 1], color=col, lw=1.6, marker=".", ms=2.0, label=lbl)

    _draw_outline(axes[0]); _draw_medial(axes[0], midlines)
    _draw_outline(axes[1]); _draw_medial(axes[1], midlines_hi)
    _draw_outline(axes[2])
    axes[2].plot(skin_mid_b[:, 0], skin_mid_b[:, 1],
                 color="#f5b041", lw=1.6, label="outer_skin (H9 analytical mid)")
    for li, (lbl, arr) in enumerate(web_axes_b.items()):
        axes[2].plot(arr[:, 0], arr[:, 1], color=cmap(li % 20), lw=2.4, marker="o", ms=3.0,
                     label=f"{lbl} (H9)")
    # Overlay H10 adapter outputs (should sit exactly on top of H9).
    # Uses ShellMidlineStrip.midline_b directly (post-PR1 API).
    axes[2].plot(skin_midline_b[:, 0], skin_midline_b[:, 1],
                 color="#ffffff", lw=0.7, ls="--", alpha=0.9, label="adapter skin_mid")
    for w in web_strips:
        axes[2].plot(w.midline_b[:, 0], w.midline_b[:, 1],
                     color="#ffffff", lw=0.9, ls="--", marker="x", ms=4, alpha=0.9,
                     label=f"{w.label} (adapter)")
    for cs in cap_strips:
        axes[2].plot(cs.midline_b[:, 0], cs.midline_b[:, 1],
                     color="#27ae60", lw=2.0, label=f"{cs.label} (adapter)")
    # Mark the expected cap B-frame x extent with vertical guide lines.
    if web_strips:
        x_guide_lo = float(web_strips[0].midline_b[0, 0])
        x_guide_hi = float(web_strips[-1].midline_b[0, 0])
        for xg, lab in [(x_guide_lo, "web0 x_B"), (x_guide_hi, "webN x_B")]:
            axes[2].axvline(xg, color="#e74c3c", ls=":", lw=0.8, alpha=0.7,
                            label=f"{lab}={xg:.3f}")

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="#e0e0e0", fontsize=10)
        h, l = ax.get_legend_handles_labels()
        seen = set()
        uniq = [(hh, ll) for hh, ll in zip(h, l) if not (ll in seen or seen.add(ll))]
        ax.legend([hh for hh, _ in uniq], [ll for _, ll in uniq], fontsize=6, loc="upper right")

    fig.suptitle(f"midline strategies — layout={layout_key}, twist={twist_deg:.1f}°, chord={chord_m:.3f} m",
                 color="#e0e0e0", fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)

    print(f"Repro complete. Log: {_LOG_PATH}")
    print(f"PNG : {out_png}")


if __name__ == "__main__":
    main()
