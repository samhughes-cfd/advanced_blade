"""
Smoke test for the per-subcomponent midline architecture.

Verifies that:
    1. ``build_shell_mesh_inputs`` returns one :class:`SubcomponentMidline`
       per ``MultiCellSection`` subcomponent (skin, webs, caps).
    2. ``build_section_v2`` emits one :class:`Panel` per midline with the
       correct laminate kind.

Reuses the same run007 / station i000 / NACA63-415 / 2D-F inputs as
``repro_geometry_to_midlines.py`` so the panel topology can be visually
compared against the section_geometry output.

Logs (NDJSON, debug-55cddb.log):
    H13 — per-subcomponent midline summary from ``build_shell_mesh_inputs``
          (one record per skin/web/cap with kind, label, n_pts, arc length,
          x_b/y_b range, thickness, alignment/surface tag).
    H14 — per-Panel summary from ``build_section_v2`` so we can confirm
          a 1:1 correspondence between SubcomponentMidline records and
          emitted Panels.

Outputs (PNG):
    blade_precompute/section_shell_model/examples/output/repro_topology_v2/
    panels_<layout>_i<idx>.png

Each panel is plotted with a hue tied to its kind (skin / web / cap) and
endpoints marked so it is obvious that webs and caps are independent
shells stacked on top of the (closed) skin midline.
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


# #region agent log
def _log(payload: dict) -> None:
    entry = {
        "sessionId": _SESSION_ID,
        "timestamp": int(time.time() * 1000),
        **payload,
    }
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
# #endregion


def main() -> None:
    _bootstrap_path()

    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.orchestration.system_layout import (
        build_section_view,
        resolve_system_type,
    )
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
        build_shell_mesh_inputs,
    )
    from blade_precompute.section_shell_model.lib.topology_v2 import (
        build_section_v2,
    )

    chord_m = 1.655959896
    twist_deg = 34.110405083
    naca_series, naca_m, naca_p, naca_xx = 6, 63.0, 4.0, 15.0
    layout_key = "2D-F"
    layout = resolve_system_type(layout_key)

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
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)

    shell_inputs = build_shell_mesh_inputs(
        section=section,
        twist_rad=twist_rad,
        layout_key=layout_key,
    )

    # #region agent log
    midline_records = [
        {
            "label": m.label,
            "kind": m.kind,
            "n_pts": int(m.midline_b.shape[0]),
            "arc_length_m": float(m.arc_length_m()),
            "x_b_range": [float(m.midline_b[:, 0].min()), float(m.midline_b[:, 0].max())],
            "y_b_range": [float(m.midline_b[:, 1].min()), float(m.midline_b[:, 1].max())],
            "thickness_m": float(m.thickness_m),
            "closed": bool(m.closed),
            "surface": m.surface,
            "alignment": m.alignment,
        }
        for m in shell_inputs.midlines
    ]
    _log({
        "runId": "topology-v2-smoke",
        "hypothesisId": "H13",
        "location": "repro_topology_v2.py:per_subcomponent_midlines",
        "message": "build_shell_mesh_inputs per-subcomponent midline summary",
        "data": {
            "layout_key": layout_key,
            "chord_m": float(shell_inputs.chord_m),
            "twist_rad": float(shell_inputs.twist_rad),
            "le_b": list(map(float, shell_inputs.leading_edge_b)),
            "te_b": list(map(float, shell_inputs.trailing_edge_b)),
            "n_midlines": len(shell_inputs.midlines),
            "kinds": [m.kind for m in shell_inputs.midlines],
            "labels": [m.label for m in shell_inputs.midlines],
            "midlines": midline_records,
        },
    })
    # #endregion

    panels, booms, webs_geom, n_cells, diag = build_section_v2(
        shell_inputs, return_diagnostics=True
    )

    # #region agent log
    _log({
        "runId": "topology-v2-smoke",
        "hypothesisId": "H14",
        "location": "repro_topology_v2.py:panel_summary",
        "message": "build_section_v2 panel summary (one Panel per midline)",
        "data": {
            "layout_key": layout_key,
            "n_panels_total": len(panels),
            "n_booms": len(booms),
            "n_webs_geom": len(webs_geom),
            "n_cells": int(n_cells),
            "panel_summary": diag.panel_summary,
        },
    })
    # #endregion

    out_dir = Path(__file__).resolve().parent / "output" / "repro_topology_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"panels_{layout_key}_i000.png"

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    kind_style = {
        "skin": dict(color="#7eb8da", lw=2.0, ls="-", alpha=0.85, zorder=4),
        "web":  dict(color="#e74c3c", lw=2.4, ls="--", alpha=0.95, zorder=5),
        "cap":  dict(color="#f1c40f", lw=3.4, ls="-", alpha=0.95, zorder=6),
    }
    drawn_kinds: set[str] = set()
    for p in panels:
        kind = p.label.split(":", 1)[0]
        style = kind_style.get(kind, dict(color="#bbbbbb", lw=1.5, ls="-", alpha=0.7))
        legend_label = None
        if kind not in drawn_kinds:
            legend_label = f"{kind} (n={sum(1 for q in panels if q.label.startswith(kind + ':'))})"
            drawn_kinds.add(kind)
        ax.plot(p.nodes[:, 0], p.nodes[:, 1], label=legend_label, **style)
        ax.scatter([p.nodes[0, 0], p.nodes[-1, 0]],
                   [p.nodes[0, 1], p.nodes[-1, 1]],
                   color=style["color"], s=14, alpha=0.9, zorder=10,
                   edgecolors="white", linewidths=0.4)

    le_b = shell_inputs.leading_edge_b
    te_b = shell_inputs.trailing_edge_b
    ax.scatter([le_b[0], te_b[0]], [le_b[1], te_b[1]],
               color="#ffffff", s=60, marker="x", zorder=20,
               label="LE / TE (B-frame)")

    ax.set_aspect("equal")
    ax.set_title(
        f"per-subcomponent midlines → Panels — {layout_key} "
        f"(run007 i000, twist={twist_deg:.2f}°)\n"
        f"{len(panels)} panels (1 per subcomponent), {n_cells} cells (diagnostic)",
        color="#e0e0e0", fontsize=10,
    )
    ax.set_xlabel("x_B [m]", color="#e0e0e0")
    ax.set_ylabel("y_B [m]", color="#e0e0e0")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a2e",
              edgecolor="#2a2a3a", labelcolor="#e0e0e0")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, facecolor="#0f1117")
    plt.close(fig)

    print(f"[topology_v2] wrote {fig_path}")
    print(
        f"[topology_v2] {len(panels)} panels (1 per subcomponent), "
        f"{n_cells} cells, kinds={sorted({p.label.split(':',1)[0] for p in panels})}"
    )


if __name__ == "__main__":
    main()
