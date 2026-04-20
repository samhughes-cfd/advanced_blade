"""
examples/system_types_twist_grid.py
===================================
Visualise section geometry on a **3×4** grid (**chord-fraction presets** × **1…4 webs**)
and **two** web-alignment panels (**chord-normal** vs **flapwise**) at a fixed twist (default 20°).

Structural naming (``SystemType{X}{Y}-{Z}``) — *see also each case's* ``structural_system_type`` *in the summary JSON*
------------------------------------------------------------------
  **X** = number of webs.  **Y** = spar-cap family (none / fixed / box).  **Z** = **CN** (chord-normal) or **F** (flapwise).

  This script builds ``MultiCellSection`` with **continuous** upper/lower spar caps (box spar between outermost webs).
  So for **X ≥ 2**, cases match **SystemType{X}C-{Z}** (box spar).  For **X = 1**, the matrix lists **1B-{Z}** only (no 1C row);
  we label **SystemType1B-{Z}** here as the closest structural row (single web, “fixed” cap span).

  **Important:** Row letters **A / B / C** in this file are **only chord-fraction layouts** (where webs sit along the chord),
  **not** structural **Y** (spar configuration).  Do not confuse **preset C** (wide-spread fractions) with **structural Y=C** (box spar).

Chord-fraction presets (columns A / B / C — web positions, LE → TE)
------------------------------------------------------------------
* **A — uniform**: ``i / (n_webs + 1)`` for ``i = 1 … n_webs``.
* **B — fixed / legacy-style**: clustered toward LE/TE; n=2 matches legacy (0.15, 0.50).
* **C — wide-spread**: webs pushed toward outer chord positions.

Run from the repository root::

    python blade_precompute/section_geometry/examples/system_types_twist_grid.py --out-dir outputs/system_types_twist20

Writes ``system_types_grid_summary.json`` and, by default, one ``sections/<alignment>_<A|B|C>_n<k>.json``
per subplot (section properties, airfoil vertices, medial axes per component; use ``--with-contours`` for φ=0 polylines — slower).
Medial polylines need **scikit-image** (see ``section_geometry/requirements.txt``); without it, ``medial_axes`` lists are usually empty.

Requires: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Repo on sys.path when executed as a file
_erp = Path(__file__).resolve().parent / "_ensure_repo_path.py"
_spec = importlib.util.spec_from_file_location("_ensure_repo_path", _erp)
_ensure_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ensure_mod)
_ensure_mod.ensure_repo_path()

from blade_precompute.section_geometry.engine.implicit_section_geometry import (
    AirfoilSDF,
    MedialAxisExtractor,
    MultiCellSection,
    SDFGrid,
)
from blade_precompute.section_geometry.interface.export import export_section_json
from blade_precompute.section_geometry.interface.plot import (
    _component_color,
    plot_section,
)
from blade_precompute.section_geometry.medial import extractor as _med_ex


# region agent log
def _agent_debug_log(payload: dict) -> None:
    import time

    p = {**payload, "sessionId": "8d9952", "timestamp": int(time.time() * 1000)}
    try:
        _logp = Path(__file__).resolve().parents[3] / "debug-8d9952.log"
        with open(_logp, "a", encoding="utf-8") as _fh:
            _fh.write(json.dumps(p, default=str) + "\n")
    except Exception:
        pass


# endregion


def web_chord_fracs(system: str, n_webs: int) -> tuple[float, ...]:
    """Return sorted chord fractions in (0, 1) for n_webs ∈ {1,2,3,4}."""
    if n_webs < 1 or n_webs > 4:
        raise ValueError("n_webs must be 1…4")
    if system == "A":
        return tuple(i / (n_webs + 1) for i in range(1, n_webs + 1))
    if system == "B":
        presets: dict[int, tuple[float, ...]] = {
            1: (0.50,),
            2: (0.15, 0.50),
            3: (0.15, 0.45, 0.75),
            4: (0.12, 0.34, 0.56, 0.78),
        }
        return presets[n_webs]
    if system == "C":
        presets = {
            1: (0.42,),
            2: (0.30, 0.70),
            3: (0.22, 0.50, 0.78),
            4: (0.18, 0.38, 0.62, 0.82),
        }
        return presets[n_webs]
    raise ValueError(system)


def structural_system_type_name(n_webs: int, align_key: str) -> str:
    """Display name matching the structural matrix for this script's geometry.

    ``MultiCellSection`` uses continuous spar caps → structural **Y = C** (box) when ``n_webs >= 2``.
    For a single web, the published matrix lists **1B** only → **SystemType1B-{Z}**.
    """
    z = "CN" if align_key == "chord_normal" else "F"
    if n_webs == 1:
        return f"SystemType1B-{z}"
    return f"SystemType{n_webs}C-{z}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--twist-deg", type=float, default=20.0, help="Section twist (CCW).")
    p.add_argument("--chord", type=float, default=1.0, help="Chord [m].")
    p.add_argument("--naca", default="2412", help="4-digit NACA.")
    p.add_argument("--nx", type=int, default=280, help="Grid nx.")
    p.add_argument("--ny", type=int, default=120, help="Grid ny.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: examples/output/system_types_twist_grid).",
    )
    p.add_argument(
        "--no-section-jsons",
        action="store_true",
        help="Skip per-panel section_*.json (summary JSON is still written).",
    )
    p.add_argument(
        "--with-contours",
        action="store_true",
        help="Also include flat component_zero_contours (duplicate of geometry.components[*].boundary).",
    )
    p.add_argument(
        "--no-geometry-detail",
        action="store_true",
        help="Skip geometry.skin / geometry.components boundaries (faster; medials unchanged).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution for the grid figure (matplotlib savefig dpi). Default: 300.",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "output" / "system_types_twist_grid"
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chord = float(args.chord)
    twist_rad = float(np.deg2rad(args.twist_deg))
    airfoil = AirfoilSDF.from_naca(str(args.naca).zfill(4), chord=chord)
    airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-12 else airfoil
    grid = SDFGrid.from_airfoil(airfoil_b, nx=int(args.nx), ny=int(args.ny))

    systems = ("A", "B", "C")
    n_web_list = (1, 2, 3, 4)
    alignments: tuple[tuple[str, str], ...] = (
        ("chord_normal", "Chord-normal web axis"),
        ("flapwise", "Flapwise web axis"),
    )

    write_section_jsons = not args.no_section_jsons
    sections_dir = out_dir / "sections"
    if write_section_jsons:
        sections_dir.mkdir(parents=True, exist_ok=True)

    summary_cases: list[dict] = []

    fig = plt.figure(figsize=(22, 14))
    gs_outer = fig.add_gridspec(2, 1, hspace=0.14, height_ratios=[1.0, 1.0])
    fig.suptitle(
        f"System types A / B / C × web count — NACA {args.naca}, chord={chord:g} m, "
        f"twist={args.twist_deg:g}°",
        fontsize=14,
        y=0.98,
    )

    for panel, (align_key, panel_title) in enumerate(alignments):
        gs_in = gs_outer[panel].subgridspec(3, 4, hspace=0.38, wspace=0.22)
        fig.text(
            0.5,
            0.96 - panel * 0.48,
            panel_title,
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
        for r, letter in enumerate(systems):
            for c, n_webs in enumerate(n_web_list):
                ax = fig.add_subplot(gs_in[r, c])
                fracs = web_chord_fracs(letter, n_webs)
                xs = sorted(float(f) * chord for f in fracs)
                section = MultiCellSection(
                    airfoil_sdf=airfoil,
                    web_x_positions=xs,
                    web_thickness=0.004,
                    web_alignment=align_key,
                    cap_height=0.012,
                    skin_thickness=0.003,
                    twist_angle=twist_rad,
                    core_enabled=True,
                )

                json_rel: str | None = None
                json_err: str | None = None
                if write_section_jsons:
                    fname = f"{align_key}_{letter}_n{n_webs}.json"
                    jpath = sections_dir / fname
                    try:
                        midlines = MedialAxisExtractor(
                            grid,
                            grad_threshold=0.92,
                            min_branch_pixels=2,
                        ).extract_for_section(section)
                        # region agent log
                        _agent_debug_log(
                            {
                                "hypothesisId": "H1",
                                "location": "system_types_twist_grid.py:export_section_json",
                                "message": "medial axes extracted for section JSON",
                                "runId": "post-fix-medials",
                                "data": {
                                    "case": f"{align_key}_{letter}_n{n_webs}",
                                    "skimage_available": bool(_med_ex._HAS_SKIMAGE),
                                    "n_labels": len(midlines),
                                    "branches_per_label": {
                                        k: len(v) for k, v in midlines.items()
                                    },
                                    "total_polylines": sum(
                                        len(v) for v in midlines.values()
                                    ),
                                    "nonempty_labels": sum(
                                        1 for v in midlines.values() if len(v) > 0
                                    ),
                                },
                            }
                        )
                        # endregion
                        export_section_json(
                            section,
                            grid,
                            midlines,
                            str(jpath),
                            include_component_zero_contours=args.with_contours,
                            include_geometry_detail=not args.no_geometry_detail,
                        )
                        json_rel = f"sections/{fname}"
                    except Exception as exc:
                        json_err = str(exc)

                stype = structural_system_type_name(n_webs, align_key)
                summary_cases.append(
                    {
                        "alignment": align_key,
                        "panel_title": panel_title,
                        "structural_system_type": stype,
                        "chord_fraction_preset": letter,
                        "system": letter,
                        "n_webs": n_webs,
                        "chord_fracs": [float(x) for x in fracs],
                        "web_x_m": [float(x) for x in xs],
                        "labels": section.labels,
                        "n_cells": section.n_cells,
                        "section_json": json_rel,
                        "export_error": json_err,
                    }
                )

                desc = {
                    "A": "uniform",
                    "B": "fixed / legacy-style",
                    "C": "wide-spread",
                }[letter]
                title = (
                    f"{stype}\n"
                    f"Preset {letter} ({desc})  |  {n_webs} web(s)  {fracs}"
                )
                plot_section(
                    section,
                    grid,
                    ax=ax,
                    alpha=0.48,
                    show_airfoil=False,
                    title=title,
                    show_legend=False,
                )
                ax.tick_params(labelsize=7)
                if c == 0:
                    ax.set_ylabel(
                        f"Preset {letter}\n(chord layout)",
                        fontsize=8,
                    )

    # One shared legend (first section's labels — same for all)
    from matplotlib.patches import Patch

    demo = MultiCellSection(
        airfoil_sdf=airfoil,
        web_x_positions=[0.5 * chord],
        web_thickness=0.004,
        web_alignment="chord_normal",
        cap_height=0.012,
        skin_thickness=0.003,
        twist_angle=twist_rad,
        core_enabled=True,
    )
    handles = [
        Patch(facecolor=_component_color(lbl), alpha=0.48, label=lbl)
        for lbl in demo.labels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    plt.subplots_adjust(bottom=0.08)

    out_png = out_dir / f"system_types_ABC_twist{args.twist_deg:g}deg.png"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    summary = {
        "twist_deg": float(args.twist_deg),
        "twist_rad": float(twist_rad),
        "naca": str(args.naca).zfill(4),
        "chord_m": chord,
        "grid": {"nx": int(args.nx), "ny": int(args.ny)},
        "figure_png_dpi": int(args.dpi),
        "structural_naming": {
            "pattern": "SystemType{X}{Y}-{Z}",
            "X": "web count",
            "Y": "spar family (this script: continuous caps ⇒ C for X≥2; single web labeled 1B per matrix)",
            "Z": "CN = chord_normal alignment, F = flapwise",
            "chord_fraction_presets_ABC": "Only web chord positions — not structural Y.",
        },
        "medial_axes_export": {
            "grad_threshold": 0.92,
            "min_branch_pixels": 2,
            "scikit_image_available": bool(_med_ex._HAS_SKIMAGE),
        },
        "system_presets": {
            "A": "uniform i/(n+1)",
            "B": "fixed / legacy-style chord fractions",
            "C": "wide-spread chord fractions",
        },
        "include_component_zero_contours": bool(args.with_contours),
        "include_geometry_detail": not bool(args.no_geometry_detail),
        "cases": summary_cases,
    }
    summary_path = out_dir / "system_types_grid_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {summary_path}")
    if write_section_jsons:
        n_ok = sum(1 for c in summary_cases if c.get("section_json") and not c.get("export_error"))
        n_bad = len(summary_cases) - n_ok
        print(f"Section JSONs: {n_ok} ok, {n_bad} failed (see export_error in summary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
