"""
examples/geometry_from_design_vector.py
=======================================
Build and plot 2-D section **computational geometry** from laminate thicknesses
and layout only — no structural / aerodynamic solves.

Design-like inputs (all lengths in metres unless noted):
  - NACA 4-digit code, chord
  - skin thickness, web thickness(es), upper/lower spar-cap heights
  - web x-positions as fractions of chord
  - optional section twist (degrees)

Run (any working directory; path is resolved from this file)::

    python path/to/blade_precompute/section_geometry/examples/geometry_from_design_vector.py --help

Or install once from the repo root: ``pip install -e .``

Requires: numpy, matplotlib
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

# ``import blade_precompute`` needs repo root on sys.path when run as a script.
_erp = Path(__file__).resolve().parent / "_ensure_repo_path.py"
if not _erp.is_file():
    raise RuntimeError(f"Missing helper {_erp.name}; run from the repository checkout.")
_spec = importlib.util.spec_from_file_location("_ensure_repo_path", _erp)
_ensure_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ensure_mod)
_ensure_mod.ensure_repo_path()


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate MultiCellSection geometry from thickness/layout parameters.",
    )
    p.add_argument("--naca", default="2412", help="4-digit NACA code (default 2412).")
    p.add_argument("--chord", type=float, default=1.0, help="Chord length [m].")
    p.add_argument("--skin-thickness", type=float, default=0.003, help="Skin laminate thickness [m].")
    p.add_argument(
        "--web-thickness",
        default="0.004",
        help="Web laminate thickness [m]: one value or comma-separated per web.",
    )
    p.add_argument(
        "--web-x-fracs",
        default="0.20,0.50",
        help="Comma-separated web x-positions as fractions of chord (sorted).",
    )
    p.add_argument(
        "--cap-height",
        default="0.014,0.012",
        help="Spar cap heights [m] as 'upper,lower' or one value for both.",
    )
    p.add_argument("--twist-deg", type=float, default=0.0, help="Section twist [deg], CCW.")
    p.add_argument("--web-alignment", default="chord_normal", choices=("chord_normal", "flapwise"))
    p.add_argument(
        "--structural-family",
        default="D",
        choices=("A", "B", "C", "D"),
        help="Structural family Y (SystemType{{X}}{{Y}}): A no caps, B fixed band, C per-web, D continuous box.",
    )
    p.add_argument(
        "--fixed-cap-anchor",
        default="pitching",
        choices=("pitching", "max_thickness"),
        help="For structural-family B: anchor chord station for the spar-cap band.",
    )
    p.add_argument(
        "--pitch-fraction-of-chord-from-le",
        type=float,
        default=1.0 / 3.0,
        help="For B + pitching anchor: x = fraction * chord from LE.",
    )
    p.add_argument(
        "--fixed-cap-chord-half-width",
        type=float,
        default=None,
        help="For B: half chordwise width of cap band [m]; default 0.05*chord.",
    )
    p.add_argument(
        "--discrete-cap-chord-half-width",
        type=float,
        default=None,
        help="For C: half width per web cap band [m]; default 0.04*chord.",
    )
    p.add_argument("--te-insert-x-frac", type=float, default=None, help="TE insert start x/chord; omit to disable.")
    p.add_argument("--le-insert-x-frac", type=float, default=None, help="LE insert end x/chord; omit to disable.")
    p.add_argument("--no-core", action="store_true", help="Disable sandwich core regions.")
    p.add_argument("--nx", type=int, default=512, help="Grid resolution x.")
    p.add_argument("--ny", type=int, default=220, help="Grid resolution y.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: examples/output/design_vector).",
    )
    p.add_argument("--json", action="store_true", help="Write design_vector.json next to the PNG.")
    p.add_argument(
        "--no-contours",
        action="store_true",
        help="Omit component_zero_contours from section_minimal.json (smaller file).",
    )
    args = p.parse_args(argv)

    from blade_precompute.section_geometry.engine.implicit_section_geometry import (
        AirfoilSDF,
        MultiCellSection,
        SDFGrid,
    )
    from blade_precompute.section_geometry.interface import export_section_json, plot_section

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "output" / "design_vector"
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chord = float(args.chord)
    web_x = sorted(f * chord for f in _parse_float_list(args.web_x_fracs))
    if len(web_x) < 1:
        p.error("At least one web x-position is required.")

    wt_parts = _parse_float_list(args.web_thickness)
    if len(wt_parts) == 1:
        web_thickness = wt_parts[0]
    elif len(wt_parts) == len(web_x):
        web_thickness = wt_parts
    else:
        p.error("--web-thickness must be one value or one per web.")

    cap_parts = _parse_float_list(args.cap_height)
    if len(cap_parts) == 1:
        cap_height = cap_parts[0]
    elif len(cap_parts) == 2:
        cap_height = (cap_parts[0], cap_parts[1])
    else:
        p.error("--cap-height must be one value or two (upper, lower).")

    twist_rad = float(np.deg2rad(args.twist_deg))
    te_x = args.te_insert_x_frac * chord if args.te_insert_x_frac is not None else None
    le_x = args.le_insert_x_frac * chord if args.le_insert_x_frac is not None else None

    airfoil = AirfoilSDF.from_naca(str(args.naca).zfill(4), chord=chord)
    mcs_kw = dict(
        airfoil_sdf=airfoil,
        web_x_positions=web_x,
        web_thickness=web_thickness,
        web_alignment=args.web_alignment,
        cap_height=cap_height,
        skin_thickness=float(args.skin_thickness),
        twist_angle=twist_rad,
        te_insert_x=te_x,
        le_insert_x=le_x,
        core_enabled=not args.no_core,
        structural_family=args.structural_family,
        fixed_cap_anchor=args.fixed_cap_anchor,
        pitch_fraction_of_chord_from_le=float(args.pitch_fraction_of_chord_from_le),
    )
    if args.fixed_cap_chord_half_width is not None:
        mcs_kw["fixed_cap_chord_half_width"] = float(args.fixed_cap_chord_half_width)
    if args.discrete_cap_chord_half_width is not None:
        mcs_kw["discrete_cap_chord_half_width"] = float(args.discrete_cap_chord_half_width)
    section = MultiCellSection(**mcs_kw)

    airfoil_plot = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-12 else airfoil
    grid = SDFGrid.from_airfoil(airfoil_plot, nx=int(args.nx), ny=int(args.ny))

    meta = {
        "naca": str(args.naca).zfill(4),
        "chord_m": chord,
        "skin_thickness_m": float(args.skin_thickness),
        "web_thickness_m": web_thickness,
        "web_x_m": web_x,
        "cap_height_m": list(cap_height) if isinstance(cap_height, tuple) else cap_height,
        "twist_deg": float(args.twist_deg),
        "web_alignment": args.web_alignment,
        "structural_family": args.structural_family,
        "fixed_cap_anchor": args.fixed_cap_anchor,
        "pitch_fraction_of_chord_from_le": float(args.pitch_fraction_of_chord_from_le),
        "labels": section.labels,
        "n_webs": section.n_webs,
        "n_cells": section.n_cells,
    }
    if args.json:
        with open(out_dir / "design_vector.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    title = (
        f"Design-vector geometry  NACA{meta['naca']}  chord={chord:.4g} m  "
        f"twist={args.twist_deg:.2f}°"
    )
    fig, ax = plot_section(section, grid, title=title, show_airfoil=False)
    fig.tight_layout()
    png = out_dir / "section_design_vector.png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass

    json_path = out_dir / "section_minimal.json"
    export_section_json(
        section,
        grid,
        {},
        str(json_path),
        include_component_zero_contours=not args.no_contours,
        include_geometry_detail=not args.no_contours,
    )

    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    if args.json:
        print(f"Wrote {out_dir / 'design_vector.json'}")
    print("Components:", ", ".join(section.labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
