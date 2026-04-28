"""
Focused repro for the shell-mesh trailing-edge truncation bug.

Exercises only the section_shell_model stage (no orchestration/optimisation),
mirroring the call site in
``blade_precompute/orchestration/precompute/stages.py::section_shell_model_impl``:

    airfoil = build_airfoil_for_station(...)            # chord-scaled (metres)
    spars   = section_shell_spars_from_layout(...)      # chord fractions [0..1]
    write_section_shell_model_station_outputs(..., airfoil=airfoil, spars=spars)

The script reruns this for three representative chord lengths spanning the run007
distribution (root chord ~1.7 m, mid-span ~1.0 m, tip ~0.7 m) and writes one
NDJSON entry per station to ``debug-55cddb.log`` capturing the airfoil x-extent,
the spars (raw values), and the ``all_x = [0.0] + sorted(spars) + [1.0]``
boundaries that ``build_section`` will use. This lets us confirm/reject the
units-mismatch hypothesis without running ``main_precompute.py``.

Usage (from repo root)::

    python blade_precompute/section_shell_model/examples/repro_shell_units_bug.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent.parent
    examples_stress = repo / "examples" / "section_stress_model"
    for p in (repo, examples_stress):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "debug-55cddb.log"
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

    from blade_precompute.section_shell_model.job_outputs import (
        build_airfoil_for_station,
        write_section_shell_model_station_outputs,
    )

    # Mirror system_layout.web_chord_fracs (sorted fractions) used by run007.
    spars: list[float] = [0.33, 0.67]

    # Three representative stations: root, mid, tip (chord_m and r_z_m
    # eyeballed from the run007 mesh PNGs to span the broken regime).
    stations = [
        {"i": 0, "rz_m": 0.0, "chord_m": 1.7, "tag": "root"},
        {"i": 4, "rz_m": 4.28, "chord_m": 1.05, "tag": "mid"},
        {"i": 8, "rz_m": 8.56, "chord_m": 0.69, "tag": "tip"},
    ]

    # NACA 2415 — naca_digits_to_params expects integer digits (m=2, p=4, xx=15
    # → m=0.02, p=0.4, t/c=0.15). Using a non-degenerate t/c is required so the
    # downstream Bredt solver matrix is non-singular and PNGs are written.
    naca_m, naca_p, naca_xx = 2.0, 4.0, 15.0

    out_root = Path(__file__).resolve().parent / "output" / "repro_shell_units_bug"
    out_root.mkdir(parents=True, exist_ok=True)

    # Run each station twice: once with the BUGGY units (spars = fractions, airfoil
    # in metres) and once with the FIXED units (spars converted to metres). We log
    # both so the verification log directly compares before/after for the same case.
    for variant in ("buggy", "post-fix"):
        for st in stations:
            chord = float(st["chord_m"])
            airfoil = build_airfoil_for_station(
                naca_m=naca_m,
                naca_p=naca_p,
                naca_xx=naca_xx,
                chord_m=chord,
                naca_series=4,
                n_points=120,
            )
            af = np.asarray(airfoil, dtype=float)

            if variant == "buggy":
                spars_used = list(spars)
            else:
                spars_used = [s * chord for s in spars]

            all_x = [0.0] + sorted(float(s) for s in spars_used) + [1.0 if variant == "buggy" else float(af[:, 0].max())]
            _log(
                {
                    "runId": variant,
                    "hypothesisId": "H1",
                    "location": "repro_shell_units_bug.py:per_station",
                    "message": f"shell stage units check ({variant})",
                    "data": {
                        "variant": variant,
                        "station_i": int(st["i"]),
                        "rz_m": float(st["rz_m"]),
                        "tag": str(st["tag"]),
                        "chord_m": chord,
                        "spars_used": [float(s) for s in spars_used],
                        "airfoil_x_min": float(af[:, 0].min()),
                        "airfoil_x_max": float(af[:, 0].max()),
                        "airfoil_npoints": int(af.shape[0]),
                        "all_x_used_by_build_section": all_x,
                        "trailing_cell_x0": float(all_x[-2]),
                        "trailing_cell_x1": float(all_x[-1]),
                        "trailing_cell_width_m": float(all_x[-1] - all_x[-2]),
                        "expected_TE_x_m": float(af[:, 0].max()),
                        "gap_to_TE_m": float(af[:, 0].max() - all_x[-1]),
                        "te_outside_meshed_region": bool(af[:, 0].max() > all_x[-1] + 1e-9),
                        "aft_web_outside_airfoil": bool(all_x[-2] > af[:, 0].max() + 1e-9),
                    },
                }
            )

            station_dir = out_root / variant / f"station_{st['tag']}_chord{chord:.2f}m"
            station_dir.mkdir(parents=True, exist_ok=True)
            try:
                write_section_shell_model_station_outputs(
                    station_dir,
                    airfoil=airfoil,
                    spars=spars_used,
                    station_tag=f"i{int(st['i']):03d}_rz{float(st['rz_m']):.3f}",
                    n_elements_per_panel=12,
                    dpi=120,
                    persist_pngs=True,
                )
            except Exception as exc:
                _log(
                    {
                        "runId": variant,
                        "hypothesisId": "H1",
                        "location": "repro_shell_units_bug.py:write_outputs",
                        "message": "write_section_shell_model_station_outputs raised",
                        "data": {
                            "variant": variant,
                            "station_i": int(st["i"]),
                            "tag": str(st["tag"]),
                            "exception_type": type(exc).__name__,
                            "exception_msg": str(exc)[:500],
                        },
                    }
                )

    print(f"Repro complete. Log: {_LOG_PATH}")
    print(f"Mesh PNGs:  {out_root}")


if __name__ == "__main__":
    main()
