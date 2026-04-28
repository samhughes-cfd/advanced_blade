"""
Emit ``blade_spanwise_distribution.dat`` from ``tidal_blade_radial_source.tsv``.

Each TSV row is ``R [m]``, ``chord [m]``, ``twist [deg]`` (hub-centre radius).
Spanwise coordinate is ``spanwise_pos = R - R`` at the first station (root z = 0).
Non-dimensional columns ``norm_radial_pos`` and ``norm_spanwise_pos`` are 0 at the
first row and 1 at the tip: ``(· - min)/(max - min)`` over the tabulated blade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _read_tidal_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_list: list[float] = []
    c_list: list[float] = []
    t_list: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Expected 3 tab columns in {path}, got {len(parts)}: {line!r}")
        r_list.append(float(parts[0]))
        c_list.append(float(parts[1]))
        t_list.append(float(parts[2]))
    if len(r_list) < 2:
        raise ValueError(f"Need at least two rows in {path}")
    return (
        np.asarray(r_list, dtype=np.float64),
        np.asarray(c_list, dtype=np.float64),
        np.asarray(t_list, dtype=np.float64),
    )


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    tsv_path = data_dir / "tidal_blade_radial_source.tsv"
    out_path = data_dir / "blade_spanwise_distribution.dat"

    R, chord, twist_deg = _read_tidal_tsv(tsv_path)
    spanwise_pos = R - float(R[0])
    z_min, z_max = float(spanwise_pos.min()), float(spanwise_pos.max())
    R_min, R_max = float(R.min()), float(R.max())
    L = max(z_max - z_min, 1e-12)
    dR = max(R_max - R_min, 1e-12)
    norm_spanwise_pos = (spanwise_pos - z_min) / L
    norm_radial_pos = (R - R_min) / dR

    naca_series = np.full_like(R, 6.0, dtype=np.float64)
    naca_m = np.full_like(R, 63.0, dtype=np.float64)
    naca_p = np.full_like(R, 4.0, dtype=np.float64)
    naca_xx = np.full_like(R, 15.0, dtype=np.float64)

    comments = """# Blade spanwise distribution
# Source: data_library/tidal_blade_radial_source.tsv (interpolated).
#
# spanwise z [m] , radial r [m], r/R [-], z/L [-]    , chord c [m],  twist β [deg],  naca_series   naca_m   naca_p  naca_xx
# Parser row (single tokens, no commas; same column order): spanwise_pos radial_pos norm_radial_pos norm_spanwise_pos chord_dist twist_dist naca_series naca_m naca_p naca_xx
# Datum: first row z = 0 at first tabulated station; radial r (hub) there 1.375 m. r/R and z/L are root-to-tip 0 to 1.
# twist β = structural built twist / washout (not angle of attack or pitch DOF).
#
# naca_series: 4=four-digit, 5=five-digit, 6=six-series (naca_m family 63|64, naca_p=design Cl*10, naca_xx=thickness %%).
# Full-span NACA 63-415: naca_series=6, naca_m=63, naca_p=4, naca_xx=15 (embedded UIUC n63415, y-scaled to thickness).
#
"""
    header = (
        f"{'spanwise_pos':>14} {'radial_pos':>14} {'norm_radial_pos':>16} {'norm_spanwise_pos':>18} "
        f"{'chord_dist':>15} {'twist_dist':>15} {'naca_series':>12} {'naca_m':>8} {'naca_p':>8} {'naca_xx':>8}"
    )
    lines = [comments.rstrip("\n"), header]
    for i in range(R.size):
        lines.append(
            f"{spanwise_pos[i]:14.6f} {R[i]:14.8f} {norm_radial_pos[i]:16.10f} {norm_spanwise_pos[i]:18.10f} "
            f"{chord[i]:15.9f} {twist_deg[i]:15.9f} {naca_series[i]:12.0f} "
            f"{naca_m[i]:8.3f} {naca_p[i]:8.3f} {naca_xx[i]:8.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({R.size} rows from {tsv_path.name})")


if __name__ == "__main__":
    main()
    sys.exit(0)
