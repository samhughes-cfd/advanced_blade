"""
Emit blade_spanwise_distribution.dat from example_blade.json.

Columns (parsed header tokens; see file comments for human labels):
  spanwise_pos — spanwise coordinate (blade z / station axis) [m]
  radial_pos   — reference locus magnitude ||r_ref|| [m]
  norm_radial_pos / norm_spanwise_pos — root-to-tip 0→1 grids [-]
  chord_dist   — chord [m]
  twist_dist   — **structural blade twist** [deg]: built-in section orientation (washout / built twist
    distribution along the span). This is **not** angle of attack ``α``, **not** collective pitch,
    and **not** inflow or turbine yaw/pitch kinematics. Values come from spec ``blade.twist``
    interpolated on ``z``, plus an optional dummy spanwise overlay for non-flat demo data.
  naca_series — NACA family code (4|5|6); emitted as 4 for this demo path
  naca_m    — NACA 4-digit: max camber as fraction of chord × 100
  naca_p    — NACA 4-digit: position of max camber along chord × 10
  naca_xx   — NACA 4-digit: thickness as fraction of chord × 100

NACA columns default to a root→tip blend (0012 → 4412) unless the input spec lists
blade.airfoil_profiles with matching station count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from blade_precompute._utils.spec_io import load_mapping


def _interp_columns(zq: np.ndarray, ztab: np.ndarray, ytab: np.ndarray) -> np.ndarray:
    zq = np.asarray(zq, dtype=np.float64).ravel()
    zt = np.asarray(ztab, dtype=np.float64).ravel()
    y = np.asarray(ytab, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    out = np.zeros((zq.shape[0], y.shape[1]), dtype=np.float64)
    for j in range(y.shape[1]):
        out[:, j] = np.interp(zq, zt, y[:, j])
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    blade_spec_path = root / "example_blade.json"
    out_path = Path(__file__).resolve().parent / "blade_spanwise_distribution.dat"

    raw = load_mapping(blade_spec_path)
    blade = raw["blade"]
    ztab = np.asarray(blade["z_stations"], dtype=np.float64).ravel()
    r_ref = np.asarray(blade["r_ref"], dtype=np.float64)
    chord_tab = np.asarray(blade["chord"], dtype=np.float64).ravel()
    twist_tab = np.asarray(blade["twist"], dtype=np.float64).ravel()

    z0, z1 = float(ztab[0]), float(ztab[-1])
    n = 25
    z = np.linspace(z0, z1, n, dtype=np.float64)
    r_node = _interp_columns(z, ztab, r_ref)
    R = np.linalg.norm(r_node, axis=1)
    chord = np.interp(z, ztab, chord_tab)
    twist_deg = np.interp(z, ztab, twist_tab)
    z_min, z_max = float(z.min()), float(z.max())
    R_min, R_max = float(R.min()), float(R.max())
    norm_spanwise_pos = (z - z_min) / max(z_max - z_min, 1e-12)
    norm_radial_pos = (R - R_min) / max(R_max - R_min, 1e-12)
    s = (z - z0) / max(z1 - z0, 1e-12)
    # Demo-only spanwise offset [deg] on tabulated structural twist (not AoA / not pitch DOF).
    twist_deg = twist_deg + (-1.25 + 13.5 * s**1.18) + 0.8 * np.sin(np.pi * s)

    profiles = blade.get("airfoil_profiles") or []
    if isinstance(profiles, list) and len(profiles) == ztab.size:
        # Expect strings like "0012" or "NACA2412" at each z_stations row.

        def parse_naca(s: str) -> tuple[float, float, float]:
            s = str(s).upper().replace("NACA", "").strip()
            if len(s) != 4 or not s.isdigit():
                raise ValueError(f"Expected 4-digit NACA string, got {s!r}")
            m = float(s[0])
            p = float(s[1])
            xx = float(s[2:4])
            return m, p, xx

        m_tab = np.zeros(ztab.size, dtype=np.float64)
        p_tab = np.zeros(ztab.size, dtype=np.float64)
        xx_tab = np.zeros(ztab.size, dtype=np.float64)
        for i, prof in enumerate(profiles):
            mi, pi, xxi = parse_naca(str(prof))
            m_tab[i] = mi
            p_tab[i] = pi
            xx_tab[i] = xxi
        naca_m = np.interp(z, ztab, m_tab)
        naca_p = np.interp(z, ztab, p_tab)
        naca_xx = np.interp(z, ztab, xx_tab)
    else:
        # Default blend for demo distributions when input has no per-station NACA list
        t = (z - z0) / max(z1 - z0, 1e-12)
        naca_m = 4.0 * t
        naca_p = 4.0 * t
        naca_xx = np.full_like(z, 12.0)

    naca_series = np.full(n, 4.0, dtype=np.float64)

    lines = [
        "# Blade spanwise distribution",
        f"# Source spec: {blade_spec_path.as_posix()}",
        "#",
        "# spanwise z [m] , radial r [m], r/R [-], z/L [-] , chord c [m],  twist β [deg],  naca_series , naca_m , naca_p , naca_xx",
        "# Parser row (single tokens, no commas; same column order): spanwise_pos radial_pos norm_radial_pos norm_spanwise_pos chord_dist twist_dist naca_series naca_m naca_p naca_xx",
        "#",
    ]
    header = (
        f"{'spanwise_pos':>14} {'radial_pos':>14} {'norm_radial_pos':>16} {'norm_spanwise_pos':>18} "
        f"{'chord_dist':>15} {'twist_dist':>15} {'naca_series':>12} {'naca_m':>8} {'naca_p':>8} {'naca_xx':>8}"
    )
    lines.append(header)
    for i in range(n):
        lines.append(
            f"{z[i]:14.6f} {R[i]:14.8f} {norm_radial_pos[i]:16.10f} {norm_spanwise_pos[i]:18.10f} "
            f"{chord[i]:15.6f} {twist_deg[i]:15.6f} {naca_series[i]:12.0f} "
            f"{naca_m[i]:8.3f} {naca_p[i]:8.3f} {naca_xx[i]:8.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
