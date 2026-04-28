"""
Emit dummy extreme and operational blade load ``.dat`` files.

The spanwise grid and ``radial_pos`` bookkeeping column are taken from
``blade_spanwise_distribution.dat`` (which may also list ``norm_radial_pos`` and
``norm_spanwise_pos`` non-dimensional grids) so load tables stay aligned with the blade geometry.
Distributed magnitudes are the same non-physical demo envelopes as before (root → tip).

Schema (distributed inputs; internal resultants come from integration elsewhere):

  - ``extreme_load_distribution.dat`` — ``spanwise_pos``, ``radial_pos``, ``q_y_Npm``, ``q_z_Npm``, ``m_x_Nmpm``
  - ``operational_load_timeseries.dat`` — long format ``t_s``, ``spanwise_pos``, ``radial_pos``, load columns as above
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from data_library.plot_inputs import read_columnar_dat


def _span_and_radius_from_blade_dat(spanwise_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not spanwise_path.is_file():
        raise FileNotFoundError(spanwise_path)
    names, data = read_columnar_dat(spanwise_path)
    cols = {n: data[:, i] for i, n in enumerate(names)}
    if "spanwise_pos" not in cols or "radial_pos" not in cols:
        raise KeyError(
            f"Expected spanwise_pos and radial_pos in {spanwise_path!s}; have {list(cols)}."
        )
    z = np.asarray(cols["spanwise_pos"], dtype=np.float64).ravel()
    R = np.asarray(cols["radial_pos"], dtype=np.float64).ravel()
    if z.size < 2:
        raise ValueError(f"Need at least two spanwise rows in {spanwise_path!s}.")
    if z.shape != R.shape:
        raise ValueError("spanwise_pos and radial_pos length mismatch.")
    return z, R


def _write_extreme(out_path: Path, z: np.ndarray, R: np.ndarray) -> None:
    s = (z - z[0]) / max(z[-1] - z[0], 1e-12)
    # Dummy spanwise envelopes [N/m] and [N·m/m], higher toward root.
    q_y = 9.0e3 * (1.0 - s) ** 1.35 + 6.0e2
    q_z = 5.5e3 * (1.0 - s) ** 1.15 + 5.0e2
    m_x = 1.4e3 * (1.0 - s) ** 1.1 + 1.2e2

    lines = [
        "# Dummy extreme distributed loads (integrate to internal resultants downstream)",
        "# Aligned with blade_spanwise_distribution.dat (spanwise_pos, radial_pos); q_*, m_* = demo root-high envelopes.",
        "# Columns: spanwise_pos, radial_pos, q_y_Npm, q_z_Npm, m_x_Nmpm",
        f"{'spanwise_pos':>14} {'radial_pos':>14} {'q_y_Npm':>16} {'q_z_Npm':>16} {'m_x_Nmpm':>16}",
    ]
    for i in range(z.size):
        lines.append(
            f"{z[i]:14.6f} {R[i]:14.8f} {q_y[i]:16.3f} {q_z[i]:16.3f} {m_x[i]:16.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_operational(out_path: Path, z: np.ndarray, R: np.ndarray) -> None:
    s = (z - z[0]) / max(z[-1] - z[0], 1e-12)
    # Base spanwise amplitudes [N/m] and [N·m/m]
    a_qy = 1.6e3 * (1.0 - s) ** 1.25 + 1.2e2
    a_qz = 1.0e3 * (1.0 - s) ** 1.15 + 9.0e1
    a_mx = 2.8e2 * (1.0 - s) ** 1.05 + 2.5e1

    # Dummy operational timeline
    t = np.linspace(0.0, 120.0, 241, dtype=np.float64)  # 0.5 s steps
    w1 = 2.0 * np.pi / 8.0   # primary
    w2 = 2.0 * np.pi / 3.5   # secondary harmonic

    lines = [
        "# Dummy operational distributed loads (long format)",
        "# Aligned with blade_spanwise_distribution.dat (spanwise_pos, radial_pos); time-varying scale on demo amplitudes.",
        "# Columns: t_s, spanwise_pos, radial_pos, q_y_Npm, q_z_Npm, m_x_Nmpm",
        f"{'t_s':>10} {'spanwise_pos':>14} {'radial_pos':>14} {'q_y_Npm':>16} {'q_z_Npm':>16} {'m_x_Nmpm':>16}",
    ]
    for ti in t:
        p1 = np.sin(w1 * ti)
        p2 = np.sin(w2 * ti + 0.7)
        p3 = np.sin(0.5 * w1 * ti + 1.4)
        q_y_t = a_qy * (0.65 * p1 + 0.25 * p2 + 0.10 * p3)
        q_z_t = a_qz * (0.60 * p1 - 0.20 * p2 + 0.20 * np.sin(1.3 * w1 * ti + 0.4))
        m_x_t = a_mx * (0.55 * p1 + 0.30 * p2 - 0.15 * np.sin(0.8 * w1 * ti + 1.1))
        for i in range(z.size):
            lines.append(
                f"{ti:10.3f} {z[i]:14.6f} {R[i]:14.8f} "
                f"{q_y_t[i]:16.3f} {q_z_t[i]:16.3f} {m_x_t[i]:16.3f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    span_path = data_dir / "blade_spanwise_distribution.dat"

    z, R = _span_and_radius_from_blade_dat(span_path)
    out_extreme = data_dir / "extreme_load_distribution.dat"
    out_operational = data_dir / "operational_load_timeseries.dat"

    _write_extreme(out_extreme, z, R)
    _write_operational(out_operational, z, R)
    print(f"Wrote {out_extreme} ({z.size} stations)")
    print(f"Wrote {out_operational} ({z.size} stations x time)")
    print(f"Span source: {span_path}")


if __name__ == "__main__":
    main()
    sys.exit(0)
