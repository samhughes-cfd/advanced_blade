"""
Emit dummy extreme and operational blade load ``.dat`` files.

Schema (distributed inputs; internal resultants come from integration elsewhere):

  - ``extreme_load_distribution.dat`` — ``r_z_m``, ``q_y_Npm``, ``q_z_Npm``, ``m_x_Nmpm``
    (optional ``R_m`` bookkeeping column)
  - ``operational_load_timeseries.dat`` — long format ``t_s``, ``r_z_m``, load columns as above
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


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


def _span_grid_from_yaml(yaml_path: Path, n_span: int = 25) -> tuple[np.ndarray, np.ndarray]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    blade = raw["blade"]
    ztab = np.asarray(blade["z_stations"], dtype=np.float64).ravel()
    r_ref = np.asarray(blade["r_ref"], dtype=np.float64)
    z = np.linspace(float(ztab[0]), float(ztab[-1]), n_span, dtype=np.float64)
    r_node = _interp_columns(z, ztab, r_ref)
    R = np.linalg.norm(r_node, axis=1)
    return z, R


def _write_extreme(out_path: Path, z: np.ndarray, R: np.ndarray) -> None:
    s = (z - z[0]) / max(z[-1] - z[0], 1e-12)
    # Dummy spanwise envelopes [N/m] and [N·m/m], higher toward root.
    q_y = 9.0e3 * (1.0 - s) ** 1.35 + 6.0e2
    q_z = 5.5e3 * (1.0 - s) ** 1.15 + 5.0e2
    m_x = 1.4e3 * (1.0 - s) ** 1.1 + 1.2e2

    lines = [
        "# Dummy extreme distributed loads (integrate to internal resultants downstream)",
        "# Columns: r_z_m, R_m, q_y_Npm, q_z_Npm, m_x_Nmpm",
        f"{'r_z_m':>12} {'R_m':>14} {'q_y_Npm':>16} {'q_z_Npm':>16} {'m_x_Nmpm':>16}",
    ]
    for i in range(z.size):
        lines.append(
            f"{z[i]:12.6f} {R[i]:14.8f} {q_y[i]:16.3f} {q_z[i]:16.3f} {m_x[i]:16.3f}"
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
        "# Columns: t_s, r_z_m, R_m, q_y_Npm, q_z_Npm, m_x_Nmpm",
        f"{'t_s':>10} {'r_z_m':>12} {'R_m':>14} {'q_y_Npm':>16} {'q_z_Npm':>16} {'m_x_Nmpm':>16}",
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
                f"{ti:10.3f} {z[i]:12.6f} {R[i]:14.8f} "
                f"{q_y_t[i]:16.3f} {q_z_t[i]:16.3f} {m_x_t[i]:16.3f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(__file__).resolve().parent
    yaml_path = root / "example_blade.yaml"

    z, R = _span_grid_from_yaml(yaml_path, n_span=25)
    out_extreme = data_dir / "extreme_load_distribution.dat"
    out_operational = data_dir / "operational_load_timeseries.dat"

    _write_extreme(out_extreme, z, R)
    _write_operational(out_operational, z, R)
    print(f"Wrote {out_extreme}")
    print(f"Wrote {out_operational}")


if __name__ == "__main__":
    main()
    sys.exit(0)
