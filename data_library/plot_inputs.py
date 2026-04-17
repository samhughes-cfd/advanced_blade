"""
plot_inputs.py
==============
Parse columnar ``.dat`` files from this folder and plot spanwise / time-series QA views.

Comment lines start with ``#``. The first non-comment line is treated as a whitespace-separated
header; subsequent lines are numeric rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def read_columnar_dat(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    """Return ``(column_names, data)`` with ``data`` shape ``(n_rows, n_cols)``."""
    text = Path(path).read_text(encoding="utf-8")
    header: list[str] | None = None
    rows: list[list[float]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if header is None:
            header = parts
            continue
        if len(parts) != len(header):
            continue
        rows.append([float(x) for x in parts])
    if header is None:
        raise ValueError(f"No header row in {path}")
    return header, np.asarray(rows, dtype=np.float64)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("data_library.plot_inputs requires matplotlib.") from e
    return plt


def plot_blade_spanwise_dat(path: str | Path, *, title: str | None = None):
    """
    Multi-panel plot for ``blade_spanwise_distribution.dat``-style files:
    span coordinate vs chord, twist, R, and NACA parameters when columns exist.
    """
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes.ravel()
    rz = col["r_z_m"] if "r_z_m" in col else data[:, 0]
    if "chord_m" in col:
        ax[0].plot(rz, col["chord_m"], "C0.-")
        ax[0].set_ylabel("chord [m]")
    if "twist_deg" in col:
        ax[1].plot(rz, col["twist_deg"], "C1.-")
        ax[1].set_ylabel("twist [deg]")
    if "R_m" in col:
        ax[2].plot(rz, col["R_m"], "C2.-")
        ax[2].set_ylabel("R [m]")
    naca_cols = [c for c in ("naca_m", "naca_p", "naca_xx") if c in col]
    if naca_cols:
        for c in naca_cols:
            ax[3].plot(rz, col[c], ".-", label=c)
        ax[3].legend(loc="best")
        ax[3].set_ylabel("NACA params")
    for a in ax:
        a.set_xlabel("r_z [m]")
        a.grid(True, alpha=0.3)
    fig.suptitle(title or Path(path).name)
    fig.tight_layout()
    return fig, axes


def plot_extreme_load_distribution_dat(path: str | Path, *, title: str | None = None):
    """Plot distributed extreme loads vs span (``extreme_load_distribution.dat``)."""
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    rz = col["r_z_m"] if "r_z_m" in col else data[:, 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ("q_y_Npm", "q_z_Npm", "m_x_Nmpm"):
        if key in col:
            ax.plot(rz, col[key], ".-", label=key)
    ax.set_xlabel("r_z [m]")
    ax.set_ylabel("load / moment per m")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title(title or Path(path).name)
    return fig, ax


def plot_operational_timeseries_dat(
    path: str | Path,
    *,
    r_z_target: float = 0.0,
    r_z_tol: float = 2e-3,
    title: str | None = None,
):
    """
    Time series at the station nearest ``r_z_target`` (default root): ``q_y``, ``q_z``, ``m_x``.
    """
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    if "t_s" not in col or "r_z_m" not in col:
        raise ValueError("Expected columns t_s and r_z_m")
    rz = col["r_z_m"]
    mask = np.abs(rz - float(r_z_target)) <= float(r_z_tol)
    if not np.any(mask):
        i = int(np.argmin(np.abs(rz - float(r_z_target))))
        mask = rz == rz[i]
    t = col["t_s"][mask]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    series = [("q_y_Npm", "q_y [N/m]"), ("q_z_Npm", "q_z [N/m]"), ("m_x_Nmpm", "m_x [N·m/m]")]
    for ax, (k, ylab) in zip(axes, series):
        if k in col:
            ax.plot(t, col[k][mask], "C0-", lw=0.8)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    rz_sel = float(np.mean(rz[mask])) if np.any(mask) else r_z_target
    fig.suptitle(title or f"{Path(path).name} @ r_z≈{rz_sel:.4g} m")
    fig.tight_layout()
    return fig, axes


def plot_operational_load_heatmap(
    path: str | Path,
    *,
    value_col: str = "q_y_Npm",
    title: str | None = None,
):
    """2D colour map of ``value_col`` vs ``t_s`` and ``r_z_m`` when the grid is regular."""
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    t = col["t_s"]
    rz = col["r_z_m"]
    v = col[value_col]
    ut = np.unique(t)
    urz = np.unique(rz)
    if ut.size * urz.size != t.size:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Irregular grid: use station time-series plot", ha="center", va="center")
        return fig, ax
    Z = v.reshape(ut.size, urz.size)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(urz, ut, Z, shading="auto")
    fig.colorbar(im, ax=ax, label=value_col)
    ax.set_xlabel("r_z [m]")
    ax.set_ylabel("t [s]")
    ax.set_title(title or f"{Path(path).name}: {value_col}")
    return fig, ax
