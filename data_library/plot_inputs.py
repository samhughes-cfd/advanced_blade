"""
plot_inputs.py
==============
Parse columnar ``.dat`` files from this folder and plot spanwise / time-series QA views.

Comment lines start with ``#``. The first non-comment line is treated as a whitespace-separated
header; subsequent lines are numeric rows. See ``DAT_STYLE.md`` for the shared file
convention used throughout ``data_library/``, including the machine-parseable
``# units:`` row consumed by :func:`read_columnar_dat_with_units`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


_UNITS_PREFIX = "# units:"


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


def read_columnar_dat_with_units(
    path: str | Path,
) -> tuple[list[str], list[str], NDArray[np.float64]]:
    """
    Return ``(column_names, units, data)`` for a single-section ``.dat`` file conforming to
    ``DAT_STYLE.md``.

    The function scans for the **last** comment line beginning ``# units:`` before the
    first non-comment row, splits it on commas, strips whitespace, and validates that
    its length equals the number of columns in the header row.

    Raises:
        ValueError: if no ``# units:`` line precedes the header, or if the unit count
            does not match the column count.
    """
    text = Path(path).read_text(encoding="utf-8")
    header: list[str] | None = None
    units: list[str] | None = None
    rows: list[list[float]] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            if header is None and s.lower().startswith(_UNITS_PREFIX):
                payload = s[len(_UNITS_PREFIX) :].strip()
                units = [tok.strip() for tok in payload.split(",") if tok.strip()]
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
    if units is None:
        raise ValueError(
            f"No '# units: ...' line found before header row in {path}. "
            "All data_library/*.dat files must declare units; see DAT_STYLE.md."
        )
    if len(units) != len(header):
        raise ValueError(
            f"{path}: '# units:' has {len(units)} entries but header has {len(header)} columns "
            f"(units={units}, header={header})."
        )
    return header, units, np.asarray(rows, dtype=np.float64)


_WS_RE = re.compile(r"\s+")
_STAR_RUN_RE = re.compile(r"\*+")
_SLASH_RUN_RE = re.compile(r"/+")


def _canonicalise_unit(s: str) -> str:
    """
    Normalise a unit string for equality comparison.

    Steps:
        1. lowercase + strip
        2. replace Unicode middle-dot (``\u00b7``) with ``*``
        3. replace any internal whitespace with ``*`` (so ``"N m / m"`` -> ``"n*m/m"``)
        4. ``**`` -> ``^``
        5. strip trailing ``^1``
        6. collapse repeated ``*`` and ``/`` runs

    Returns the canonicalised string.
    """
    t = s.strip().lower().replace("\u00b7", "*")
    t = _WS_RE.sub("*", t)
    t = t.replace("**", "^")
    t = _STAR_RUN_RE.sub("*", t)
    t = _SLASH_RUN_RE.sub("/", t)
    if t.endswith("^1"):
        t = t[:-2]
    return t


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
    Optional ``norm_radial_pos`` and ``norm_spanwise_pos`` columns (root-to-tip 0→1 grids)
    are ignored here; they are carried in the file for downstream / non-dimensional use.
    """
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes.ravel()
    rz = col["spanwise_pos"] if "spanwise_pos" in col else data[:, 0]
    if "chord_dist" in col:
        ax[0].plot(rz, col["chord_dist"], "C0.-")
        ax[0].set_ylabel("chord [m]")
    if "twist_dist" in col:
        ax[1].plot(rz, col["twist_dist"], "C1.-")
        ax[1].set_ylabel("twist [deg]")
    if "radial_pos" in col:
        ax[2].plot(rz, col["radial_pos"], "C2.-")
        ax[2].set_ylabel("R [m]")
    naca_cols = [c for c in ("naca_series", "naca_m", "naca_p", "naca_xx") if c in col]
    if naca_cols:
        for c in naca_cols:
            ax[3].plot(rz, col[c], ".-", label=c)
        ax[3].legend(loc="best")
        ax[3].set_ylabel("NACA params")
    for a in ax:
        a.set_xlabel("spanwise [m]")
        a.grid(True, alpha=0.3)
    fig.suptitle(title or Path(path).name)
    fig.tight_layout()
    return fig, axes


def plot_extreme_load_distribution_dat(path: str | Path, *, title: str | None = None):
    """Plot distributed extreme loads vs span (``extreme_load_distribution.dat``)."""
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    rz = col["spanwise_pos"] if "spanwise_pos" in col else data[:, 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ("q_y_Npm", "q_z_Npm", "m_x_Nmpm"):
        if key in col:
            ax.plot(rz, col[key], ".-", label=key)
    ax.set_xlabel("spanwise [m]")
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
    if "t_s" not in col or "spanwise_pos" not in col:
        raise ValueError("Expected columns t_s and spanwise_pos")
    rz = col["spanwise_pos"]
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
    fig.suptitle(title or f"{Path(path).name} @ spanwise≈{rz_sel:.4g} m")
    fig.tight_layout()
    return fig, axes


def plot_operational_load_heatmap(
    path: str | Path,
    *,
    value_col: str = "q_y_Npm",
    title: str | None = None,
):
    """2D colour map of ``value_col`` vs ``t_s`` and ``spanwise_pos`` when the grid is regular."""
    plt = _plt()
    names, data = read_columnar_dat(path)
    col = {n: data[:, i] for i, n in enumerate(names)}
    t = col["t_s"]
    rz = col["spanwise_pos"]
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
    ax.set_xlabel("spanwise [m]")
    ax.set_ylabel("t [s]")
    ax.set_title(title or f"{Path(path).name}: {value_col}")
    return fig, ax
