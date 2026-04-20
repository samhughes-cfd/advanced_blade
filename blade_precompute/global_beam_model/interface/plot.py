"""
global_beam_model.interface.plot
==================================
Matplotlib helpers for static solve review (optional dependency: matplotlib).

All functions import pyplot lazily so importing this package stays lightweight without matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import matplotlib.axes as m_axes
    from matplotlib.figure import Figure

    from ..core.types import BeamLoads, BeamModel, BeamSolveResult


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("global_beam_model plotting requires matplotlib.") from e
    return plt


def _interp_1d_sorted(z_new: NDArray[np.float64], z_old: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation along span; ``z_old`` need not be sorted (uses ``np.unique`` abscissa)."""
    z_old = np.asarray(z_old, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z_new = np.asarray(z_new, dtype=np.float64).ravel()
    if z_old.size == 0 or y.size == 0:
        return np.zeros_like(z_new, dtype=np.float64)
    if z_old.size == 1:
        return np.full_like(z_new, float(y[0]), dtype=np.float64)
    u, idx = np.unique(z_old, return_index=True)
    yu = y[idx]
    return np.interp(z_new, u, yu)


def _interp_cols_sorted(z_new: NDArray[np.float64], z_old: NDArray[np.float64], arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate matrix columns along span (same ``z_old`` for each column)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return _interp_1d_sorted(z_new, z_old, arr).reshape(-1, 1)
    out = np.zeros((z_new.shape[0], arr.shape[1]), dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = _interp_1d_sorted(z_new, z_old, arr[:, j])
    return out


def span_abscissa_union(
    res: "BeamSolveResult",
    model: "BeamModel",
    loads: "BeamLoads",
    n: int,
) -> NDArray[np.float64]:
    """
    Common uniformly spaced span samples covering all z grids used in beam exports:
    Gauss output stations, nodal span coords, section recovery stations, and element midpoints
    for distributed loads.
    """
    parts: list[NDArray[np.float64]] = []
    if res.z_stations_out is not None and res.z_stations_out.size:
        parts.append(np.asarray(res.z_stations_out, dtype=np.float64).ravel())
    if res.z_nodal_out is not None and res.z_nodal_out.size:
        parts.append(np.asarray(res.z_nodal_out, dtype=np.float64).ravel())
    if res.z_section_recovery is not None and res.z_section_recovery.size:
        parts.append(np.asarray(res.z_section_recovery, dtype=np.float64).ravel())
    if loads.distributed_q is not None and model.elements:
        zm = np.array([el.z_mid for el in model.elements], dtype=np.float64)
        if zm.size:
            parts.append(zm.ravel())
    zw = _node_span_z(model)
    if zw.size:
        parts.append(np.asarray(zw, dtype=np.float64).ravel())
    if not parts:
        return np.linspace(0.0, 1.0, max(2, int(n)), dtype=np.float64)
    zcat = np.concatenate(parts)
    zmin = float(np.min(zcat))
    zmax = float(np.max(zcat))
    if zmax <= zmin:
        return np.array([zmin], dtype=np.float64)
    return np.linspace(zmin, zmax, int(n), dtype=np.float64)


def _node_span_z(model: "BeamModel") -> NDArray[np.float64]:
    if model.z_node is not None:
        return np.asarray(model.z_node, dtype=np.float64).ravel()
    z = np.zeros(model.n_nodes, dtype=np.float64)
    for el in model.elements:
        j = el.node_ids[1]
        z[j] = z[el.node_ids[0]] + el.L0
    return z


def _lateral_index(model: "BeamModel") -> int:
    """Pick a non-span axis for 2D centerline projection (flapwise-ish)."""
    sa = int(model.span_axis)
    for i in (0, 1, 2):
        if i != sa:
            return i
    return 0


def plot_centerline_ref_def(
    model: "BeamModel",
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    lateral_axis: int | None = None,
    title: str = "Reference vs deformed centerline",
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    sa = int(model.span_axis)
    lat = int(lateral_axis) if lateral_axis is not None else _lateral_index(model)
    Xr = np.asarray(model.X_ref, dtype=np.float64)
    Xd = np.asarray(res.nodal_positions, dtype=np.float64)
    ax.plot(Xr[:, lat], Xr[:, sa], "k--", lw=1.2, label="reference")
    ax.plot(Xd[:, lat], Xd[:, sa], "C0-", lw=1.5, label="deformed")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x_{lat} [m]")
    ax.set_ylabel(f"x_{sa} (span) [m]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_spanwise_resultants(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Spanwise resultants",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_resultants uses a 4×2 grid; pass ax=None.")
    z = np.asarray(res.z_stations_out, dtype=np.float64).ravel()
    R = np.asarray(res.resultants, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 7:
        raise ValueError("resultants must have shape (n, 7).")
    if z_abscissa is not None:
        R = _interp_cols_sorted(np.asarray(z_abscissa, dtype=np.float64), z, R)
        z = np.asarray(z_abscissa, dtype=np.float64).ravel()
    labels = ["N", "Vy", "Vz", "My", "Mz", "T", "B"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.ravel()
    for k in range(7):
        axes[k].plot(z, R[:, k], "C0-", lw=1.2)
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True, alpha=0.3)
    axes[6].set_xlabel("z [m]")
    axes[7].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_strains(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Spanwise strains (7 DOF sampling)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_strains uses a 4×2 grid; pass ax=None.")
    z = np.asarray(res.z_stations_out, dtype=np.float64).ravel()
    e = np.asarray(res.strains, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 7:
        raise ValueError("strains must have shape (n, 7).")
    if z_abscissa is not None:
        e = _interp_cols_sorted(np.asarray(z_abscissa, dtype=np.float64), z, e)
        z = np.asarray(z_abscissa, dtype=np.float64).ravel()
    labels = [f"ε_{i}" for i in range(7)]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.ravel()
    for k in range(7):
        axes[k].plot(z, e[:, k], "C1-", lw=1.2)
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True, alpha=0.3)
    axes[6].set_xlabel("z [m]")
    axes[7].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_resultants_nodal(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Spanwise resultants (nodal average)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_resultants_nodal uses a 4×2 grid; pass ax=None.")
    if res.z_nodal_out is None or res.resultants_nodal is None:
        raise ValueError(
            "Nodal resultants require BeamSolveResult.z_nodal_out and resultants_nodal "
            "(from Gauss→nodal projection)."
        )
    z = np.asarray(res.z_nodal_out, dtype=np.float64).ravel()
    R = np.asarray(res.resultants_nodal, dtype=np.float64)
    if R.ndim != 2 or R.shape[1] != 7:
        raise ValueError("resultants_nodal must have shape (n_nodes, 7).")
    if z_abscissa is not None:
        R = _interp_cols_sorted(np.asarray(z_abscissa, dtype=np.float64), z, R)
        z = np.asarray(z_abscissa, dtype=np.float64).ravel()
    labels = ["N", "Vy", "Vz", "My", "Mz", "T", "B"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.ravel()
    for k in range(7):
        axes[k].plot(z, R[:, k], "C0-", lw=1.2)
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True, alpha=0.3)
    axes[6].set_xlabel("z [m]")
    axes[7].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_strains_nodal(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Spanwise strains (nodal average)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_strains_nodal uses a 4×2 grid; pass ax=None.")
    if res.z_nodal_out is None or res.strains_nodal is None:
        raise ValueError(
            "Nodal strains require BeamSolveResult.z_nodal_out and strains_nodal "
            "(from Gauss→nodal projection)."
        )
    z = np.asarray(res.z_nodal_out, dtype=np.float64).ravel()
    e = np.asarray(res.strains_nodal, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 7:
        raise ValueError("strains_nodal must have shape (n_nodes, 7).")
    if z_abscissa is not None:
        e = _interp_cols_sorted(np.asarray(z_abscissa, dtype=np.float64), z, e)
        z = np.asarray(z_abscissa, dtype=np.float64).ravel()
    labels = [f"ε_{i}" for i in range(7)]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.ravel()
    for k in range(7):
        axes[k].plot(z, e[:, k], "C1-", lw=1.2)
        axes[k].set_ylabel(labels[k])
        axes[k].grid(True, alpha=0.3)
    axes[6].set_xlabel("z [m]")
    axes[7].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_section_stress(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Section ply stress (|σ| max over plies, material frame)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_stress uses a 3×1 grid; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Section stress requires BeamSolveResult.z_section_recovery (precompute section recovery).")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_stress_voigt_gp
    nd = res.section_stress_voigt_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_stress_voigt_gp and/or section_stress_voigt_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_cols_sorted(zn, z, np.asarray(gp, dtype=np.float64))
        if nd is not None:
            nd = _interp_cols_sorted(zn, z, np.asarray(nd, dtype=np.float64))
        z = zn
    labels = [r"max $|\sigma_{11}|$", r"max $|\sigma_{22}|$", r"max $|\tau_{12}|$"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k in range(3):
        axk = axes[k]
        if gp is not None:
            axk.plot(z, np.asarray(gp[:, k], dtype=np.float64) / 1e6, "C3--", lw=1.2, label="GP interp.")
        if nd is not None:
            axk.plot(z, np.asarray(nd[:, k], dtype=np.float64) / 1e6, "C0-", lw=1.2, label="nodal interp.")
        axk.set_ylabel(labels[k] + "\n[MPa]")
        axk.grid(True, alpha=0.3)
        if k == 0:
            axk.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("z [m]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_section_strain_laminate(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Laminate strain (max |ε| over subcomponents, CLPT)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_strain_laminate uses a 3×2 grid; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Section strain requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_strain_maxabs_gp
    nd = res.section_strain_maxabs_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_strain_maxabs_gp and/or section_strain_maxabs_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_cols_sorted(zn, z, np.asarray(gp, dtype=np.float64))
        if nd is not None:
            nd = _interp_cols_sorted(zn, z, np.asarray(nd, dtype=np.float64))
        z = zn
    labels = [r"$\varepsilon_0$", r"$\varepsilon_1$", r"$\varepsilon_2$", r"$\varepsilon_3$", r"$\varepsilon_4$", r"$\varepsilon_5$"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    axes = axes.ravel()
    for k in range(6):
        axk = axes[k]
        if gp is not None:
            axk.plot(z, np.asarray(gp[:, k], dtype=np.float64), "C1--", lw=1.2, label="GP interp.")
        if nd is not None:
            axk.plot(z, np.asarray(nd[:, k], dtype=np.float64), "C2-", lw=1.2, label="nodal interp.")
        axk.set_ylabel(labels[k])
        axk.grid(True, alpha=0.3)
        if k == 0:
            axk.legend(loc="best", fontsize=8)
    axes[4].set_xlabel("z [m]")
    axes[5].set_xlabel("z [m]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_section_tsai_wu(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Tsai–Wu FI (max over composite plies)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_tsai_wu uses a single axis; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_tsai_wu_fi_max_gp
    nd = res.section_tsai_wu_fi_max_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_tsai_wu_fi_max_gp and/or section_tsai_wu_fi_max_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_1d_sorted(zn, z, np.asarray(gp, dtype=np.float64).ravel())
        if nd is not None:
            nd = _interp_1d_sorted(zn, z, np.asarray(nd, dtype=np.float64).ravel())
        z = zn
    fig, axp = plt.subplots(figsize=(8, 3.5))
    if gp is not None:
        axp.plot(z, np.asarray(gp, dtype=np.float64).ravel(), "C3--", lw=1.2, label="GP interp.")
    if nd is not None:
        axp.plot(z, np.asarray(nd, dtype=np.float64).ravel(), "C0-", lw=1.2, label="nodal interp.")
    axp.set_xlabel("z [m]")
    axp.set_ylabel("FI (max ply)")
    axp.set_title(title)
    axp.legend(loc="best", fontsize=8)
    axp.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axp


def plot_spanwise_section_von_mises_fi(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "von Mises FI (max over isotropic subcomponents)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_von_mises_fi uses a single axis; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_von_mises_fi_max_gp
    nd = res.section_von_mises_fi_max_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_von_mises_fi_max_gp and/or section_von_mises_fi_max_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_1d_sorted(zn, z, np.asarray(gp, dtype=np.float64).ravel())
        if nd is not None:
            nd = _interp_1d_sorted(zn, z, np.asarray(nd, dtype=np.float64).ravel())
        z = zn
    fig, axp = plt.subplots(figsize=(8, 3.5))
    if gp is not None:
        axp.plot(z, np.asarray(gp, dtype=np.float64).ravel(), "C3--", lw=1.2, label="GP interp.")
    if nd is not None:
        axp.plot(z, np.asarray(nd, dtype=np.float64).ravel(), "C0-", lw=1.2, label="nodal interp.")
    axp.set_xlabel("z [m]")
    axp.set_ylabel("FI (max iso.)")
    axp.set_title(title)
    axp.legend(loc="best", fontsize=8)
    axp.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axp


def plot_spanwise_section_delamination_fi(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Delamination FI (Tier‑3, max over interfaces)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_delamination_fi uses a single axis; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_delamination_fi_max_gp
    nd = res.section_delamination_fi_max_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_delamination_fi_max_gp and/or section_delamination_fi_max_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_1d_sorted(zn, z, np.asarray(gp, dtype=np.float64).ravel())
        if nd is not None:
            nd = _interp_1d_sorted(zn, z, np.asarray(nd, dtype=np.float64).ravel())
        z = zn
    fig, axp = plt.subplots(figsize=(8, 3.5))
    if gp is not None:
        axp.plot(z, np.asarray(gp, dtype=np.float64).ravel(), "C3--", lw=1.2, label="GP interp.")
    if nd is not None:
        axp.plot(z, np.asarray(nd, dtype=np.float64).ravel(), "C0-", lw=1.2, label="nodal interp.")
    axp.set_xlabel("z [m]")
    axp.set_ylabel("FI (max)")
    axp.set_title(title)
    axp.legend(loc="best", fontsize=8)
    axp.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axp


def plot_spanwise_section_stress_secframe(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Section-frame ply stress (|σ| max, blade_utilities.recovery)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_stress_secframe uses a 3×1 grid; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_stress_voigt_secframe_gp
    nd = res.section_stress_voigt_secframe_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_stress_voigt_secframe_gp and/or section_stress_voigt_secframe_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_cols_sorted(zn, z, np.asarray(gp, dtype=np.float64))
        if nd is not None:
            nd = _interp_cols_sorted(zn, z, np.asarray(nd, dtype=np.float64))
        z = zn
    labels = [r"max $|\sigma_{11}|$", r"max $|\sigma_{22}|$", r"max $|\tau_{12}|$"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k in range(3):
        axk = axes[k]
        if gp is not None:
            axk.plot(z, np.asarray(gp[:, k], dtype=np.float64) / 1e6, "C3--", lw=1.2, label="GP interp.")
        if nd is not None:
            axk.plot(z, np.asarray(nd[:, k], dtype=np.float64) / 1e6, "C0-", lw=1.2, label="nodal interp.")
        axk.set_ylabel(labels[k] + "\n[MPa]")
        axk.grid(True, alpha=0.3)
        if k == 0:
            axk.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("z [m]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_spanwise_section_d_tsai_wu_dz(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "d(Tsai–Wu FI)/dz along section stations",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is not None:
        raise ValueError("plot_spanwise_section_d_tsai_wu_dz uses a single axis; pass ax=None.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    gp = res.section_d_tsai_wu_fi_dz_gp
    nd = res.section_d_tsai_wu_fi_dz_nodal
    if gp is None and nd is None:
        raise ValueError("Need section_d_tsai_wu_fi_dz_gp and/or section_d_tsai_wu_fi_dz_nodal.")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        if gp is not None:
            gp = _interp_1d_sorted(zn, z, np.asarray(gp, dtype=np.float64).ravel())
        if nd is not None:
            nd = _interp_1d_sorted(zn, z, np.asarray(nd, dtype=np.float64).ravel())
        z = zn
    fig, axp = plt.subplots(figsize=(8, 3.5))
    if gp is not None:
        axp.plot(z, np.asarray(gp, dtype=np.float64).ravel(), "C3--", lw=1.2, label="GP interp.")
    if nd is not None:
        axp.plot(z, np.asarray(nd, dtype=np.float64).ravel(), "C0-", lw=1.2, label="nodal interp.")
    axp.set_xlabel("z [m]")
    axp.set_ylabel("dFI/dz [1/m]")
    axp.set_title(title)
    axp.legend(loc="best", fontsize=8)
    axp.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axp


def plot_spanwise_section_tsai_wu_fi_heatmap(
    res: "BeamSolveResult",
    *,
    source: str = "gp",
    ax: "m_axes.Axes | None" = None,
    title: str | None = None,
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    """
    Tsai–Wu FI heatmap: span ``z`` (horizontal) × ply index (vertical), values are
    max over composite subcomponents at each (station, ply).
    """
    plt = _require_pyplot()
    if source not in ("gp", "nodal"):
        raise ValueError("source must be 'gp' or 'nodal'.")
    if res.z_section_recovery is None:
        raise ValueError("Requires BeamSolveResult.z_section_recovery.")
    if source == "gp":
        data = res.section_tsai_wu_fi_ply_envelope_gp
        label = "GP interp."
    else:
        data = res.section_tsai_wu_fi_ply_envelope_nodal
        label = "nodal interp."
    if data is None:
        raise ValueError(
            f"Need section_tsai_wu_fi_ply_envelope_{source} from section recovery."
        )
    z = np.asarray(res.z_section_recovery, dtype=np.float64).ravel()
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != z.shape[0]:
        raise ValueError("ply envelope must have shape (n_stations, n_ply_max).")
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        arr = _interp_cols_sorted(zn, z, arr)
        z = zn
    n_ply = int(arr.shape[1])
    if ax is not None:
        raise ValueError("plot_spanwise_section_tsai_wu_fi_heatmap uses a single axes; pass ax=None.")
    fig, axp = plt.subplots(figsize=(10, 4))
    # Rows = ply, cols = span → transpose for imshow with x = span
    z0, z1 = float(z[0]), float(z[-1])
    im = axp.imshow(
        arr.T,
        aspect="auto",
        origin="lower",
        extent=(z0, z1, -0.5, float(n_ply) - 0.5),
        interpolation="nearest",
    )
    axp.set_xlabel("z [m]")
    axp.set_ylabel("ply index (padded)")
    ttl = title or f"Tsai–Wu FI ({label} max over composite subcomponents)"
    axp.set_title(ttl)
    cbar = fig.colorbar(im, ax=axp, fraction=0.046, pad=0.04)
    cbar.set_label("FI")
    axp.grid(False)
    fig.tight_layout()
    return fig, axp


def plot_nodal_warping(
    model: "BeamModel",
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Nodal warping ψ",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure
    z = _node_span_z(model)
    psi = np.asarray(res.nodal_warping, dtype=np.float64).ravel()
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        psi = _interp_1d_sorted(zn, z, psi)
        z = zn
    ax.plot(z, psi, "C2.-", lw=1.2, ms=4)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("ψ [m²]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_iteration_history(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Newton / load-step history",
) -> Tuple["Figure", Any]:
    plt = _require_pyplot()
    hist: List[Dict[str, float]] = res.iteration_history
    if not hist:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No iteration history", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax

    it = np.array([h["iter"] for h in hist], dtype=np.float64)
    rn = np.array([h["residual_norm"] for h in hist], dtype=np.float64)
    du = np.array([h["displacement_norm"] for h in hist], dtype=np.float64)
    psi = np.array([float(h.get("warping_amplitude_max", 0.0)) for h in hist], dtype=np.float64)

    if ax is not None:
        raise ValueError("plot_iteration_history uses three stacked axes; pass ax=None.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(it, np.maximum(rn, 1e-30), "C0-", lw=1.0)
    axes[0].set_ylabel("|residual|")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_title(title)
    axes[1].semilogy(it, np.maximum(du, 1e-30), "C1-", lw=1.0)
    axes[1].set_ylabel("|Δu|")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[2].plot(it, psi, "C2-", lw=1.0)
    axes[2].set_ylabel("max |ψ| [m²]")
    axes[2].set_xlabel("cumulative iteration")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_reactions(
    res: "BeamSolveResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Support reactions (unbalanced residual on fixed DOFs)",
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure
    reacts = res.reactions
    if not reacts:
        ax.text(0.5, 0.5, "No reactions", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    dof_names = ["ux", "uy", "uz", "rx", "ry", "rz", "ψ"]
    keys = sorted(reacts.keys(), key=lambda t: (t[0], t[1]))
    labels = [f"n{k[0]}:{dof_names[k[1]]}" for k in keys]
    vals = np.array([reacts[k] for k in keys], dtype=np.float64)
    x = np.arange(len(keys))
    ax.bar(x, vals, color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("reaction")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_distributed_loads(
    model: "BeamModel",
    loads: "BeamLoads",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Distributed line load q (global, per element)",
    z_abscissa: NDArray[np.float64] | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _require_pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    q = loads.distributed_q
    if q is None:
        ax.text(0.5, 0.5, "No distributed_q", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    q = np.asarray(q, dtype=np.float64)
    zm = np.array([el.z_mid for el in model.elements], dtype=np.float64)
    if z_abscissa is not None:
        zn = np.asarray(z_abscissa, dtype=np.float64).ravel()
        q = _interp_cols_sorted(zn, zm, q)
        zm = zn
    ax.plot(zm, q[:, 0], ".-", label="qx")
    ax.plot(zm, q[:, 1], ".-", label="qy")
    ax.plot(zm, q[:, 2], ".-", label="qz")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("q [N/m]")
    ax.legend(loc="best")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax
