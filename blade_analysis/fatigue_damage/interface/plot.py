"""
blade_analysis.fatigue_damage.interface.plot
=================================
Matplotlib helpers for rainflow, damage/life vs span, and S–N review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import matplotlib.axes as m_axes
    from matplotlib.figure import Figure

    from ..core.types import FatigueResult, RainflowBins
    from ..engine.sn_curves import SNcurve


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("fatigue_damage plotting requires matplotlib.") from e
    return plt


def plot_rainflow_composite(
    bins: "RainflowBins",
    *,
    station: int = 0,
    subcomp: int = 0,
    ply: int = 0,
    ax: "m_axes.Axes | None" = None,
    title: str | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    r = np.asarray(bins.ranges_comp[:, station, subcomp, ply], dtype=np.float64)
    c = np.asarray(bins.counts_comp[:, station, subcomp, ply], dtype=np.float64)
    m = np.asarray(bins.means_comp[:, station, subcomp, ply], dtype=np.float64)
    sc = ax.scatter(r, c, c=m, cmap="coolwarm", s=36, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="mean stress [Pa]")
    ax.set_xlabel("Δσ bin center [Pa]")
    ax.set_ylabel("cycle count")
    ax.set_title(title or f"Rainflow (composite) s={station} sub={subcomp} ply={ply}")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_rainflow_isotropic(
    bins: "RainflowBins",
    *,
    station: int = 0,
    subcomp: int = 0,
    ax: "m_axes.Axes | None" = None,
    title: str | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    r = np.asarray(bins.ranges_iso[:, station, subcomp], dtype=np.float64)
    c = np.asarray(bins.counts_iso[:, station, subcomp], dtype=np.float64)
    m = np.asarray(bins.means_iso[:, station, subcomp], dtype=np.float64)
    sc = ax.scatter(r, c, c=m, cmap="viridis", s=36, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="mean stress [Pa]")
    ax.set_xlabel("Δσ_VM bin center [Pa]")
    ax.set_ylabel("cycle count")
    ax.set_title(title or f"Rainflow (isotropic VM) s={station} sub={subcomp}")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_damage_life_vs_span(
    result: "FatigueResult",
    z_stations: NDArray[np.float64],
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Damage and life vs span",
) -> Tuple["Figure", Any]:
    plt = _plt()
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    if z.size != result.damage_composite.shape[0]:
        raise ValueError("z_stations length must match fatigue result span dimension.")

    d_c = np.asarray(result.damage_composite, dtype=np.float64).max(axis=(1, 2))
    d_i = np.asarray(result.damage_isotropic, dtype=np.float64).max(axis=1)
    life_c = np.asarray(result.life_composite, dtype=np.float64)
    life_i = np.asarray(result.life_isotropic, dtype=np.float64)
    lc = np.where(np.isfinite(life_c) & (life_c > 0.0), life_c, np.nan)
    li = np.where(np.isfinite(life_i) & (life_i > 0.0), life_i, np.nan)
    life_c_min = np.nanmin(lc, axis=(1, 2))
    life_i_min = np.nanmin(li, axis=1)

    if ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        raise ValueError("plot_damage_life_vs_span expects ax=None (uses two axes).")

    axes[0].plot(z, d_c, "C0.-", label="max composite damage")
    axes[0].plot(z, d_i, "C1.-", label="max isotropic damage")
    axes[0].set_ylabel("Miner damage")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].semilogy(z, np.maximum(life_c_min, 1e-6), "C0.-", label="min composite life [yr]")
    axes[1].semilogy(z, np.maximum(life_i_min, 1e-6), "C1.-", label="min isotropic life [yr]")
    axes[1].axhline(result.design_life_years, color="k", ls="--", lw=1, label="design life")
    axes[1].set_xlabel("z [m]")
    axes[1].set_ylabel("life [yr]")
    axes[1].legend(loc="best")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_sn_curve_with_ranges(
    sn: "SNcurve",
    range_centers: NDArray[np.float64],
    *,
    ax: "m_axes.Axes | None" = None,
    title: str | None = None,
) -> Tuple["Figure", "m_axes.Axes"]:
    """
    Log–log S–N curve ``N_f(Δσ)`` and optional vertical markers at ``range_centers``.
    """
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    smin = max(float(np.min(range_centers[range_centers > 0])) * 0.5 if np.any(range_centers > 0) else 1e4, 1.0)
    smax = max(float(np.max(range_centers)) * 1.5 if range_centers.size else 1e8, smin * 10.0)
    S = np.geomspace(smin, smax, 200)
    Nf = sn.cycles_to_failure(S)
    ax.loglog(S, np.maximum(Nf, 1.0), "k-", lw=1.5, label=f"S–N {sn.name} (m={sn.m})")
    for rv in np.asarray(range_centers, dtype=np.float64).ravel():
        if rv > 0:
            ax.axvline(rv, color="C0", alpha=0.15, lw=1.0)
    ax.set_xlabel("Δσ [Pa]")
    ax.set_ylabel("N_f")
    ax.set_title(title or f"S–N curve: {sn.name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    return fig, ax


def plot_static_fi_vs_span(
    result: "FatigueResult",
    z_stations: NDArray[np.float64],
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Static failure indices vs span",
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    z = np.asarray(z_stations, dtype=np.float64).ravel()
    tw = np.asarray(result.fi_static_tw, dtype=np.float64).max(axis=(1, 2))
    vm = np.asarray(result.fi_static_vm, dtype=np.float64).max(axis=1)
    ax.plot(z, tw, "C0.-", label="max Tsai–Wu FI")
    ax.plot(z, vm, "C1.-", label="max von Mises FI")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="FI = 1")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("FI")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig, ax
