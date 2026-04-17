"""
section_optimisation.interface.plot
=====================================
Thickness profiles and optimisation history from :class:`OptimisationResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import matplotlib.axes as m_axes
    from matplotlib.figure import Figure

    from ..core.types import DesignVector, OptimisationResult


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("section_optimisation plotting requires matplotlib.") from e
    return plt


def plot_design_vector_vs_span(
    z_stations: NDArray[np.float64],
    dv: "DesignVector",
    *,
    dv_compare: "DesignVector | None" = None,
    ax: "m_axes.Axes | None" = None,
    title: str = "Thickness design variables vs span",
) -> Tuple["Figure", "m_axes.Axes"]:
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    ax.plot(z, dv.t_skin, "C0.-", label="t_skin")
    ax.plot(z, dv.t_cap, "C1.-", label="t_cap")
    ax.plot(z, dv.t_web, "C2.-", label="t_web")
    if dv_compare is not None:
        ax.plot(z, dv_compare.t_skin, "C0:", alpha=0.6, label="t_skin (ref)")
        ax.plot(z, dv_compare.t_cap, "C1:", alpha=0.6, label="t_cap (ref)")
        ax.plot(z, dv_compare.t_web, "C2:", alpha=0.6, label="t_web (ref)")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("thickness [m]")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_optimisation_history(
    result: "OptimisationResult",
    *,
    ax: "m_axes.Axes | None" = None,
    title: str = "Optimisation history",
) -> Tuple[Any, Any]:
    plt = _plt()
    evs = result.evaluations
    if not evs:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "No evaluations", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax

    if ax is not None:
        raise ValueError("plot_optimisation_history uses two axes; pass ax=None.")

    idx = np.arange(len(evs), dtype=np.float64)
    mass = np.array([e.mass for e in evs], dtype=np.float64)
    fi_tw = np.array([e.max_fi_tw for e in evs], dtype=np.float64)
    fi_vm = np.array([e.max_fi_vm for e in evs], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(idx, mass, "ko-", ms=4, lw=1.0)
    axes[0].set_ylabel("mass [kg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[1].plot(idx, fi_tw, "C0.-", label="max FI Tsai–Wu")
    axes[1].plot(idx, fi_vm, "C1.-", label="max FI von Mises")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("evaluation index")
    axes[1].set_ylabel("failure index")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes
