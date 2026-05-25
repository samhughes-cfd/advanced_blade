"""
section_optimisation.interface.plot
=====================================
Thickness profiles, spanwise failure indices, optimisation history, and
constraint diagnostics from :class:`OptimisationResult` / :class:`DesignEvaluation`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_optimisation.engine.iteration_report import (
    ks_slack_dict,
    objective_scalar,
    per_station_max_fi_hashin,
    per_station_max_fi_vm,
)

if TYPE_CHECKING:
    import matplotlib.axes as m_axes
    from matplotlib.figure import Figure

    from ..core.types import (
        DesignEvaluation,
        DesignProblem,
        DesignVector,
        OptimisationResult,
    )


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("section_optimisation plotting requires matplotlib.") from e
    return plt


def _align_z(
    z: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    zv = np.asarray(z, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    n = int(min(zv.size, yv.size))
    return zv[:n], yv[:n]


def _provenance_text(problem: Any | None, *, n_iter: int | None = None) -> str:
    if problem is None:
        return ""
    parts = [
        f"stress_recovery={getattr(problem, 'stress_recovery', '?')}",
        f"beam_driver={getattr(problem, 'beam_driver', '?')}",
        f"objective={getattr(problem, 'objective', '?')}",
    ]
    if n_iter is not None:
        parts.append(f"n_iter={int(n_iter)}")
    return " | ".join(parts)


def _apply_provenance(fig: Any, problem: Any | None, *, n_iter: int | None = None) -> None:
    txt = _provenance_text(problem, n_iter=n_iter)
    if txt:
        fig.suptitle(txt, fontsize=8, y=0.01, color="0.35")


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
    fi_h = np.array([e.max_fi_hashin for e in evs], dtype=np.float64)
    fi_vm = np.array([e.max_fi_vm for e in evs], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(idx, mass, "ko-", ms=4, lw=1.0)
    axes[0].set_ylabel("mass [kg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[1].plot(idx, fi_h, "C0.-", label="max FI Hashin")
    axes[1].plot(idx, fi_vm, "C1.-", label="max FI von Mises")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("evaluation index")
    axes[1].set_ylabel("failure index")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_max_fi_vs_span(
    z_stations: NDArray[np.float64],
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None" = None,
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Max failure index vs span",
) -> Tuple[Any, Any]:
    """Plot A: per-station max Hashin (and VM) vs span; optional second evaluation."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    h0 = per_station_max_fi_hashin(ev_initial.fi_hashin)
    zz, hh = _align_z(z, h0)
    ax.plot(zz, hh, "C0.-", ms=5, lw=1.2, label="max Hashin (initial)")
    if ev_initial.fi_vm.size:
        vm0 = per_station_max_fi_vm(ev_initial.fi_vm)
        zzv, vv = _align_z(z, vm0)
        ax.plot(zzv, vv, "C1.--", ms=4, alpha=0.85, label="max VM (initial)")
    if ev_final is not None:
        h1 = per_station_max_fi_hashin(ev_final.fi_hashin)
        zz, hh = _align_z(z, h1)
        ax.plot(zz, hh, "C2.-", ms=5, lw=1.2, label="max Hashin (optimised)")
        if ev_final.fi_vm.size:
            vm1 = per_station_max_fi_vm(ev_final.fi_vm)
            zzv, vv = _align_z(z, vm1)
            ax.plot(zzv, vv, "C3.--", ms=4, alpha=0.85, label="max VM (optimised)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("failure index")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_fi_reserve_vs_span(
    z_stations: NDArray[np.float64],
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None" = None,
    *,
    problem: "DesignProblem | None" = None,
    eps: float = 1e-12,
    title: str = "Reserve factor (1/max FI) vs span",
) -> Tuple[Any, Any]:
    """Plot B: reserve 1/max(FI) per station."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()

    def _res(arr: np.ndarray) -> np.ndarray:
        m = per_station_max_fi_hashin(arr)
        return 1.0 / np.maximum(m, eps)

    zz, rr = _align_z(z, _res(ev_initial.fi_hashin))
    ax.plot(zz, rr, "C0.-", ms=5, label="reserve Hashin (initial)")
    if ev_final is not None:
        zz, rr = _align_z(z, _res(ev_final.fi_hashin))
        ax.plot(zz, rr, "C2.-", ms=5, label="reserve Hashin (optimised)")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="unity margin")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("1 / max FI_hashin")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_governing_subcomp_hashin_vs_span(
    z_stations: NDArray[np.float64],
    ev: "DesignEvaluation",
    composite_subcomp_names: Sequence[str] | None,
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Governing subcomponent (max Hashin ply) vs span",
) -> Tuple[Any, Any]:
    """Plot C: per-station argmax subcomponent index vs span (step plot)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    fh = np.asarray(ev.fi_hashin, dtype=np.float64)
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    if fh.ndim != 3:
        ax.text(0.5, 0.5, "Need 3D fi_hashin (n_stations, n_sub, n_ply)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    n_s = int(min(fh.shape[0], z.size))
    y_idx = np.zeros(n_s, dtype=np.float64)
    for si in range(n_s):
        sub = fh[si]
        li = int(np.argmax(sub))
        ci, _pi = np.unravel_index(li, sub.shape)
        y_idx[si] = float(ci)
    zr = z[:n_s]
    ax.plot(zr, y_idx, "ko-", drawstyle="steps-mid", ms=4, lw=1.2)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("governing subcomponent index")
    n_sub = int(fh.shape[1])
    tick_labels = [
        (
            str(composite_subcomp_names[j])
            if composite_subcomp_names is not None and 0 <= j < len(composite_subcomp_names)
            else str(j)
        )
        for j in range(n_sub)
    ]
    ax.set_yticks(np.arange(n_sub, dtype=np.float64))
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_ylim(-0.5, max(n_sub - 0.5, 0.5))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_mitc4_vs_hashin_span(
    z_stations: NDArray[np.float64],
    ev: "DesignEvaluation",
    *,
    problem: "DesignProblem | None" = None,
    title: str | None = None,
) -> Tuple[Any, Any]:
    """Plot D: MITC4 per-station FI vs max Hashin per station when both exist."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    sr = str(getattr(problem, "stress_recovery", "mitc4")) if problem is not None else "mitc4"
    hh = per_station_max_fi_hashin(ev.fi_hashin)
    zz, hha = _align_z(z, hh)
    ax.plot(zz, hha, "C0.-", ms=5, label="max Hashin / station")
    if ev.fi_mitc4 is not None and np.asarray(ev.fi_mitc4).size:
        m4 = np.asarray(ev.fi_mitc4, dtype=np.float64).ravel()
        zzm, mm = _align_z(z, m4)
        ax.plot(zzm, mm, "C1.-", ms=5, label="FI MITC4 / station")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("failure index")
    ttl = title or f"MITC4 vs Hashin spanwise (stress_recovery={sr})"
    ax.set_title(ttl)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_optimisation_slack_stiffness_history(
    result: "OptimisationResult",
    problem: "DesignProblem",
    *,
    title: str = "Mass, FI, stiffness, and KS slacks vs iteration",
) -> Tuple[Any, Any]:
    """Plot E: extended iteration history with slacks and stiffness."""
    plt = _plt()
    evs = result.evaluations
    if not evs:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No evaluations", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax

    idx = np.arange(1, len(evs) + 1, dtype=np.float64)
    mass = np.array([e.mass for e in evs], dtype=np.float64)
    fi_h = np.array([e.max_fi_hashin for e in evs], dtype=np.float64)
    fi_vm = np.array([e.max_fi_vm for e in evs], dtype=np.float64)
    stiff = np.array([e.stiffness_metric for e in evs], dtype=np.float64)
    spec = stiff / np.maximum(mass, 1e-30)

    sh: list[float] = []
    svm: list[float | np.floating] = []
    sm4: list[float | np.floating] = []
    for e in evs:
        d = ks_slack_dict(e, problem)
        sh.append(float(d["slack_ks_hashin"]))
        svm.append(float(d["slack_ks_vm"]) if d["slack_ks_vm"] is not None else np.nan)
        sm4.append(float(d["slack_ks_mitc4"]) if d["slack_ks_mitc4"] is not None else np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(idx, mass, "ko-", ms=4)
    axes[0].set_ylabel("mass [kg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[1].plot(idx, fi_h, "C0.-", label="max Hashin")
    axes[1].plot(idx, fi_vm, "C1.-", label="max VM")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("max FI")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(idx, stiff, "C2.-", label="stiffness_metric")
    ax2b = axes[2].twinx()
    ax2b.plot(idx, spec, "C3.:", ms=3, label="specific_stiffness")
    axes[2].set_ylabel("stiffness_metric", color="C2")
    ax2b.set_ylabel("S/m", color="C3")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(idx, sh, "C0.-", label="slack KS Hashin")
    if not all(math.isnan(float(x)) for x in svm):
        axes[3].plot(idx, svm, "C1.--", label="slack KS VM")
    if not all(math.isnan(float(x)) for x in sm4):
        axes[3].plot(idx, sm4, "C2.--", label="slack KS MITC4")
    axes[3].axhline(0.0, color="k", ls=":", lw=0.8)
    axes[3].set_xlabel("iteration")
    axes[3].set_ylabel("KS slack (1−KS)")
    axes[3].legend(loc="best", fontsize=7)
    axes[3].grid(True, alpha=0.3)

    _apply_provenance(fig, problem, n_iter=int(result.n_iter))
    fig.tight_layout()
    return fig, axes


def plot_resultants_with_max_fi(
    z_stations: NDArray[np.float64],
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None" = None,
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Beam resultants and max Hashin FI vs span",
) -> Tuple[Any, Any]:
    """Plot F: My, Mz (and FI twinx) from stored section resultants."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    R0 = np.asarray(ev_initial.resultants, dtype=np.float64)
    if R0.ndim != 2 or R0.shape[1] < 6:
        ax.text(0.5, 0.5, "resultants not available or wrong shape", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    # Optimisation resultants are [N, My, Mz, T, Vy, Vz, B].
    i_my, i_mz = 1, 2
    if R0.shape[1] < 6:
        i_my, i_mz = min(3, R0.shape[1] - 1), min(4, R0.shape[1] - 1)
    zz, my = _align_z(z, R0[:, i_my])
    _, mz = _align_z(z, R0[:, i_mz])
    ax.plot(zz, my, "C0.-", label="My (initial)")
    ax.plot(zz, mz, "C1.-", label="Mz (initial)")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("resultant (native units)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    h0 = per_station_max_fi_hashin(ev_initial.fi_hashin)
    zz, hh = _align_z(z, h0)
    ax2.plot(zz, hh, "k--", alpha=0.5, lw=1.0, label="max Hashin FI (initial)")
    if ev_final is not None:
        R1 = np.asarray(ev_final.resultants, dtype=np.float64)
        if R1.shape == R0.shape:
            zz, my = _align_z(z, R1[:, i_my])
            _, mz = _align_z(z, R1[:, i_mz])
            ax.plot(zz, my, "C0:", alpha=0.7, label="My (opt)")
            ax.plot(zz, mz, "C1:", alpha=0.7, label="Mz (opt)")
        h1 = per_station_max_fi_hashin(ev_final.fi_hashin)
        zz, hh = _align_z(z, h1)
        ax2.plot(zz, hh, "C3.-", lw=1.2, label="max Hashin FI (optimised)")
    ax2.set_ylabel("max Hashin FI")
    ax2.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_title(title)
    h1l, l1 = ax.get_legend_handles_labels()
    h2l, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1l + h2l, l1 + l2, loc="upper left", fontsize=7, ncol=2)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_thickness_delta_vs_span(
    z_stations: NDArray[np.float64],
    dv_initial: "DesignVector",
    dv_final: "DesignVector",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Thickness change vs span (optimised − initial)",
) -> Tuple[Any, Any]:
    """§2b: Δt_skin, Δt_cap, Δt_web vs z."""
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    ds = np.asarray(dv_final.t_skin, dtype=np.float64).ravel() - np.asarray(dv_initial.t_skin, dtype=np.float64).ravel()
    dc = np.asarray(dv_final.t_cap, dtype=np.float64).ravel() - np.asarray(dv_initial.t_cap, dtype=np.float64).ravel()
    dw = np.asarray(dv_final.t_web, dtype=np.float64).ravel() - np.asarray(dv_initial.t_web, dtype=np.float64).ravel()
    axes[0].plot(z, ds, "C0.-", label="Δt_skin")
    axes[0].axhline(0.0, color="k", ls=":", lw=0.6)
    axes[0].set_ylabel("Δt_skin [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[1].plot(z, dc, "C1.-", label="Δt_cap")
    axes[1].axhline(0.0, color="k", ls=":", lw=0.6)
    axes[1].set_ylabel("Δt_cap [m]")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(z, dw, "C2.-", label="Δt_web")
    axes[2].axhline(0.0, color="k", ls=":", lw=0.6)
    axes[2].set_ylabel("Δt_web [m]")
    axes[2].set_xlabel("z [m]")
    axes[2].grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, axes


def plot_thickness_share_vs_span(
    z_stations: NDArray[np.float64],
    dv: "DesignVector",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Thickness share vs span (normalised by sum per station)",
) -> Tuple[Any, Any]:
    """§2b: stacked fraction skin/cap/web."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    ts = np.asarray(dv.t_skin, dtype=np.float64).ravel()
    tc = np.asarray(dv.t_cap, dtype=np.float64).ravel()
    tw = np.asarray(dv.t_web, dtype=np.float64).ravel()
    s = ts + tc + tw
    s = np.maximum(s, 1e-18)
    ax.stackplot(z, ts / s, tc / s, tw / s, labels=("skin share", "cap share", "web share"), alpha=0.85)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("fraction of (t_skin+t_cap+t_web)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_fi_span_heatmap(
    z_stations: NDArray[np.float64],
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "FI metrics vs station (heatmap)",
) -> Tuple[Any, Any]:
    """§2b: rows = metrics, columns = station index."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(max(8.0, float(z_stations.size) * 0.35), 4.0))
    n = int(np.asarray(z_stations, dtype=np.float64).ravel().size)
    rows: list[np.ndarray] = []
    row_labels: list[str] = []
    r0h = per_station_max_fi_hashin(ev_initial.fi_hashin)
    r0h = r0h[:n] if r0h.size else np.zeros(n)
    rows.append(r0h)
    row_labels.append("Hashin max (init)")
    if ev_initial.fi_vm.size:
        r0v = per_station_max_fi_vm(ev_initial.fi_vm)[:n]
        rows.append(np.pad(r0v, (0, max(0, n - r0v.size)), constant_values=np.nan))
        row_labels.append("VM max (init)")
    if ev_final is not None:
        r1h = per_station_max_fi_hashin(ev_final.fi_hashin)[:n]
        rows.append(np.pad(r1h, (0, max(0, n - r1h.size)), constant_values=np.nan))
        row_labels.append("Hashin max (opt)")
        if ev_final.fi_vm.size:
            r1v = per_station_max_fi_vm(ev_final.fi_vm)[:n]
            rows.append(np.pad(r1v, (0, max(0, n - r1v.size)), constant_values=np.nan))
            row_labels.append("VM max (opt)")
        if ev_final.fi_mitc4 is not None and np.asarray(ev_final.fi_mitc4).size:
            m4 = np.asarray(ev_final.fi_mitc4, dtype=np.float64).ravel()[:n]
            rows.append(np.pad(m4, (0, max(0, n - m4.size)), constant_values=np.nan))
            row_labels.append("MITC4 (opt)")
    mat = np.row_stack(rows) if rows else np.zeros((1, n))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"{i}" for i in range(n)], fontsize=7)
    ax.set_xlabel("station index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_panel_buckling_fi_vs_span(
    z_stations: NDArray[np.float64],
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Panel buckling FI (max per station) vs span",
) -> Tuple[Any, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()

    def _mx(ev: Any) -> np.ndarray | None:
        fb = getattr(ev, "fi_panel_buckling", None)
        if fb is None:
            return None
        a = np.asarray(fb, dtype=np.float64)
        if a.size == 0:
            return None
        return np.max(a, axis=tuple(range(1, a.ndim))).astype(np.float64)

    m0 = _mx(ev_initial)
    if m0 is None:
        ax.text(0.5, 0.5, "fi_panel_buckling not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    zz, yy = _align_z(z, m0)
    ax.plot(zz, yy, "C0.-", label="initial")
    if ev_final is not None:
        m1 = _mx(ev_final)
        if m1 is not None:
            zz, yy = _align_z(z, m1)
            ax.plot(zz, yy, "C1.-", label="optimised")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("max panel buckling FI")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_k7_condition_summary(
    ev_initial: "DesignEvaluation",
    ev_final: "DesignEvaluation | None",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "K7 condition number summary",
) -> Tuple[Any, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = ["cond_min", "cond_mean", "cond_max"]
    x = np.arange(len(labels))
    w = 0.35
    k0 = ev_initial.k7_cond_stats or {}
    v0 = [float(k0.get("cond_min", np.nan)), float(k0.get("cond_mean", np.nan)), float(k0.get("cond_max", np.nan))]
    ax.bar(x - w / 2, v0, width=w, label="initial", color="C0")
    if ev_final is not None and ev_final.k7_cond_stats:
        k1 = ev_final.k7_cond_stats
        v1 = [float(k1.get("cond_min", np.nan)), float(k1.get("cond_mean", np.nan)), float(k1.get("cond_max", np.nan))]
        ax.bar(x + w / 2, v1, width=w, label="optimised", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("condition number")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax


def plot_optimisation_objective_dual_axis(
    result: "OptimisationResult",
    problem: "DesignProblem",
    *,
    title: str = "Mass and specific stiffness vs iteration",
) -> Tuple[Any, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    evs = result.evaluations
    if not evs:
        ax.text(0.5, 0.5, "No evaluations", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    idx = np.arange(1, len(evs) + 1, dtype=np.float64)
    mass = np.array([e.mass for e in evs], dtype=np.float64)
    spec = np.array([e.stiffness_metric / max(e.mass, 1e-30) for e in evs], dtype=np.float64)
    ax.plot(idx, mass, "ko-", ms=4, label="mass [kg]")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mass [kg]", color="k")
    ax2 = ax.twinx()
    ax2.plot(idx, spec, "C0s-", ms=3, label="specific stiffness S/m")
    ax2.set_ylabel("S/m", color="C0")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    obj = str(getattr(problem, "objective", "min_mass"))
    if obj == "max_specific_stiffness":
        logobj = np.array([objective_scalar(e, obj) for e in evs], dtype=np.float64)
        ax2.plot(idx, logobj, "m:", alpha=0.6, label="log objective")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    _apply_provenance(fig, problem, n_iter=int(result.n_iter))
    fig.tight_layout()
    return fig, ax


def plot_fi_vs_span_per_iteration(
    z_stations: NDArray[np.float64],
    result: "OptimisationResult",
    *,
    problem: "DesignProblem | None" = None,
    title: str = "Max Hashin FI vs span (each iteration)",
) -> Tuple[Any, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    z = np.asarray(z_stations, dtype=np.float64).ravel()
    evs = result.evaluations
    if len(evs) < 2:
        ax.text(0.5, 0.5, "Need multiple evaluations", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    try:
        _cmap = plt.colormaps["gray"]
        cmap = _cmap(np.linspace(0.35, 0.85, len(evs)))
    except AttributeError:  # pragma: no cover
        import matplotlib.cm as cm

        cmap = cm.get_cmap("gray")(np.linspace(0.35, 0.85, len(evs)))
    for k, ev in enumerate(evs):
        h = per_station_max_fi_hashin(ev.fi_hashin)
        zz, hh = _align_z(z, h)
        lw = 2.0 if k == len(evs) - 1 else 0.8
        ax.plot(zz, hh, color=cmap[k], lw=lw, label=f"iter {k + 1}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("max Hashin FI")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _apply_provenance(fig, problem, n_iter=int(result.n_iter))
    fig.tight_layout()
    return fig, ax


def plot_beam_nr_residual_tail(
    ev: "DesignEvaluation",
    *,
    problem: "DesignProblem | None" = None,
    k: int = 8,
    title: str = "Beam NR residual tail (last iterations)",
) -> Tuple[Any, Any]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    from blade_precompute.section_optimisation.engine.iteration_report import beam_nr_residual_tail_array

    tail = beam_nr_residual_tail_array(ev, k=int(k))
    if tail is None or tail.size == 0:
        ax.text(0.5, 0.5, "No beam NR history", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    ax.bar(np.arange(tail.size), tail, color="C0")
    ax.set_xlabel("tail index")
    ax.set_ylabel("residual_norm")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _apply_provenance(fig, problem)
    fig.tight_layout()
    return fig, ax
