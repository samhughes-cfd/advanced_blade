"""
PNG outputs for the section_shell_model example: mesh strips and stress recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _stress_model_root() -> Path:
    return Path(__file__).resolve().parents[2] / "section_stress_model"


def _ensure_stress_path() -> None:
    s = str(_stress_model_root())
    if s not in sys.path:
        sys.path.insert(0, s)


def save_shell_mesh_figure(
    outfile: Path | str,
    panels: list[Any],
    webs_geom: list,
    airfoil: np.ndarray,
    spar_positions: list[float],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
) -> Path:
    """
    Plot midline thin-wall mesh: each edge between consecutive panel nodes is one
    linear strip element; nodes indicate discretisation density.
    """
    _ensure_stress_path()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lib.sectorial_warping import open_outline_from_airfoil  # type: ignore[import-untyped]

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(
        af[:, 0],
        af[:, 1],
        color="#5dade2",
        lw=1.2,
        alpha=0.85,
        zorder=1,
    )

    import matplotlib.cm as cm_base

    cmap = cm_base.get_cmap("tab20")
    n_seg_total = 0
    n_node_samples = 0
    for pi, p in enumerate(panels):
        nodes = np.asarray(p.nodes, dtype=float)
        if len(nodes) < 2:
            continue
        col = cmap((pi % 20) / 19.0)
        ax.plot(
            nodes[:, 0],
            nodes[:, 1],
            color=col,
            lw=2.2,
            solid_capstyle="round",
            zorder=3,
        )
        n_seg_total += len(nodes) - 1
        n_node_samples += len(nodes)
        ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=[col],
            s=14,
            zorder=5,
            edgecolors="#ffffff",
            linewidths=0.35,
        )
        lbl = getattr(p, "label", None) or f"P{pi}"
        mid = nodes[len(nodes) // 2]
        ax.text(
            mid[0],
            mid[1],
            f" {lbl}",
            fontsize=6,
            color=col,
            alpha=0.95,
            zorder=6,
            clip_on=True,
        )

    for (u, l) in webs_geom:
        ax.plot(
            [u[0], l[0]],
            [u[1], l[1]],
            color="#ecf0f1",
            lw=2.5,
            ls="--",
            alpha=0.9,
            zorder=4,
        )

    all_x = [0.0] + list(spar_positions) + [1.0]
    n_half = len(airfoil) // 2
    upper = airfoil[:n_half]
    order = np.argsort(upper[:, 0])
    seg = upper[order]
    for i in range(len(all_x) - 1):
        xm = 0.5 * (all_x[i] + all_x[i + 1])
        z_u = float(np.interp(xm, seg[:, 0], seg[:, 1]))
        ax.text(
            xm,
            z_u * 0.15,
            f"C{i+1}",
            ha="center",
            va="center",
            fontsize=9,
            color="#aaaaaa",
            alpha=0.7,
        )

    ax.set_aspect("equal")
    ax.set_title(
        "Thin-wall mesh: linear strip elements along panel midlines\n"
        f"(segments={n_seg_total}, node samples={n_node_samples}, panels={len(panels)})",
        color="#e0e0e0",
        fontsize=10,
    )
    ax.set_xlabel("y/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z/c [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_thin_wall_stress_figures(
    outfile_prefix: Path | str,
    panels: list[Any],
    booms: list[Any],
    webs_geom: list,
    airfoil: np.ndarray,
    spar_positions: list[float],
    q_tot: list,
    sig_p: list,
    sig_b: list,
    *,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """
    Shear-flow and axial-stress ribbon plots via ``multi_cell_blade_section.plot_section``.
    Writes ``{prefix}_shear_flow.png`` and ``{prefix}_axial_stress.png``.
    """
    _ensure_stress_path()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from multi_cell_blade_section import plot_section  # type: ignore[import-untyped]

    base = Path(outfile_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig1, ax1 = plt.subplots(figsize=(9.0, 6.5), facecolor="#0f1117")
    plot_section(
        ax1,
        panels,
        booms,
        webs_geom,
        airfoil,
        spar_positions,
        q_tot,
        sig_p,
        sig_b,
        "Shear flow q(s) — thin-wall recovery",
        plot_shear=True,
        plot_bending=False,
        title_fontsize=11.0,
    )
    fig1.patch.set_facecolor("#0f1117")
    p1 = Path(f"{base}_shear_flow.png")
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9.0, 6.5), facecolor="#0f1117")
    plot_section(
        ax2,
        panels,
        booms,
        webs_geom,
        airfoil,
        spar_positions,
        q_tot,
        sig_p,
        sig_b,
        "Axial stress σ_xx(s) — thin-wall recovery",
        plot_shear=False,
        plot_bending=True,
        title_fontsize=11.0,
    )
    fig2.patch.set_facecolor("#0f1117")
    p2 = Path(f"{base}_axial_stress.png")
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig2)

    return p1, p2


def save_clpt_ply_figure(
    outfile: Path | str,
    panels: list[Any],
    q_tot: list,
    sig_p: list,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict[str, float] | None = None,
) -> Path:
    """Ply σ, ε, Tsai–Wu at one station (delegates to stress-model helper; savefig dpi=150)."""
    _ensure_stress_path()
    from multi_cell_blade_section import plot_clpt_laminate_stress_fi  # type: ignore[import-untyped]

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plot_clpt_laminate_stress_fi(
        panels,
        q_tot,
        sig_p,
        panel_index=panel_index,
        station_index=station_index,
        strengths=strengths,
        outfile=outfile,
        title="section_shell_model: CLPT ply σ, ε, Tsai–Wu (reference skin station)",
    )
    return outfile


def save_panel_along_contour_figure(
    outfile: Path | str,
    panels: list[Any],
    q_tot: list,
    sig_p: list,
    panel_index: int,
    *,
    dpi: int = 150,
) -> Path:
    """q(s) and σ_xx(s) along contour coordinate s [m] for one panel."""
    _ensure_stress_path()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    p = panels[panel_index]
    s_along = np.asarray(p.s, dtype=float)
    q_along = np.asarray(q_tot[panel_index], dtype=float).ravel()
    sig_along = np.asarray(sig_p[panel_index], dtype=float).ravel()
    n = min(len(s_along), len(q_along), len(sig_along))
    s_along = s_along[:n]
    q_along = q_along[:n]
    sig_along = sig_along[:n]

    lbl = getattr(p, "label", None) or f"panel_{panel_index}"

    plt.style.use("dark_background")
    fig, (axq, axs) = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True, facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    for ax in (axq, axs):
        ax.set_facecolor("#0f1117")

    axq.plot(s_along, q_along, color="#2ecc71", lw=1.3, label="q")
    axq.set_ylabel("q [N/m]", color="#e0e0e0", fontsize=9)
    axq.set_title(
        f"Along-panel resultants — {lbl}",
        color="#e0e0e0",
        fontsize=10,
    )
    axq.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axq.tick_params(colors="#aaaaaa", labelsize=8)
    axq.legend(loc="best", fontsize=8)

    axs.plot(s_along, sig_along / 1e6, color="#3498db", lw=1.3, label="σ_xx")
    axs.set_xlabel("s along panel [m]", color="#e0e0e0", fontsize=9)
    axs.set_ylabel("σ_xx [MPa]", color="#e0e0e0", fontsize=9)
    axs.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axs.tick_params(colors="#aaaaaa", labelsize=8)
    axs.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile
