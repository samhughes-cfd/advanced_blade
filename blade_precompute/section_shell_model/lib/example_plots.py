"""
PNG outputs for the section_shell_model example: mesh strips and stress recovery.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _stress_model_root() -> Path:
    return Path(__file__).resolve().parents[3] / "examples" / "section_stress_model"


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
    """Ply σ, ε, Hashin envelope FI at one station (delegates to stress-model helper; savefig dpi=150)."""
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
        title="section_shell_model: CLPT ply σ, ε, Hashin (reference skin station)",
    )
    return outfile


def save_panel_along_contour_figure(
    outfile: Path | str,
    panels: list[Any],
    q_tot: list,
    sig_p: list,
    panel_index: int,
    *,
    sigma_omega_mids: np.ndarray | None = None,
    dpi: int = 150,
) -> Path:
    """
    q(s), σ_xx(s) — and optionally σ_ω(s) — along contour coordinate s [m] for one panel.

    Parameters
    ----------
    sigma_omega_mids : optional (n_elem,) array of Vlasov warping stress [Pa] at element mids.
        When provided, a third subplot is added showing the warping stress contribution.
    """
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
    has_omega = sigma_omega_mids is not None and len(sigma_omega_mids) > 0

    n_rows = 3 if has_omega else 2
    figheight = 8.5 if has_omega else 6.0

    plt.style.use("dark_background")
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.0, figheight), sharex=True,
                             facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    axq, axs = axes[0], axes[1]
    for ax in axes:
        ax.set_facecolor("#0f1117")

    axq.plot(s_along, q_along, color="#2ecc71", lw=1.3, label="q")
    axq.set_ylabel("q [N/m]", color="#e0e0e0", fontsize=9)
    axq.set_title(f"Along-panel resultants — {lbl}", color="#e0e0e0", fontsize=10)
    axq.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axq.tick_params(colors="#aaaaaa", labelsize=8)
    axq.legend(loc="best", fontsize=8)

    axs.plot(s_along, sig_along / 1e6, color="#3498db", lw=1.3, label="σ_xx")
    axs.set_ylabel("σ_xx [MPa]", color="#e0e0e0", fontsize=9)
    axs.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    axs.tick_params(colors="#aaaaaa", labelsize=8)
    axs.legend(loc="best", fontsize=8)

    if has_omega:
        axw = axes[2]
        # sigma_omega_mids is at element mids; build corresponding s-mids
        s_mids = 0.5 * (s_along[:-1] + s_along[1:])
        n_om = min(len(s_mids), len(sigma_omega_mids))
        axw.plot(s_mids[:n_om], np.asarray(sigma_omega_mids[:n_om]) / 1e6,
                 color="#e67e22", lw=1.3, label="σ_ω (Vlasov)")
        axw.axhline(0.0, color="#555", lw=0.6, ls="--")
        axw.set_xlabel("s along panel [m]", color="#e0e0e0", fontsize=9)
        axw.set_ylabel("σ_ω [MPa]", color="#e0e0e0", fontsize=9)
        axw.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
        axw.tick_params(colors="#aaaaaa", labelsize=8)
        axw.legend(loc="best", fontsize=8)
    else:
        axs.set_xlabel("s along panel [m]", color="#e0e0e0", fontsize=9)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_fi_figure(
    outfile: Path | str,
    all_panel_fi: list,
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """
    Heatmap of Hashin-envelope failure indices across all panels and elements.

    Parameters
    ----------
    all_panel_fi  : list[np.ndarray]  — FI per element per panel (max over plies)
    panel_labels  : optional panel label strings
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(all_panel_fi)
    if n_panels == 0:
        return outfile

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11.0, 4.5), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Build a 2D grid: rows = panels, cols = elements (variable lengths → pad)
    max_n = max(len(fi) for fi in all_panel_fi if len(fi) > 0)
    grid = np.full((n_panels, max_n), np.nan)
    for pi, fi in enumerate(all_panel_fi):
        if len(fi) > 0:
            grid[pi, : len(fi)] = fi

    norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(np.nanmax(grid))))
    im = ax.imshow(grid, aspect="auto", norm=norm, cmap="plasma", origin="upper",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Hashin FI (max over plies)", color="#e0e0e0", fontsize=9)
    cb.ax.tick_params(colors="#aaaaaa", labelsize=7)

    ax.set_xlabel("element index along panel contour", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("panel index", color="#e0e0e0", fontsize=9)
    ax.set_title("MITC4 full-section Hashin failure index sweep", color="#e0e0e0", fontsize=10)
    ax.tick_params(colors="#aaaaaa", labelsize=7)

    if panel_labels:
        ax.set_yticks(range(n_panels))
        ax.set_yticklabels(panel_labels, fontsize=7, color="#aaaaaa")

    # Contour at FI = 1 (failure threshold)
    if np.nanmax(grid) >= 1.0:
        ax.contour(grid, levels=[1.0], colors=["#e74c3c"], linewidths=1.2)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_multi_panel_hashin_montage(
    outfile: Path | str,
    all_panel_fi: list,
    panel_labels: list[str] | None = None,
    *,
    dpi: int = 150,
) -> Path:
    """One mini heatmap subplot per panel (Hashin FI along contour)."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    n = len(all_panel_fi)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no panel FI", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return outfile
    ncols = int(math.ceil(math.sqrt(float(n))))
    nrows = int(math.ceil(n / float(ncols)))
    plt.style.use("dark_background")
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), facecolor="#0f1117")
    if nrows * ncols == 1:
        ax_flat = [axes]
    else:
        ax_flat = axes.ravel()
    vals: list[float] = []
    for fi in all_panel_fi:
        a = np.asarray(fi, dtype=np.float64).ravel()
        if a.size:
            vals.append(float(np.nanmax(a)))
    mx = max(1.0, max(vals) if vals else 0.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=mx)
    for k in range(nrows * ncols):
        ax = ax_flat[k]
        ax.set_facecolor("#0f1117")
        if k >= n:
            ax.axis("off")
            continue
        fi = np.asarray(all_panel_fi[k], dtype=np.float64).reshape(1, -1)
        ax.imshow(fi, aspect="auto", norm=norm, cmap="plasma", interpolation="nearest")
        lab = (panel_labels[k] if panel_labels and k < len(panel_labels) else f"panel_{k}")[:28]
        ax.set_title(lab, fontsize=7, color="#e0e0e0")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Per-panel Hashin FI (montage)", color="#e0e0e0", fontsize=10)
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_clpt_fi_on_section_geometry(
    outfile: Path | str,
    airfoil: np.ndarray,
    webs_geom: list,
    spar_positions: list[float],
    panels: list[Any],
    all_panel_fi: list[np.ndarray],
    *,
    dpi: int = 150,
    figsize: tuple[float, float] = (10.0, 7.0),
    vmax: float | None = None,
    cmap: str = "turbo",
) -> Path:
    """
    Map per-MITC4-element max ply **Hashin FI (CLPT)** onto panel midlines in the
    section plane (y, z) [m]. This is a static index field (not Miner damage from
    the beam-scale fatigue module).

    ``all_panel_fi[pi]`` has length equal to the number of MITC4 elements on that
    panel, matching :func:`sweep_panel_clpt_fi` output order. Each element is drawn as
    a line segment in arc-length [``s``] between the same end nodes used by
    :func:`solve_panel_mitc4` (``linspace(s_min, s_max, n_elem + 1)``).
    """
    _ensure_stress_path()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from lib.sectorial_warping import open_outline_from_airfoil  # type: ignore[import-untyped]

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    segments: list = []
    values: list[float] = []
    mid_y: list[float] = []
    mid_z: list[float] = []
    for p, fi in zip(panels, all_panel_fi, strict=True):
        fi = np.asarray(fi, dtype=float).ravel()
        n_e = int(fi.size)
        if n_e == 0:
            continue
        s_panel = np.asarray(p.s, dtype=float)
        nodes = np.asarray(p.nodes, dtype=float)
        if s_panel.size < 2 or nodes.shape[0] < 2:
            continue
        s_min, s_max = float(s_panel.min()), float(s_panel.max())
        if s_max <= s_min + 1e-30:
            continue
        s_nodes = np.linspace(s_min, s_max, n_e + 1, dtype=np.float64)
        for k in range(n_e):
            sa, sb = float(s_nodes[k]), float(s_nodes[k + 1])
            y0 = float(np.interp(sa, s_panel, nodes[:, 0]))
            z0 = float(np.interp(sa, s_panel, nodes[:, 1]))
            y1 = float(np.interp(sb, s_panel, nodes[:, 0]))
            z1 = float(np.interp(sb, s_panel, nodes[:, 1]))
            segments.append(((y0, z0), (y1, z1)))
            values.append(float(fi[k]))
            sm = 0.5 * (sa + sb)
            mid_y.append(float(np.interp(sm, s_panel, nodes[:, 0])))
            mid_z.append(float(np.interp(sm, s_panel, nodes[:, 1])))

    if not values:
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.set_title("No per-element CLPT FI to plot (empty MITC4 sweep).", color="#e0e0e0", fontsize=10)
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig)
        return outfile

    v_arr = np.asarray(values, dtype=np.float64)
    fmax = float(np.nanmax(v_arr)) if v_arr.size else 0.0
    # If all FI are << 1, a [0, 1] scale maps the whole field to the bottom
    # of "plasma" and lines vanish on a dark background.
    if vmax is not None:
        v_hi = max(float(vmax), 1e-30)
    else:
        if fmax > 0.0 and fmax < 1.0:
            v_hi = fmax
        else:
            v_hi = max(1.0, fmax) if fmax > 0.0 else 1.0
    v_hi = max(v_hi, 1e-30)
    norm = mcolors.Normalize(vmin=0.0, vmax=v_hi)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1117")
    # Slightly lighter than pure black so low-end colormap is not invisible.
    ax.set_facecolor("#1a1d2e")

    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(af[:, 0], af[:, 1], color="#5dade2", lw=1.0, alpha=0.5, zorder=1, label="Airfoil")
    for (u, w) in webs_geom:
        ax.plot([u[0], w[0]], [u[1], w[1]], color="#ecf0f1", lw=1.2, ls="--", alpha=0.5, zorder=2)

    segs = np.asarray(segments, dtype=np.float64)
    lc = LineCollection(
        segs,
        array=v_arr,
        cmap=cmap,
        norm=norm,
        linewidths=7.0,
        capstyle="round",
        joinstyle="round",
        zorder=4,
        antialiased=True,
    )
    lc.set_clip_on(False)
    ax.add_collection(lc)
    my = np.asarray(mid_y, dtype=np.float64)
    mz = np.asarray(mid_z, dtype=np.float64)
    ax.scatter(
        my,
        mz,
        c=v_arr,
        cmap=cmap,
        norm=norm,
        s=72,
        zorder=5,
        edgecolors="#f8f8f8",
        linewidths=0.85,
        clip_on=False,
    )
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(v_arr)
    cbar = fig.colorbar(m, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("max ply Hashin FI (CLPT)", color="#e0e0e0", fontsize=9)
    cbar.ax.tick_params(colors="#aaaaaa", labelsize=7)

    all_x = [0.0] + list(spar_positions) + [1.0]
    n_half = len(airfoil) // 2
    upper = airfoil[:n_half]
    order = np.argsort(upper[:, 0])
    seg_u = upper[order]
    for i in range(len(all_x) - 1):
        xm = 0.5 * (all_x[i] + all_x[i + 1])
        z_uf = float(np.interp(xm, seg_u[:, 0], seg_u[:, 1]))
        ax.text(
            xm,
            z_uf * 0.12,
            f"C{i+1}",
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
            alpha=0.9,
        )

    # LineCollection does not always expand the axes; also merge with airfoil.
    el_pts = segs.reshape(-1, 2)
    all_xy = np.vstack([af, el_pts])
    wx = float(np.ptp(all_xy[:, 0]))
    wy = float(np.ptp(all_xy[:, 1]))
    pad = max(0.02 * max(wx, wy, 0.01), 1e-6)
    ax.set_xlim(float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad)
    ax.set_ylim(float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z/c [m]", color="#e0e0e0", fontsize=9)
    ax.set_title(
        "CLPT: max ply Hashin FI on section geometry (per MITC4 element along contour)\n"
        f"(max FI = {fmax:.3e}; colour scale 0 … {v_hi:.3e})",
        color="#e0e0e0",
        fontsize=10,
    )
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")

    plt.tight_layout()
    fig.savefig(
        outfile,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="#0f1117",
        transparent=False,
        pil_kwargs={"compress_level": 6},
    )
    plt.close(fig)
    return outfile


def save_mitc4_resultants_figure(
    outfile: Path | str,
    panel_resultants: list,
    panel_label: str = "",
    *,
    dpi: int = 150,
) -> Path:
    """
    Six-panel figure showing all MITC4 shell resultants along panel contour.

    Parameters
    ----------
    panel_resultants : list[ShellPanelResultants]  — output of solve_panel_mitc4
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    indices = [r.station_index for r in panel_resultants]
    fields = [
        ("Nx",  [r.Nx  for r in panel_resultants], "N/m",   "#3498db"),
        ("Ny",  [r.Ny  for r in panel_resultants], "N/m",   "#2ecc71"),
        ("Nxy", [r.Nxy for r in panel_resultants], "N/m",   "#e74c3c"),
        ("Mx",  [r.Mx  for r in panel_resultants], "N·m/m", "#9b59b6"),
        ("My",  [r.My  for r in panel_resultants], "N·m/m", "#f39c12"),
        ("Mxy", [r.Mxy for r in panel_resultants], "N·m/m", "#1abc9c"),
    ]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), facecolor="#0f1117")
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"MITC4 shell resultants — {panel_label or 'panel'}",
        color="#e0e0e0",
        fontsize=11,
    )

    for ax, (name, vals, unit, col) in zip(axes.flat, fields):
        ax.set_facecolor("#0f1117")
        ax.plot(indices, vals, color=col, lw=1.5, marker="o", markersize=3)
        ax.axhline(0.0, color="#555", lw=0.6, ls="--")
        ax.set_title(name, color="#e0e0e0", fontsize=10)
        ax.set_ylabel(unit, color="#aaaaaa", fontsize=8)
        ax.set_xlabel("element index", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_loads_provenance_png(
    outfile: Path | str,
    *,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    station_tag: str = "",
    rz_m: float | None = None,
    n_elements_per_panel: int | None = None,
    dpi: int = 150,
) -> Path:
    """Single-panel text summary of section resultants (for final-pass provenance)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 2.2), facecolor="#0f1117")
    ax.axis("off")
    lines = [
        f"station_tag={station_tag!r}",
        f"N={N:.6g}  Vy={Vy:.6g}  Vz={Vz:.6g}  My={My:.6g}  Mz={Mz:.6g}  T={T:.6g}",
    ]
    if rz_m is not None:
        lines.append(f"r_z_m={float(rz_m):.6g} m")
    if n_elements_per_panel is not None:
        lines.append(f"n_elements_per_panel={int(n_elements_per_panel)}")
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="#e0e0e0",
    )
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_v2_section_mesh_figure(
    outfile: Path | str,
    mesh: Any,
    *,
    dpi: int = 150,
    caption_lines: list[str] | None = None,
) -> Path:
    """Plot MITC4 v2 section mesh (panel quads in ``y,z``) from :class:`Mitc4SectionMesh`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9.0, 7.0), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    pmeshes = getattr(mesh, "panel_meshes", None) or []
    for pm in pmeshes:
        yz = np.asarray(getattr(pm, "yz_nodes", []), dtype=np.float64)
        if yz.ndim != 2 or yz.shape[1] < 2:
            continue
        for el in getattr(pm, "elements", []) or []:
            if len(el) < 3:
                continue
            loop = list(el) + [el[0]]
            ii = [int(i) for i in loop if 0 <= int(i) < yz.shape[0]]
            if len(ii) < 3:
                continue
            ax.plot(yz[ii, 0], yz[ii, 1], color="#5dade2", lw=0.9, alpha=0.85)
        ax.scatter(yz[:, 0], yz[:, 1], s=6, c="#ecf0f1", alpha=0.35, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]", color="#e0e0e0", fontsize=9)
    ax.set_ylabel("z [m]", color="#e0e0e0", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.grid(True, color="#2a2a3a", lw=0.3, alpha=0.4)
    ttl = "MITC4 v2 — section mesh (midline strip quads)"
    if caption_lines:
        ttl = ttl + "\n" + " | ".join(caption_lines[:3])
    ax.set_title(ttl, color="#e0e0e0", fontsize=10)
    if caption_lines and len(caption_lines) > 3:
        fig.text(0.02, 0.02, "\n".join(caption_lines[3:]), fontsize=7, color="#aaaaaa", va="bottom")
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile


def save_mitc4_v2_dashboard_figure(
    outfile: Path | str,
    mesh: Any,
    *,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    station_tag: str,
    rz_m: float,
    n_elements_per_panel: int,
    dpi: int = 150,
) -> Path:
    """Two-panel v2 dashboard: mesh (left) + JSON-style summary (right)."""
    import json as _json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(12.0, 6.0), facecolor="#0f1117", gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    ax0.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")
    pmeshes = getattr(mesh, "panel_meshes", None) or []
    for pm in pmeshes:
        yz = np.asarray(getattr(pm, "yz_nodes", []), dtype=np.float64)
        if yz.ndim != 2 or yz.shape[1] < 2:
            continue
        for el in getattr(pm, "elements", []) or []:
            if len(el) < 3:
                continue
            loop = list(el) + [el[0]]
            ii = [int(i) for i in loop if 0 <= int(i) < yz.shape[0]]
            if len(ii) < 3:
                continue
            ax0.plot(yz[ii, 0], yz[ii, 1], color="#5dade2", lw=0.9, alpha=0.9)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_title("Mesh", color="#e0e0e0", fontsize=10)
    ax0.set_xlabel("y [m]", color="#aaaaaa", fontsize=8)
    ax0.set_ylabel("z [m]", color="#aaaaaa", fontsize=8)
    summ: dict[str, Any] = {}
    if hasattr(mesh, "summary"):
        try:
            summ = mesh.summary()
        except Exception:
            summ = {}
    txt = _json.dumps(
        {
            "station_tag": station_tag,
            "rz_m": float(rz_m),
            "resultants": {"N": N, "Vy": Vy, "Vz": Vz, "My": My, "Mz": Mz, "T": T},
            "n_elements_per_panel": int(n_elements_per_panel),
            "mesh_summary": summ,
        },
        indent=2,
    )
    ax1.axis("off")
    ax1.text(
        0.02,
        0.98,
        txt,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        color="#e0e0e0",
    )
    fig.suptitle(f"MITC4 v2 station dashboard — {station_tag}", color="#e0e0e0", fontsize=11)
    plt.tight_layout()
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    return outfile
