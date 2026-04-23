from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Kirchhoff: DOFs per section node [u, v, w, theta_s]; in-plane wireframe uses (v, w) in y-z.
NDPN_KIRCHHOFF = 4


def plot_signature_curve(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    sig = payload.get("signature_curve") or {}
    hw = np.asarray(sig.get("half_wave_lengths_m"), dtype=np.float64)
    lam = np.asarray(sig.get("lambda_cr"), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    if hw.size and lam.size:
        m = np.isfinite(lam) & (lam > 0)
        ax.plot(hw[m], lam[m], "o-", ms=4, lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
    z = payload.get("station_z_m", 0.0)
    ax.set_xlabel("Half wavelength [m]")
    ax.set_ylabel("λ_cr")
    ax.set_title(f"GBT signature curve @ z={float(z):.3g} m")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    conv = payload.get("convergence") or {}
    if isinstance(conv, dict) and "error" in conv:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.text(0.5, 0.5, conv["error"], ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_axis_off()
    else:
        ec = np.asarray(conv.get("elem_counts"), dtype=np.float64)
        lc = np.asarray(conv.get("lambda_cr"), dtype=np.float64)
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        ax.plot(ec, lc, "s-", ms=5, lw=1.2)
        ax.set_xlabel("Hermite elements")
        ax.set_ylabel("λ_cr")
        z = payload.get("station_z_m", 0.0)
        ax.set_title(f"Mesh convergence @ z={float(z):.3g} m")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_critical_mode_shape(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    mb = payload.get("member_buckling") or {}
    x = np.asarray(mb.get("buckling_mode_x_m"), dtype=np.float64)
    w = np.asarray(mb.get("buckling_mode_w_combined"), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    if x.size and w.size and x.shape == w.shape:
        wn = w / (np.max(np.abs(w)) + 1e-30)
        ax.plot(x, wn, "-", lw=1.8)
        ax.set_xlabel("x along member [m]")
        ax.set_ylabel("Normalised |w| (combined modes)")
    z = payload.get("station_z_m", 0.0)
    lam = mb.get("lambda_cr")
    title = f"Critical mode shape @ z={float(z):.3g} m"
    if lam is not None:
        title += f" (λ_cr={float(lam):.4g})"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_eigenvalue_spectrum(payload: dict[str, Any], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    mb = payload.get("member_buckling") or {}
    ev = np.asarray(mb.get("eigenvalues"), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    if ev.size:
        n = min(12, ev.size)
        ax.bar(np.arange(n), ev[:n], color="C0", alpha=0.85)
        ax.set_xlabel("Mode index (sorted)")
        ax.set_ylabel("λ")
    z = payload.get("station_z_m", 0.0)
    ax.set_title(f"Member buckling eigenvalues @ z={float(z):.3g} m")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_buckling_member_overview_grid(
    payload: dict[str, Any],
    out_png: Path,
    *,
    suptitle: str = "",
) -> None:
    """
    Single figure: 2x2 grid of signature curve, mesh convergence, critical mode
    shape, and eigenvalue spectrum (member buckling diagnostics).
    """
    import matplotlib.pyplot as plt

    z = float(payload.get("station_z_m", 0.0))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.8))
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    sig = payload.get("signature_curve") or {}
    hw = np.asarray(sig.get("half_wave_lengths_m"), dtype=np.float64)
    lam = np.asarray(sig.get("lambda_cr"), dtype=np.float64)
    if hw.size and lam.size:
        m = np.isfinite(lam) & (lam > 0)
        ax00.plot(hw[m], lam[m], "o-", ms=3, lw=1.0)
        ax00.set_xscale("log")
        ax00.set_yscale("log")
    ax00.set_xlabel("Half wavelength [m]")
    ax00.set_ylabel("λ_cr")
    ax00.set_title("Signature curve", fontsize=10)
    ax00.grid(True, which="both", alpha=0.25)

    conv = payload.get("convergence") or {}
    if isinstance(conv, dict) and "error" in conv:
        ax01.text(0.5, 0.5, str(conv["error"]), ha="center", va="center", transform=ax01.transAxes, fontsize=8)
        ax01.set_axis_off()
    else:
        ec = np.asarray(conv.get("elem_counts"), dtype=np.float64)
        lc = np.asarray(conv.get("lambda_cr"), dtype=np.float64)
        ax01.plot(ec, lc, "s-", ms=4, lw=1.0)
        ax01.set_xlabel("Hermite elements")
        ax01.set_ylabel("λ_cr")
        ax01.set_title("Mesh convergence", fontsize=10)
        ax01.grid(True, alpha=0.25)

    mb = payload.get("member_buckling") or {}
    x = np.asarray(mb.get("buckling_mode_x_m"), dtype=np.float64)
    w = np.asarray(mb.get("buckling_mode_w_combined"), dtype=np.float64)
    if x.size and w.size and x.shape == w.shape:
        wn = w / (np.max(np.abs(w)) + 1e-30)
        ax10.plot(x, wn, "-", lw=1.5)
    ax10.set_xlabel("x along member [m]")
    ax10.set_ylabel("Norm. |w|")
    lam_cr = mb.get("lambda_cr")
    t10 = "Critical mode shape"
    if lam_cr is not None:
        t10 += f" (λ_cr={float(lam_cr):.4g})"
    ax10.set_title(t10, fontsize=10)
    ax10.grid(True, alpha=0.25)

    ev = np.asarray(mb.get("eigenvalues"), dtype=np.float64)
    if ev.size:
        n = min(10, ev.size)
        ax11.bar(np.arange(n), ev[:n], color="C0", alpha=0.85)
    ax11.set_xlabel("Mode index")
    ax11.set_ylabel("λ")
    ax11.set_title("Eigenvalue spectrum", fontsize=10)
    ax11.grid(True, axis="y", alpha=0.25)

    st = suptitle or f"Member buckling overview @ z={z:.3g} m"
    fig.suptitle(st, fontsize=11, y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _cross_section_strip_edges(section: Any) -> NDArray[np.int32]:
    """GBT strip endpoints as node index pairs (n_edges, 2)."""
    n = int(section.n_strips)
    e = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        s = section.get_strip(i)
        e[i, 0] = int(s.node_i)
        e[i, 1] = int(s.node_j)
    return e


def _deformed_yz_from_mode_column(
    phi_col: NDArray[np.float64],
    y0: NDArray[np.float64],
    z0: NDArray[np.float64],
    *,
    ndpn: int = NDPN_KIRCHHOFF,
    scale: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Apply in-plane offsets (v, w) at DOF indices 1 and 2 per node (GBT convention).

    u and theta_s are not drawn in the y-z wireframe.
    """
    n_nodes = int(y0.shape[0])
    v = np.zeros(n_nodes, dtype=np.float64)
    w = np.zeros(n_nodes, dtype=np.float64)
    for j in range(n_nodes):
        base = ndpn * j
        if base + 2 < len(phi_col):
            v[j] = float(phi_col[base + 1])
            w[j] = float(phi_col[base + 2])
    mag = float(np.max(np.sqrt(v * v + w * w))) if n_nodes else 0.0
    mag = max(mag, 1e-30)
    if scale is None:
        char_l = max(float(np.ptp(y0)) + float(np.ptp(z0)), 1e-6)
        scale_used = 0.12 * char_l / mag
    else:
        scale_used = float(scale)
    return y0 + scale_used * v, z0 + scale_used * w, scale_used


def plot_cross_section_mode_wireframes(
    section: Any,
    modal: Any,
    out_png: Path,
    *,
    station_z_m: float = 0.0,
    n_modes_plot: int = 4,
    title_suffix: str = "",
) -> None:
    """
    Plot undeformed vs deformed midsurface wireframe (strip graph) for the first
    cross-section GBT modes. Deformation uses (v, w) DOF components only.
    """
    import matplotlib.pyplot as plt

    nodes = np.asarray(section.node_coordinates, dtype=np.float64)
    if nodes.size == 0:
        return
    y0 = nodes[:, 0]
    z0 = nodes[:, 1]
    edges = _cross_section_strip_edges(section)
    modes = np.asarray(modal.modes, dtype=np.float64)
    ev = np.asarray(modal.eigenvalues, dtype=np.float64)
    n_modes = min(int(n_modes_plot), modes.shape[1], ev.size)
    if n_modes < 1:
        return

    ncols = min(2, max(1, n_modes))
    nrows = int(np.ceil(n_modes / ncols)) if n_modes else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.5 * nrows), squeeze=False)
    suffix = f" {title_suffix}".rstrip()
    for k in range(n_modes):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        phi = modes[:, k]
        yd, zd, sc = _deformed_yz_from_mode_column(phi, y0, z0)
        for ei in range(edges.shape[0]):
            i0, i1 = int(edges[ei, 0]), int(edges[ei, 1])
            ax.plot([y0[i0], y0[i1]], [z0[i0], z0[i1]], "k-", lw=1.0, alpha=0.55)
            ax.plot([yd[i0], yd[i1]], [zd[i0], zd[i1]], "C0--", lw=1.35, alpha=0.9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y [m]")
        ax.set_ylabel("z [m]")
        lamk = float(ev[k]) if k < ev.size else float("nan")
        ax.set_title(f"Mode {k + 1}  λ={lamk:.3g}{suffix}", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.text(
            0.02,
            0.98,
            f"scale={sc:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color="0.35",
        )
    for k in range(n_modes, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_axis_off()
    fig.suptitle(
        f"GBT section wireframe (undeformed black, deformed blue) @ z={float(station_z_m):.3g} m\n"
        "(v,w) offsets only; u and θ_s not shown)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_member_coupled_section_wireframe_approx(
    section: Any,
    modal: Any,
    member_res: Any,
    out_png: Path,
    *,
    station_z_m: float = 0.0,
) -> None:
    """
    Approximate midspan section shape: blend cross-section mode columns using
    ``MemberBucklingResult.modal_participation`` weights (qualitative visualisation).
    """
    import matplotlib.pyplot as plt

    nodes = np.asarray(section.node_coordinates, dtype=np.float64)
    if nodes.size == 0:
        return
    y0 = nodes[:, 0]
    z0 = nodes[:, 1]
    edges = _cross_section_strip_edges(section)
    nm = int(member_res.n_modes)
    n_node_m = int(member_res.n_elem) + 1
    w_part = np.asarray(member_res.modal_participation(nm, n_node_m), dtype=np.float64).ravel()
    w_part = w_part[:nm]
    modes = np.asarray(modal.modes, dtype=np.float64)
    m_use = min(nm, modes.shape[1], w_part.size)
    if m_use < 1:
        return
    phi_eff = modes[:, :m_use] @ w_part[:m_use].reshape(-1, 1)
    phi_eff = phi_eff.ravel()
    yd, zd, sc = _deformed_yz_from_mode_column(phi_eff, y0, z0)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for ei in range(edges.shape[0]):
        i0, i1 = int(edges[ei, 0]), int(edges[ei, 1])
        ax.plot(
            [y0[i0], y0[i1]],
            [z0[i0], z0[i1]],
            "k-",
            lw=1.1,
            alpha=0.55,
            label="undeformed" if ei == 0 else None,
        )
        ax.plot(
            [yd[i0], yd[i1]],
            [zd[i0], zd[i1]],
            "C3--",
            lw=1.5,
            alpha=0.95,
            label="approx coupled" if ei == 0 else None,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(
        f"Approx. coupled section shape @ z={float(station_z_m):.3g} m\n"
        f"(blend of GBT modes via participation, λ_cr={float(member_res.lambda_cr):.4g})",
        fontsize=9,
    )
    ax.text(0.02, 0.98, f"scale={sc:.3g}", transform=ax.transAxes, va="top", fontsize=8, color="0.35")
    ax.grid(True, alpha=0.2)
    if edges.shape[0]:
        ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
