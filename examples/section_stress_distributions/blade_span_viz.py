"""
Spanwise 3D visualization (sweepkit-style B-side-forward): geometry and scalar fields
on panel segments along the span.

Uses ``run_section`` outputs directly (not raster compositing of existing PNGs).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

from lib.blade_frames import rotation_B_to_S
from blade_span import SpanRunConfig, eval_theta_geom_rad, run_span_stations_with_airfoils

WebConfig = tuple[list[float], str]

# Match sweepkit ``span3d_figures.py`` B-side-forward camera (scaled_profiles canonical view).
_VIEW_ELEV_B_SIDE_FORWARD = 25.0
_VIEW_AZIM_B_SIDE_FORWARD = -50.0
_VIEW_ROLL_B_SIDE_FORWARD = 0.0


def points_yz_S_to_edge_flap(yz: np.ndarray, theta_geom_rad: float) -> np.ndarray:
    """Section coordinates in S (y,z) to B (edge, flap); ``v_B = R.T @ v_S`` (see ``blade_frames``)."""
    R = rotation_B_to_S(float(theta_geom_rad))
    y = np.asarray(yz, dtype=float).reshape(-1, 2)
    return (R.T @ y.T).T


def mpl_xyz_B_side_forward(span_z_m: np.ndarray, yz_s: np.ndarray, theta_geom_rad: float) -> np.ndarray:
    """
    Matplotlib 3D coordinates aligned with ``blade-structure`` sweepkit ``span3d_figures``:

    ``(x,y,z)`` = (spanwise Z [m], edgewise Y_B [m], flapwise X_B [m]).
    """
    pb = points_yz_S_to_edge_flap(yz_s, theta_geom_rad)
    n = pb.shape[0]
    sz = np.asarray(span_z_m, dtype=float).ravel()
    if sz.size == 1:
        sz = np.full(n, float(sz[0]))
    return np.column_stack([sz, pb[:, 0], pb[:, 1]])


def run_span_all_web_configs(
    base_cfg: SpanRunConfig,
    web_configs: Sequence[WebConfig],
    N_B: np.ndarray,
    V_edge_B: np.ndarray,
    V_flap_B: np.ndarray,
    M_edge_B: np.ndarray,
    M_flap_B: np.ndarray,
    T_B: np.ndarray,
    B_warp: np.ndarray | None = None,
) -> list[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]]:
    """
    For each ``(spar_fractions, label)``, clone ``base_cfg`` with ``spar_positions=spars``
    and run ``run_span_stations_with_airfoils``.

    Returns ``[(label, spars, [(airfoil_m, run_section_out), ...]), ...]``.
    """
    out: list[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]] = []
    for spars, label in web_configs:
        sp = list(spars)
        cfg = replace(base_cfg, spar_positions=sp)
        rows = run_span_stations_with_airfoils(
            cfg,
            N_B,
            V_edge_B,
            V_flap_B,
            M_edge_B,
            M_flap_B,
            T_B,
            B_warp,
        )
        out.append((label, sp, rows))
    return out


def station_column_indices(
    n: int,
    *,
    station_stride: int = 1,
    max_cols: int | None = 12,
) -> np.ndarray:
    """Indices along the span grid (root→tip) for 3D figures when subsampling stations."""
    stride = max(1, int(station_stride))
    idx = np.arange(0, n, stride, dtype=int)
    if max_cols is not None and idx.size > int(max_cols):
        idx = np.unique(np.round(np.linspace(0, n - 1, int(max_cols))).astype(int))
    return idx


def _apply_proportional_3d_axes(
    ax,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    margin: float = 0.02,
) -> None:
    """Physical aspect ratios in matplotlib 3D (same idea as sweepkit ``span3d_figures``)."""

    def _pad(lo: float, hi: float) -> tuple[float, float]:
        if hi <= lo:
            mid = 0.5 * (lo + hi)
            return mid - 0.5, mid + 0.5
        span = hi - lo
        m = margin * span
        return lo - m, hi + m

    xmin, xmax = _pad(xmin, xmax)
    ymin, ymax = _pad(ymin, ymax)
    zmin, zmax = _pad(zmin, zmax)
    rx = max(float(xmax - xmin), 1e-9)
    ry = max(float(ymax - ymin), 1e-9)
    rz = max(float(zmax - zmin), 1e-9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_box_aspect([rx, ry, rz])


def _view_B_side_forward(ax) -> None:
    ax.view_init(
        elev=_VIEW_ELEV_B_SIDE_FORWARD,
        azim=_VIEW_AZIM_B_SIDE_FORWARD,
        roll=_VIEW_ROLL_B_SIDE_FORWARD,
    )
    ax.grid(False)


def _set_span3d_axis_labels(ax) -> None:
    ax.set_xlabel("Spanwise Z [m]", labelpad=8, fontsize=8)
    ax.set_ylabel("Edgewise Y_B [m]", labelpad=8, fontsize=8)
    ax.set_zlabel("Flapwise X_B [m]", labelpad=8, fontsize=8)


def _extent_from_points(xyz: np.ndarray) -> tuple[float, float, float, float, float, float]:
    if xyz.size == 0:
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (
        float(np.min(x)),
        float(np.max(x)),
        float(np.min(y)),
        float(np.max(y)),
        float(np.min(z)),
        float(np.max(z)),
    )


def _collect_panel_segments_mpl(
    stations: list[tuple[np.ndarray, tuple]],
    x_grid: np.ndarray,
    col_indices: Sequence[int],
    theta_at_x: np.ndarray,
    field: Literal["q", "sigma"],
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Build 3D segment array (n_seg, 2, 3) in matplotlib B-side-forward coords and midpoint
    scalar values (colour) per segment, for one web configuration.
    """
    x_grid = np.asarray(x_grid, dtype=float).ravel()
    theta_at_x = np.asarray(theta_at_x, dtype=float).ravel()
    seg_list: list[np.ndarray] = []
    val_list: list[float] = []
    for j in col_indices:
        _af, sec_out = stations[j]
        theta = float(theta_at_x[j])
        span = float(x_grid[j])
        panels = sec_out[0]
        q_tot = sec_out[3]
        sig_p = sec_out[4]
        for p, qv, sv in zip(panels, q_tot, sig_p):
            nodes = np.asarray(p.nodes, dtype=float)
            if field == "q":
                v_raw = np.asarray(qv, dtype=float).ravel()
            else:
                v_raw = np.asarray(sv, dtype=float).ravel()
            m = min(nodes.shape[0], v_raw.shape[0])
            if m < 2:
                continue
            v_t = np.asarray(transform(v_raw[:m]), dtype=float).ravel()
            for i in range(m - 1):
                v0, v1 = float(v_t[i]), float(v_t[i + 1])
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                seg2 = nodes[i : i + 2, :]
                pts3 = mpl_xyz_B_side_forward(np.full(2, span), seg2, theta)
                if not np.all(np.isfinite(pts3)):
                    continue
                seg_list.append(np.stack([pts3[0], pts3[1]], axis=0))
                val_list.append(0.5 * (v0 + v1))
    if not seg_list:
        return None, None
    return np.stack(seg_list, axis=0), np.asarray(val_list, dtype=float)


def _global_vmin_vmax_per_field(
    per_config: Sequence[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]],
    x_grid: np.ndarray,
    col_indices: Sequence[int],
    theta_at_x: np.ndarray,
    field: Literal["q", "sigma"],
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float]:
    vals: list[float] = []
    for _lb, _sp, stations in per_config:
        _, v = _collect_panel_segments_mpl(
            stations, x_grid, col_indices, theta_at_x, field, transform
        )
        if v is not None and v.size:
            vals.extend(v.tolist())
    if not vals:
        return 0.0, 1.0
    arr = np.asarray(vals, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def save_span3d_geometry_B_side_forward(
    per_config: Sequence[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]],
    x_grid: np.ndarray,
    *,
    col_indices: Sequence[int],
    theta_at_x: np.ndarray,
    outfile: str | Path,
    dpi: int = 150,
) -> None:
    """Outer airfoil polylines only — same layout as sweepkit ``span3d_geometry.png`` intent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lib.sectorial_warping import open_outline_from_airfoil

    x_grid = np.asarray(x_grid, dtype=float).ravel()
    theta_at_x = np.asarray(theta_at_x, dtype=float).ravel()
    n_cf = len(per_config)
    n_cols = min(3, max(1, int(np.ceil(np.sqrt(n_cf)))))
    n_row = int(np.ceil(n_cf / n_cols))
    fig = plt.figure(figsize=(9.5, 6.5), facecolor="white")
    all_pts: list[np.ndarray] = []

    for idx, (label, _sp, stations) in enumerate(per_config):
        ax = fig.add_subplot(n_row, n_cols, idx + 1, projection="3d")
        for j in col_indices:
            af_i, _ = stations[j]
            theta = float(theta_at_x[j])
            span = float(x_grid[j])
            ring = np.asarray(open_outline_from_airfoil(np.asarray(af_i, dtype=float)), dtype=float)
            if ring.shape[0] < 2:
                continue
            pts = mpl_xyz_B_side_forward(np.full(len(ring), span), ring, theta)
            all_pts.append(pts)
            ax.plot3D(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="#111111",
                linewidth=0.85,
                alpha=0.95,
            )
        _set_span3d_axis_labels(ax)
        ax.set_title(label, fontsize=9, pad=4)
        _view_B_side_forward(ax)

    if all_pts:
        big = np.vstack(all_pts)
        s0, s1, y0, y1, x0, x1 = _extent_from_points(big)
        for idx in range(n_cf):
            ax = fig.axes[idx]
            _apply_proportional_3d_axes(ax, s0, s1, y0, y1, x0, x1)

    fig.suptitle("B-frame stacked sections (twist visible) — outer airfoil", fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outfile, dpi=dpi, facecolor="white")
    plt.close()
    print(f"Saved: {outfile}")


def save_span3d_scalar_field_B_side_forward(
    per_config: Sequence[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]],
    x_grid: np.ndarray,
    *,
    col_indices: Sequence[int],
    theta_at_x: np.ndarray,
    field: Literal["q", "sigma"],
    outfile: str | Path,
    title: str,
    cbar_label: str,
    transform: Callable[[np.ndarray], np.ndarray],
    dpi: int = 150,
) -> None:
    """Per-panel ``Line3DCollection`` with unified colour scale (sweepkit-style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    x_grid = np.asarray(x_grid, dtype=float).ravel()
    theta_at_x = np.asarray(theta_at_x, dtype=float).ravel()
    vmin, vmax = _global_vmin_vmax_per_field(
        per_config, x_grid, col_indices, theta_at_x, field, transform
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    n_cf = len(per_config)
    n_cols = min(3, max(1, int(np.ceil(np.sqrt(n_cf)))))
    n_row = int(np.ceil(n_cf / n_cols))
    fig = plt.figure(figsize=(9.5, 6.5), facecolor="white")
    all_seg_pts: list[np.ndarray] = []
    axes_3d: list = []

    for idx, (_label, _sp, stations) in enumerate(per_config):
        ax = fig.add_subplot(n_row, n_cols, idx + 1, projection="3d")
        axes_3d.append(ax)
        segs, vals = _collect_panel_segments_mpl(
            stations, x_grid, col_indices, theta_at_x, field, transform
        )
        if segs is not None and vals is not None and len(segs) > 0:
            colors = cmap(norm(vals))
            lc = Line3DCollection(segs, colors=colors, linewidths=1.2, zorder=2)
            ax.add_collection3d(lc)
            all_seg_pts.append(segs.reshape(-1, 3))
        _set_span3d_axis_labels(ax)
        ax.set_title(_label, fontsize=8, pad=2)
        _view_B_side_forward(ax)

    if all_seg_pts:
        big = np.vstack(all_seg_pts)
        s0, s1, y0, y1, x0, x1 = _extent_from_points(big)
        for ax in axes_3d:
            _apply_proportional_3d_axes(ax, s0, s1, y0, y1, x0, x1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_3d, shrink=0.5, label=cbar_label, pad=0.02)
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.06)
    plt.savefig(outfile, dpi=dpi, facecolor="white")
    plt.close()
    print(f"Saved: {outfile}")


def save_span3d_outputs(
    out_dir: Path,
    per_config: Sequence[tuple[str, list[float], list[tuple[np.ndarray, tuple]]]],
    x_grid: np.ndarray,
    cfg: SpanRunConfig,
    *,
    station_stride: int = 1,
    max_cols: int | None = 12,
) -> None:
    """Write sweepkit-style B-side-forward 3D PNGs into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_grid = np.asarray(x_grid, dtype=float).ravel()
    n = len(x_grid)
    cols = station_column_indices(n, station_stride=station_stride, max_cols=max_cols)
    theta_at_x = eval_theta_geom_rad(cfg, x_grid)

    save_span3d_geometry_B_side_forward(
        per_config,
        x_grid,
        col_indices=cols,
        theta_at_x=theta_at_x,
        outfile=out_dir / "span3d_geometry.png",
    )
    save_span3d_scalar_field_B_side_forward(
        per_config,
        x_grid,
        col_indices=cols,
        theta_at_x=theta_at_x,
        field="q",
        outfile=out_dir / "span3d_q.png",
        title=r"Shear flow $q$ — unified scale across stations (B-side-forward)",
        cbar_label=r"$q$ (kN/m)",
        transform=lambda a: np.asarray(a, dtype=float) * 1e-3,
    )
    save_span3d_scalar_field_B_side_forward(
        per_config,
        x_grid,
        col_indices=cols,
        theta_at_x=theta_at_x,
        field="sigma",
        outfile=out_dir / "span3d_sigma.png",
        title=r"$\sigma$ — unified scale across stations (B-side-forward, skin panels)",
        cbar_label=r"$\sigma$ (MPa)",
        transform=lambda a: np.asarray(a, dtype=float) * 1e-6,
    )


if __name__ == "__main__":
    from lib.beam_vlasov_1d import solve_span_equilibrium
    from multi_cell_blade_section import naca_four_digit

    DEMO_WEB_CONFIGS: list[WebConfig] = [
        ([], "0 Webs — 1 Cell"),
        ([0.35], "1 Web  — 2 Cells  (@ 35%)"),
        ([0.25, 0.60], "2 Webs — 3 Cells  (@ 25%, 60%)"),
        ([0.20, 0.45, 0.70], "3 Webs — 4 Cells  (@ 20%, 45%, 70%)"),
        ([0.15, 0.35, 0.55, 0.75], "4 Webs — 5 Cells  (@ 15%, 35%, 55%, 75%)"),
        ([0.12, 0.28, 0.45, 0.62, 0.78], "5 Webs — 6 Cells"),
    ]

    L = 0.6
    x = np.linspace(0.0, L, 21)
    n = len(x)
    qx = np.zeros(n)
    p_edge = 50.0 * (1.0 - x / L)
    p_flap = np.zeros(n)
    mx = np.zeros(n)
    EIw = 2e3 * np.ones(n)
    GJ = 8e3 * np.ones(n)
    eq = solve_span_equilibrium(x, qx, p_edge, p_flap, mx, EIw, GJ)

    af = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=120)
    base = SpanRunConfig(
        L=L,
        x_grid=x,
        chord_m=0.25 * (1.0 - 0.4 * x / L),
        theta_geom_rad=0.15 * (x / L),
        spar_positions=[0.35],
        airfoil_norm=af,
    )

    per = run_span_all_web_configs(
        base,
        DEMO_WEB_CONFIGS,
        eq["N"],
        eq["V_edge"],
        eq["V_flap"],
        eq["M_edge"],
        eq["M_flap"],
        eq["T"],
        eq["B"],
    )

    out_dir = Path(__file__).resolve().parent / "outputs"
    save_span3d_outputs(
        out_dir,
        per,
        x,
        base,
        station_stride=2,
        max_cols=11,
    )
    print("blade_span_viz demo: OK")
