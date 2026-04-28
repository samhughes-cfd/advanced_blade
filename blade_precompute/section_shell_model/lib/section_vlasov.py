"""
E-weighted thin-wall sectorial Vlasov warping for a blade cross-section.

Physics
-------
For a closed thin-walled section, the warping function ω̂(s) must be
single-valued (continuous) around each cell.  The open-outline sectorial
coordinate ω_open (computed by ``omega_vertices_open_chain``) has a jump
discontinuity at the cut point equal to twice the cell area.  The
**Batho (1/t) correction** removes this jump:

  Single cell:
    c = ∮ ω_open (ds/t) / ∮ (ds/t)
    ω̂ = ω_open − c

  Multi-cell (n cells, solved simultaneously):
    Flexibility matrix:  δ_ii = ∮_i ds/t,   δ_ij = −∫_{shared web ij} ds/t
    RHS:                 R_i  = ∮_i ω_open ds/t
    System:              [δ] c = R  →  c_i per cell
    ω̂(s) = ω_open(s) − c_{cell(s)}

This replaces the t-weighted mean used in ``normalized_warping``, which is
only correct for sections with uniform thickness.

E-weighted warping constant
---------------------------
  I_ω_E = Σ_panels ∫ E_axial(s) ω̂²(s) t(s) ds   (correct units: Pa·m⁶)

Warping stresses (given bimoment B and its gradient dB/dx)
----------------------------------------------------------
  σ_ω(s) = B · ω̂(s) / I_ω_E                  [Pa]
  q_ω(s) = −(dB/dx / I_ω_E) ∫₀ˢ E ω̂ t ds'    [N/m]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _ensure_stress_imports() -> None:
    root = Path(__file__).resolve().parents[3] / "examples" / "section_stress_model"
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


@dataclass
class SectionVlasovResult:
    """Sectorial warping quantities for one cross-section."""

    y_sc: float              # shear centre y-coordinate [m]
    z_sc: float              # shear centre z-coordinate [m]
    omega_hat_v: NDArray     # ω̂ at outline vertices [m²] (n_verts,)
    I_omega_E: float         # E-weighted warping constant [Pa·m⁶ / E_ref]
    n_cells: int             # number of closed cells detected
    web_panel_indices: list[int]   # panel indices identified as spar webs
    # Per-panel arrays (one entry per panel, values at element mid-stations)
    panel_s_mids: list[NDArray]    # arc-length at element mids [m]
    sigma_omega: list[NDArray]     # warping normal stress [Pa] (zero for webs)
    q_omega: list[NDArray]         # secondary warping shear flow [N/m] (zero for webs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_section_vlasov(
    airfoil: NDArray,
    panels: list,
    B: float,
    dB_dx: float,
    webs_geom: list | None = None,
    t_web: float | None = None,
) -> SectionVlasovResult:
    """
    Compute E-weighted closed-cell sectorial warping for a blade cross-section.

    Parameters
    ----------
    airfoil   : (N, 2) array [y, z]
    panels    : Panel objects from ``multi_cell_blade_section``
    B         : bimoment [N·m²]
    dB_dx     : rate of bimoment along span [N·m]
    webs_geom : list of (upper_pt, lower_pt) tuples for spar webs; if provided,
                enables multi-cell Batho correction
    t_web     : representative web thickness [m] for flexibility integrals;
                defaults to the skin t_mean when omitted
    """
    _ensure_stress_imports()
    from lib.sectorial_warping import (  # type: ignore[import-untyped]
        open_outline_from_airfoil,
        shear_center_outer_skin_loop,
        omega_vertices_open_chain,
    )

    # --- weighted-mean panel properties -----------------------------------
    total_len, E_sum, t_sum = 0.0, 0.0, 0.0
    for p in panels:
        if len(p.s) < 2:
            continue
        L_p = float(p.s[-1])
        total_len += L_p
        E_sum += float(p.lam.E) * L_p
        t_sum += float(p.lam.t) * L_p
    E_mean = E_sum / max(total_len, 1e-30)
    t_mean = t_sum / max(total_len, 1e-30)
    if t_web is None:
        t_web = t_mean

    # 1. Open outline & shear centre
    verts = open_outline_from_airfoil(airfoil)           # (n_verts, 2)
    y_sc, z_sc, _, _ = shear_center_outer_skin_loop(verts, t_mean, E_mean)

    # 2. Open-chain sectorial coordinate (pole at SC, ω_open[0] = 0)
    omega_open = omega_vertices_open_chain(verts, y_sc, z_sc)

    # 3. Batho multi-cell correction → ω̂ continuous around each cell
    omega_hat_v, n_cells = _batho_correction(
        verts, omega_open, t_mean, webs_geom or [], t_web
    )

    # 4. E-weighted I_ω over all panels
    I_omega_E = _integrate_panels_E_omega2(panels, verts, omega_hat_v)
    if I_omega_E < 1e-60:
        I_omega_E = 1.0

    # 5. Per-panel stresses and shear flows
    s_outline = _outline_arc_lengths(verts)
    panel_s_mids, sigma_panels, q_panels = [], [], []
    web_panel_indices: list[int] = []
    cumulative_integral = 0.0

    for pi, p in enumerate(panels):
        n_elem = max(len(p.s) - 1, 0)
        if len(p.s) < 2:
            panel_s_mids.append(np.array([]))
            sigma_panels.append(np.array([]))
            q_panels.append(np.array([]))
            continue

        is_web = "web" in getattr(p, "label", "").lower()
        if is_web:
            web_panel_indices.append(pi)
            # Web warping function is not defined on the open skin outline —
            # set contributions to zero to avoid nonphysical interpolation.
            s_mids = 0.5 * (p.s[:-1] + p.s[1:])
            panel_s_mids.append(s_mids)
            sigma_panels.append(np.zeros(n_elem))
            q_panels.append(np.zeros(n_elem))
            continue

        E_p, t_p = float(p.lam.E), float(p.lam.t)
        s_global = _panel_global_s(p, verts)
        omega_nodes = np.interp(s_global, s_outline, omega_hat_v)

        sigma_w = B * omega_nodes / I_omega_E

        integral_panel = np.zeros(len(p.s))
        for i in range(1, len(p.s)):
            ds = p.s[i] - p.s[i - 1]
            om_mid = 0.5 * (omega_nodes[i] + omega_nodes[i - 1])
            integral_panel[i] = integral_panel[i - 1] + E_p * om_mid * t_p * ds
        q_w = -(dB_dx / I_omega_E) * (cumulative_integral + integral_panel)
        cumulative_integral += float(integral_panel[-1])

        s_mids = 0.5 * (p.s[:-1] + p.s[1:])
        panel_s_mids.append(s_mids)
        sigma_panels.append(0.5 * (sigma_w[:-1] + sigma_w[1:]))
        q_panels.append(0.5 * (q_w[:-1] + q_w[1:]))

    return SectionVlasovResult(
        y_sc=y_sc,
        z_sc=z_sc,
        omega_hat_v=omega_hat_v,
        I_omega_E=I_omega_E,
        n_cells=n_cells,
        web_panel_indices=web_panel_indices,
        panel_s_mids=panel_s_mids,
        sigma_omega=sigma_panels,
        q_omega=q_panels,
    )


# ---------------------------------------------------------------------------
# Batho multi-cell closed-section warping correction
# ---------------------------------------------------------------------------

def _outline_arc_lengths(verts: NDArray) -> NDArray:
    n = len(verts)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + float(np.hypot(verts[i, 0] - verts[i - 1, 0],
                                           verts[i, 1] - verts[i - 1, 1]))
    return s


def _batho_correction(
    verts: NDArray,
    omega_open: NDArray,
    t_skin: float,
    webs_geom: list,
    t_web: float,
) -> tuple[NDArray, int]:
    """
    Apply Batho (1/t)-weighted closed-cell warping correction.

    For a single-cell section (no webs), the unique correction is:
        c = ∮ ω_open (ds/t) / ∮ (ds/t)
    For multi-cell: solve the n_c × n_c flexibility system.

    Returns (omega_hat, n_cells).
    """
    s_v = _outline_arc_lengths(verts)
    n_v = len(verts)

    # Locate web split points on the outline (upper and lower intersections)
    web_splits = _web_split_indices(verts, s_v, webs_geom)  # list of (i_upper, i_lower)
    n_cells = len(web_splits) + 1

    if n_cells == 1:
        # Single-cell: analytic formula
        num, den = 0.0, 0.0
        for i in range(n_v - 1):
            ds = s_v[i + 1] - s_v[i]
            om_mid = 0.5 * (omega_open[i] + omega_open[i + 1])
            num += om_mid * ds / t_skin
            den += ds / t_skin
        # Closing edge (TE → LE): not in open outline; its contribution is small
        # (the open outline already accounts for the dominant skin area)
        c = num / max(den, 1e-60)
        return omega_open - c, 1

    # Multi-cell: identify the s-ranges belonging to each cell
    # Outline runs: LE → (split_upper_0) → … → TE → (split_lower_{n-1}) → LE (reversed lower)
    # We use the outline as-is and split it at the web upper and lower vertices.
    # Cell i: outline segment from web_{i-1}_lower to web_i_upper (upper skin portion)
    #         plus outline segment from web_i_lower to web_{i-1}_upper (lower skin portion)
    # This is approximate for a 2D thin-wall section.

    cell_segments = _cell_outline_segments(n_v, web_splits)

    # Flexibility matrix and RHS
    delta = np.zeros((n_cells, n_cells))
    R = np.zeros(n_cells)

    for ci, segs in enumerate(cell_segments):
        for (i0, i1) in segs:
            for i in range(min(i0, i1), max(i0, i1)):
                ds = abs(s_v[i + 1] - s_v[i])
                om_mid = 0.5 * (omega_open[i] + omega_open[i + 1])
                delta[ci, ci] += ds / t_skin
                R[ci] += om_mid * ds / t_skin

    # Web contributions: each web is shared between cell ci and ci+1
    for wi, (i_up, i_lo) in enumerate(web_splits):
        ci = wi  # web wi is between cell wi and wi+1
        cj = wi + 1
        # Web length (approximate: straight line from upper to lower web point)
        y_u, z_u = verts[i_up]
        y_l, z_l = verts[i_lo]
        L_web = float(np.hypot(y_u - y_l, z_u - z_l))
        flex_web = L_web / t_web
        delta[ci, ci] += flex_web
        delta[cj, cj] += flex_web
        delta[ci, cj] -= flex_web
        delta[cj, ci] -= flex_web
        # Web warping contribution: use mid-value of omega_open at web endpoints
        om_web_mid = 0.5 * (omega_open[i_up] + omega_open[i_lo])
        R[ci] += om_web_mid * flex_web
        R[cj] += om_web_mid * flex_web

    # Solve for correction constants (regularise if singular)
    try:
        c_vec = np.linalg.solve(delta, R)
    except np.linalg.LinAlgError:
        c_vec = np.zeros(n_cells)

    # Apply correction: each vertex belongs to a cell; assign correction by s-position
    vertex_cell = _assign_vertex_to_cell(n_v, web_splits)
    omega_hat = omega_open - c_vec[vertex_cell]

    return omega_hat, n_cells


def _web_split_indices(
    verts: NDArray, s_v: NDArray, webs_geom: list
) -> list[tuple[int, int]]:
    """
    For each web (upper_pt, lower_pt), find nearest outline vertex indices.
    Returns list of (i_upper, i_lower) sorted by upper s-coordinate.
    """
    splits = []
    for (up, lo) in webs_geom:
        up = np.asarray(up, dtype=float)
        lo = np.asarray(lo, dtype=float)
        i_up = int(np.argmin(np.hypot(verts[:, 0] - up[0], verts[:, 1] - up[1])))
        i_lo = int(np.argmin(np.hypot(verts[:, 0] - lo[0], verts[:, 1] - lo[1])))
        splits.append((i_up, i_lo))
    # Sort by the upper-vertex s-coordinate (ascending along outline)
    splits.sort(key=lambda p: s_v[p[0]])
    return splits


def _cell_outline_segments(
    n_v: int, web_splits: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """
    Build the list of (i_start, i_end) index pairs for each cell's outline portion.

    The open outline runs from index 0 (LE upper) to n_v-1 (LE lower).
    Webs split the outline at (i_up, i_lo) pairs.
    Cell 0 is from LE to the first web upper point (upper skin part)
    plus from the first web lower point to LE (lower skin part).
    """
    n_cells = len(web_splits) + 1
    # Collect all split indices on the outline (sorted ascending)
    split_up = sorted(p[0] for p in web_splits)
    split_lo = sorted((p[1] for p in web_splits), reverse=True)  # lower half goes backward

    # Upper boundary split points: 0, up_0, up_1, …, up_{n-1}, n_v//2
    # Lower boundary split points (reversed): n_v//2, lo_{n-1}, …, lo_0, n_v-1
    n_half = n_v // 2
    up_pts = [0] + split_up + [n_half]
    lo_pts = [n_half] + split_lo + [n_v - 1]

    cell_segs: list[list[tuple[int, int]]] = []
    for ci in range(n_cells):
        segs = []
        # Upper skin segment of cell ci
        i0_up, i1_up = up_pts[ci], up_pts[ci + 1]
        if i1_up > i0_up:
            segs.append((i0_up, i1_up))
        # Lower skin segment of cell ci (reversed: lo_pts goes from mid to LE)
        i0_lo, i1_lo = lo_pts[ci], lo_pts[ci + 1]
        if i1_lo > i0_lo:
            segs.append((i0_lo, i1_lo))
        cell_segs.append(segs)
    return cell_segs


def _assign_vertex_to_cell(n_v: int, web_splits: list[tuple[int, int]]) -> NDArray:
    """
    Return an int array of shape (n_v,) giving cell index for each outline vertex.
    """
    cell_idx = np.zeros(n_v, dtype=int)
    if not web_splits:
        return cell_idx

    split_up = sorted(p[0] for p in web_splits)
    n_half = n_v // 2
    up_pts = [0] + split_up + [n_half]
    split_lo = sorted((p[1] for p in web_splits), reverse=True)
    lo_pts = [n_half] + split_lo + [n_v - 1]

    n_cells = len(web_splits) + 1
    for ci in range(n_cells):
        for i in range(up_pts[ci], up_pts[ci + 1] + 1):
            if 0 <= i < n_v:
                cell_idx[i] = ci
        for i in range(lo_pts[ci], lo_pts[ci + 1] + 1):
            if 0 <= i < n_v:
                cell_idx[i] = ci
    return cell_idx


# ---------------------------------------------------------------------------
# Warping constant and panel mapping helpers
# ---------------------------------------------------------------------------

def _panel_global_s(panel, verts: NDArray) -> NDArray:
    """Map panel local arc-length to global open-outline arc-length."""
    s_outline = _outline_arc_lengths(verts)
    p0 = panel.nodes[0]
    dists = np.hypot(verts[:, 0] - p0[0], verts[:, 1] - p0[1])
    i_start = int(np.argmin(dists))
    return float(s_outline[i_start]) + panel.s


def _integrate_panels_E_omega2(panels: list, verts: NDArray, omega_hat_v: NDArray) -> float:
    """I_ω_E = Σ_panels ∫ E_p ω̂² t_p ds (trapezoid)."""
    s_outline = _outline_arc_lengths(verts)
    acc = 0.0
    for p in panels:
        if len(p.s) < 2:
            continue
        E_p, t_p = float(p.lam.E), float(p.lam.t)
        s_global = _panel_global_s(p, verts)
        omega_nodes = np.interp(s_global, s_outline, omega_hat_v)
        acc += float(np.trapezoid(E_p * omega_nodes ** 2 * t_p, p.s))
    return acc
