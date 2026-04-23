"""
Panel-level MITC4 shell model.

Builds a strip of MITC4 elements along one skin panel's contour, applies the
thin-wall Nx(s) and Nxy(s) distributions as distributed boundary loads, and
solves for the displacement field. Shell resultants (all 6 components) are then
recovered at each element centroid, replacing the MVP placeholder zeros.

Coordinate conventions
-----------------------
  s   — arc-length along panel contour (physical, metres)
  x   — span / beam-axis direction (η in element coords)

For a cross-section unit-slice analysis the span length L_x = 1.0 m.

Curved panel (Donnell approximation)
--------------------------------------
When panel node coordinates ``nodes_yz`` are supplied, the geometric curvature
κ(s) = dθ/ds is computed from successive tangent angles θ = atan2(dz, dy).
This induces a distributed normal load:

    q_n(s) = Nx(s) · κ(s)    [N/m²]

which is added to the out-of-plane (w) load vector — the Donnell shallow-shell
approximation.  For flat panels κ ≈ 0 and this term vanishes.

Boundary conditions
--------------------
At each spar attachment (by arc-length proximity), the out-of-plane DOF w and
the two rotations β_s, β_x of the closest node are pinned (simply supported).
The bottom face (η = −1) has u_x and u_s fixed as the reference frame.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .mitc4_element import mitc4_resultants, mitc4_stiffness
from .types import FieldProvenance, ProvenanceKind, ShellPanelResultants


# ---------------------------------------------------------------------------
# DOF layout helpers
# ---------------------------------------------------------------------------

_NDOF_NODE = 5   # [u_x, u_s, w, β_s, β_x]
_U_X  = 0
_U_S  = 1
_W    = 2
_BETA_S = 3
_BETA_X = 4


def _node_dof(node_idx: int, local_dof: int) -> int:
    return node_idx * _NDOF_NODE + local_dof


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def _build_mesh(s_nodes: NDArray) -> tuple[NDArray, list[list[int]]]:
    """
    Build the 2*(N_e+1) node mesh for a 1-element-deep (η) strip.

    Returns
    -------
    nodes   : (2*(N_e+1), 2) array of (s, x) coordinates; bottom row x=0, top x=1
    elements: list of N_e element node-index lists [i0, i1, i2, i3] (CCW)
    """
    n_nodes_s = len(s_nodes)   # N_e + 1
    coords = np.zeros((2 * n_nodes_s, 2))
    coords[:n_nodes_s, 0] = s_nodes        # bottom row  (x = 0)
    coords[:n_nodes_s, 1] = 0.0
    coords[n_nodes_s:, 0] = s_nodes        # top row     (x = 1)
    coords[n_nodes_s:, 1] = 1.0

    elems = []
    for i in range(n_nodes_s - 1):
        i0 = i
        i1 = i + 1
        i2 = i + 1 + n_nodes_s
        i3 = i + n_nodes_s
        elems.append([i0, i1, i2, i3])   # CCW: bottom-left, bottom-right, top-right, top-left

    return coords, elems


# ---------------------------------------------------------------------------
# Global stiffness assembly
# ---------------------------------------------------------------------------

def _assemble(
    s_nodes: NDArray,
    elements: list[list[int]],
    ABD: NDArray,
    thickness: float,
    G_eff: float,
    L_x: float = 1.0,
) -> sp.csr_matrix:
    n_nodes = len(s_nodes) * 2
    n_dof = n_nodes * _NDOF_NODE

    rows, cols, vals = [], [], []

    for elem_nodes in elements:
        i0, i1 = elem_nodes[0], elem_nodes[1]
        L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
        if L_s < 1e-30:
            continue

        Ke = mitc4_stiffness(L_s, L_x, ABD, thickness, G_eff=G_eff)

        # Element DOF list: [i0_bottom, i1_bottom, i1_top, i0_top] × 5
        glob_dofs = []
        for ni in elem_nodes:
            for d in range(_NDOF_NODE):
                glob_dofs.append(_node_dof(ni, d))

        for a, ga in enumerate(glob_dofs):
            for b, gb in enumerate(glob_dofs):
                rows.append(ga)
                cols.append(gb)
                vals.append(Ke[a, b])

    K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    return K


# ---------------------------------------------------------------------------
# Load vector: consistent nodal forces from distributed Nx(s) and Nxy(s)
# ---------------------------------------------------------------------------

def _assemble_loads(
    s_nodes: NDArray,
    elements: list[list[int]],
    Nx_nodes: NDArray,
    Nxy_nodes: NDArray,
    L_x: float = 1.0,
) -> NDArray:
    """
    Apply Nx and Nxy as consistent nodal tractions on the η = +1 (top, x = L_x) face.

    The bottom face (η = −1, x = 0) is the fixed reference frame — its u_x and u_s
    DOFs are pinned in _fixed_dofs_for_panel.  Only the top face nodes (i2, i3) receive
    traction loads.  Consistent 2-node edge force: each node gets N_avg × L_s / 2.
    """
    n_nodes_s = len(s_nodes)
    n_dof = 2 * n_nodes_s * _NDOF_NODE
    f = np.zeros(n_dof)
    fx_total = 0.0
    fs_total = 0.0

    for elem_nodes in elements:
        i0, i1, i2, i3 = elem_nodes
        # CCW ordering: i0=bot-left, i1=bot-right, i2=top-right, i3=top-left
        L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
        if L_s < 1e-30:
            continue

        Nx_avg  = 0.5 * (Nx_nodes[i0]  + Nx_nodes[i1])
        Nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])

        # Top face (η=+1): nodes i2 (ξ=+1) and i3 (ξ=−1).
        # Consistent nodal force for linear 2-node edge: N_avg × L_s / 2 each.
        force_node = Nx_avg * L_s / 2.0
        shear_node = Nxy_avg * L_s / 2.0
        f[_node_dof(i2, _U_X)] += force_node
        f[_node_dof(i3, _U_X)] += force_node
        f[_node_dof(i2, _U_S)] += shear_node
        f[_node_dof(i3, _U_S)] += shear_node
        fx_total += 2.0 * force_node
        fs_total += 2.0 * shear_node

    # Static-consistency correction: enforce exact integrated resultants from nodal
    # Nx(s), Nxy(s) profiles while keeping the same top-face loading strategy.
    target_fx = float(np.trapezoid(Nx_nodes, s_nodes) * L_x)
    target_fs = float(np.trapezoid(Nxy_nodes, s_nodes) * L_x)
    d_fx = target_fx - fx_total
    d_fs = target_fs - fs_total
    if abs(d_fx) > 0.0 or abs(d_fs) > 0.0:
        w = np.zeros(n_nodes_s)
        for i in range(n_nodes_s):
            if i == 0:
                w[i] = 0.5 * abs(s_nodes[1] - s_nodes[0]) if n_nodes_s > 1 else 1.0
            elif i == n_nodes_s - 1:
                w[i] = 0.5 * abs(s_nodes[-1] - s_nodes[-2])
            else:
                w[i] = 0.5 * abs(s_nodes[i] - s_nodes[i - 1]) + 0.5 * abs(s_nodes[i + 1] - s_nodes[i])
        w_sum = float(np.sum(w))
        if w_sum < 1e-30:
            w = np.ones(n_nodes_s)
            w_sum = float(n_nodes_s)
        for i in range(n_nodes_s):
            frac = float(w[i] / w_sum)
            top_node = i + n_nodes_s
            f[_node_dof(top_node, _U_X)] += d_fx * frac
            f[_node_dof(top_node, _U_S)] += d_fs * frac
        fx_total = target_fx
        fs_total = target_fs

    return f, {"Fx_total": fx_total, "Fs_total": fs_total, "Fx_target": target_fx, "Fs_target": target_fs}


def _assemble_curvature_loads(
    s_nodes: NDArray,
    elements: list[list[int]],
    Nx_nodes: NDArray,
    kappa_nodes: NDArray,
    L_x: float = 1.0,
) -> NDArray:
    """
    Add Donnell curvature-induced lateral load q_n = Nx · κ to the w DOF vector.

    q_n [N/m²] acts in the shell-normal direction (out-of-plane w).
    Consistent nodal force for a rectangular element: q_n_avg × L_s × L_x / 4
    applied to each of the 4 nodes.
    """
    n_nodes_s = len(s_nodes)
    n_dof = 2 * n_nodes_s * _NDOF_NODE
    f = np.zeros(n_dof)

    for elem_nodes in elements:
        i0, i1, i2, i3 = elem_nodes
        L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
        if L_s < 1e-30:
            continue

        Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
        kappa_avg = 0.5 * (kappa_nodes[i0] + kappa_nodes[i1])
        q_n = Nx_avg * kappa_avg           # [N/m²] normal pressure from curvature

        # Consistent uniform pressure on rectangle: Q/4 per node
        f_node = q_n * L_s * L_x / 4.0
        for ni in (i0, i1, i2, i3):
            f[_node_dof(ni, _W)] += f_node

    return f


def _panel_curvature(nodes_yz: NDArray, s_panel: NDArray) -> NDArray:
    """
    Curvature κ(s) = dθ/ds at each panel node, where θ = atan2(dz, dy) is the
    tangent angle in the section (y, z) plane.

    For n nodes there are n−1 tangent segments; κ is estimated at nodes by
    central differencing of consecutive tangent angles, with boundary values
    extrapolated from the nearest interior value.
    """
    n = len(nodes_yz)
    if n < 3:
        return np.zeros(n)

    y, z = nodes_yz[:, 0], nodes_yz[:, 1]
    dy, dz = np.diff(y), np.diff(z)
    ds = np.hypot(dy, dz)
    ds = np.maximum(ds, 1e-30)
    theta = np.arctan2(dz, dy)   # tangent angle at each segment midpoint

    # κ at interior nodes: central difference of θ over arc length
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        dtheta = theta[i] - theta[i - 1]
        # Wrap to [−π, π]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        arc = 0.5 * (ds[i - 1] + ds[i])
        kappa[i] = dtheta / arc

    # Boundary nodes: copy nearest interior value
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def _apply_bcs(
    K: sp.csr_matrix,
    f: NDArray,
    fixed_dofs: list[int],
) -> tuple[sp.csr_matrix, NDArray]:
    """Penalty method (large stiffness) for zero displacement BCs."""
    K_lil = K.tolil()
    penalty = float(K.diagonal().max()) * 1e12
    for dof in fixed_dofs:
        K_lil[dof, :] = 0.0
        K_lil[:, dof] = 0.0
        K_lil[dof, dof] = penalty
        f[dof] = 0.0
    return K_lil.tocsr(), f


def _fixed_dofs_for_panel(
    s_nodes: NDArray,
    n_nodes_s: int,
    spar_s_coords: list[float],
    *,
    bc_mode: str = "minimal_rbm",
) -> list[int]:
    """
    BCs for the unit-slice panel model.

    Bottom face (η=−1, nodes 0…n_nodes_s−1): fix u_x and u_s — these are the
    reference frame for the in-plane problem.  The top face receives applied loads.

    At each spar attachment node (both top and bottom rows): pin w, β_s, β_x —
    simple support in the out-of-plane direction.
    """
    fixed: list[int] = []

    # In-plane rigid-body suppression (default) or legacy full-bottom clamp.
    if bc_mode == "full_bottom_clamp":
        for i in range(n_nodes_s):
            fixed.append(_node_dof(i, _U_X))
            fixed.append(_node_dof(i, _U_S))
    else:
        # Minimal constraints: anchor one node ux/us and a second node ux.
        i0 = 0
        i1 = max(n_nodes_s - 1, 0)
        fixed.extend([
            _node_dof(i0, _U_X),
            _node_dof(i0, _U_S),
            _node_dof(i1, _U_X),
        ])

    # Simple support (w, β) at spar attachment locations, interpolated by bracketing
    # nodes to reduce nearest-node locking artifacts.
    for s_spar in spar_s_coords:
        idx_r = int(np.searchsorted(s_nodes, float(s_spar), side="left"))
        idx_l = max(0, idx_r - 1)
        idx_r = min(n_nodes_s - 1, idx_r)
        nodes_pair = sorted(set([idx_l, idx_r]))
        for idx in nodes_pair:
            for row in (idx, idx + n_nodes_s):
                fixed.append(_node_dof(row, _W))
                fixed.append(_node_dof(row, _BETA_S))
                fixed.append(_node_dof(row, _BETA_X))

    return list(set(fixed))


# ---------------------------------------------------------------------------
# Per-element resultant recovery
# ---------------------------------------------------------------------------

def _recover_resultants(
    d_full: NDArray,
    s_nodes: NDArray,
    elements: list[list[int]],
    ABD: NDArray,
    panel_label: str,
    panel_index: int,
    L_x: float = 1.0,
) -> list[ShellPanelResultants]:
    results = []
    n_nodes_s = len(s_nodes)

    prov = {
        "Nx":  FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane x-resultant"),
        "Ny":  FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane s-resultant"),
        "Nxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 in-plane shear resultant"),
        "Mx":  FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (x-axis)"),
        "My":  FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (s-axis)"),
        "Mxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 twisting moment"),
        "Qx":  FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
        "Qy":  FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
    }

    for station_idx, elem_nodes in enumerate(elements):
        i0, i1 = elem_nodes[0], elem_nodes[1]
        L_s = float(abs(s_nodes[i1] - s_nodes[i0]))

        # Gather element displacement vector (20 DOFs)
        d_elem = np.zeros(20)
        for local_i, ni in enumerate(elem_nodes):
            for d in range(_NDOF_NODE):
                d_elem[local_i * _NDOF_NODE + d] = d_full[_node_dof(ni, d)]

        res = mitc4_resultants(d_elem, L_s, L_x, ABD)

        results.append(ShellPanelResultants(
            Nx=res["Nx"],
            Ny=res["Ny"],
            Nxy=res["Nxy"],
            Mx=res["Mx"],
            My=res["My"],
            Mxy=res["Mxy"],
            Qx=None,
            Qy=None,
            provenance=dict(prov),
            sigma_xx_pa=res["Nx"] / max(ABD[0, 0] / 1.0, 1e-30),  # diagnostic only
            tau_xy_pa=res["Nxy"],
            q_n_per_m=res["Nxy"],
            thickness_m=0.0,
            panel_label=panel_label,
            panel_index=panel_index,
            station_index=station_idx,
        ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_panel_mitc4(
    ABD: NDArray,
    thickness: float,
    G_eff: float,
    s_panel: NDArray,
    Nx_panel: NDArray,
    Nxy_panel: NDArray,
    spar_s_coords: list[float],
    panel_label: str = "",
    panel_index: int = 0,
    *,
    nodes_yz: NDArray | None = None,
    n_elements: int = 10,
    L_x: float = 1.0,
    return_diagnostics: bool = False,
    bc_mode: str = "minimal_rbm",
) -> list[ShellPanelResultants] | tuple[list[ShellPanelResultants], dict]:
    """
    Solve MITC4 panel model and return per-element shell resultants.

    Parameters
    ----------
    ABD          : 6×6 laminate stiffness [[A, B],[B, D]]
    thickness    : wall thickness [m]
    G_eff        : effective transverse shear modulus [Pa]
    s_panel      : arc-length coordinates of available data stations [m]
    Nx_panel     : Nx [N/m] at each station in s_panel
    Nxy_panel    : Nxy [N/m] at each station in s_panel
    spar_s_coords: arc-length positions of spar/boundary attachments [m]
    nodes_yz     : (n_stations, 2) [y, z] panel node coordinates; enables curvature loads
    n_elements   : number of MITC4 elements along contour
    L_x          : span length of unit slice [m] (default 1.0)
    """
    s_min, s_max = float(s_panel.min()), float(s_panel.max())
    s_nodes = np.linspace(s_min, s_max, n_elements + 1)

    Nx_nodes  = np.interp(s_nodes, s_panel, Nx_panel)
    Nxy_nodes = np.interp(s_nodes, s_panel, Nxy_panel)

    n_nodes_s = len(s_nodes)
    coords, elements = _build_mesh(s_nodes)

    K = _assemble(s_nodes, elements, ABD, thickness, G_eff, L_x)
    f, load_diag = _assemble_loads(s_nodes, elements, Nx_nodes, Nxy_nodes, L_x)

    # Curvature-induced lateral load (Donnell approximation)
    if nodes_yz is not None and len(nodes_yz) >= 3:
        kappa_panel = _panel_curvature(np.asarray(nodes_yz, dtype=float), s_panel)
        kappa_nodes = np.interp(s_nodes, s_panel, kappa_panel)
        f += _assemble_curvature_loads(s_nodes, elements, Nx_nodes, kappa_nodes, L_x)

    fixed = _fixed_dofs_for_panel(s_nodes, n_nodes_s, spar_s_coords, bc_mode=bc_mode)
    K_bc, f_bc = _apply_bcs(K, f, fixed)

    d = spla.spsolve(K_bc.tocsc(), f_bc)

    results = _recover_resultants(d, s_nodes, elements, ABD, panel_label, panel_index, L_x)
    if not return_diagnostics:
        return results

    r_full = np.asarray(K @ d - f, dtype=float)
    fixed_mask = np.zeros_like(r_full, dtype=bool)
    for dof in fixed:
        fixed_mask[dof] = True
    free_mask = ~fixed_mask
    free_res = r_full[free_mask]
    free_res_norm = float(np.linalg.norm(free_res))
    free_load_norm = float(np.linalg.norm(f[free_mask]))
    free_res_rel = free_res_norm / max(free_load_norm, 1e-12)

    n_nodes = len(s_nodes)
    node_b0 = 0
    node_b1 = n_nodes - 1
    node_t0 = n_nodes
    node_t1 = 2 * n_nodes - 1

    def _ux(node: int) -> int:
        return _node_dof(node, _U_X)

    def _us(node: int) -> int:
        return _node_dof(node, _U_S)

    start_nodes = [node_b0, node_t0]
    end_nodes = [node_b1, node_t1]
    start_fx_set = float(sum(r_full[_ux(n)] for n in start_nodes))
    start_fs_set = float(sum(r_full[_us(n)] for n in start_nodes))
    end_fx_set = float(sum(r_full[_ux(n)] for n in end_nodes))
    end_fs_set = float(sum(r_full[_us(n)] for n in end_nodes))

    diag = {
        "load_totals": {
            "Fx_total": float(load_diag["Fx_total"]),
            "Fs_total": float(load_diag["Fs_total"]),
        },
        "residual": {
            "free_res_norm": free_res_norm,
            "free_res_rel": free_res_rel,
        },
        "boundary_reaction": {
            "start": {
                "Fx": float(r_full[_ux(node_b0)]),
                "Fs": float(r_full[_us(node_b0)]),
            },
            "end": {
                "Fx": float(r_full[_ux(node_b1)]),
                "Fs": float(r_full[_us(node_b1)]),
            },
        },
        "boundary_reaction_set": {
            "start": {"Fx": start_fx_set, "Fs": start_fs_set},
            "end": {"Fx": end_fx_set, "Fs": end_fs_set},
        },
        "boundary_applied_top": {
            "start": {
                "Fx": float(f[_ux(node_t0)]),
                "Fs": float(f[_us(node_t0)]),
            },
            "end": {
                "Fx": float(f[_ux(node_t1)]),
                "Fs": float(f[_us(node_t1)]),
            },
        },
        "bc_mode": bc_mode,
    }
    return results, diag
