from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mitc4_element import (
    mitc4_edge_resultants,
    mitc4_edge_shear_traction_integrated,
    mitc4_resultants,
    mitc4_stiffness,
)
from .types import FieldProvenance, ProvenanceKind, ShellPanelResultants

_NDOF_NODE = 5
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4


def _panel_mesh(s_panel: np.ndarray, n_elements: int) -> tuple[np.ndarray, list[list[int]]]:
    s_min, s_max = float(s_panel.min()), float(s_panel.max())
    s_nodes = np.linspace(s_min, s_max, n_elements + 1)
    n = len(s_nodes)
    elems: list[list[int]] = []
    for i in range(n - 1):
        i0 = i
        i1 = i + 1
        i2 = i + 1 + n
        i3 = i + n
        elems.append([i0, i1, i2, i3])
    return s_nodes, elems


def _panel_endpoints(nodes_yz: np.ndarray) -> dict[str, np.ndarray]:
    return {"start": np.asarray(nodes_yz[0], dtype=float), "end": np.asarray(nodes_yz[-1], dtype=float)}


def _panel_end_tangent(nodes_yz: np.ndarray, which: str) -> np.ndarray:
    if len(nodes_yz) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if which == "start":
        t = np.asarray(nodes_yz[1] - nodes_yz[0], dtype=float)
    else:
        t = np.asarray(nodes_yz[-1] - nodes_yz[-2], dtype=float)
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return t / n


def _cluster_points(points: list[tuple[int, str, np.ndarray]], tol: float) -> list[list[tuple[int, str, np.ndarray]]]:
    clusters: list[list[tuple[int, str, np.ndarray]]] = []
    for p in points:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(p[2] - c[0][2])) <= tol:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters


def _dof(node_idx: int, local: int) -> int:
    return node_idx * _NDOF_NODE + local


def _normal_2d(t: np.ndarray) -> np.ndarray:
    """90° CCW rotation of a 2-D tangent unit vector → panel outward normal in (Y,Z)."""
    return np.array([-float(t[1]), float(t[0])], dtype=float)


# Type alias for the unified multi-master constraint format.
# Each slave DOF maps to a list of (master_dof, coefficient) pairs so that
# q_slave = sum_k coeff_k * q_master_k.
_Constraints = dict[int, list[tuple[int, float]]]


def _build_transform_constraints(
    panel_maps: list[_PanelGlobalMap],
    panels: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
) -> _Constraints:
    """
    Build scalar-ratio MPC constraints (legacy scalar path, used by ``"transformed"`` mode).

    Each slave DOF is a scalar multiple of one master DOF.  This is retained for
    backward compatibility; the physically correct block constraints are in
    ``_build_basis_transform_constraints``.
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 2:
            continue
        pi_m, end_m, _ = cluster[0]
        pm_m = panel_maps[pi_m]
        n_m = len(pm_m.s_nodes)
        row_m = 0 if end_m == "start" else n_m - 1
        t_m = _panel_end_tangent(np.asarray(getattr(panels[pi_m], "nodes"), dtype=float), end_m)
        t_ref = t_m
        c_m = float(np.dot(t_m, t_ref))
        c_m = 1.0 if abs(c_m) < 1e-12 else c_m
        for pi_s, end_s, _ in cluster[1:]:
            pm_s = panel_maps[pi_s]
            n_s = len(pm_s.s_nodes)
            row_s = 0 if end_s == "start" else n_s - 1
            t_s = _panel_end_tangent(np.asarray(getattr(panels[pi_s], "nodes"), dtype=float), end_s)
            c_s = float(np.dot(t_s, t_ref))
            c_s = 1.0 if abs(c_s) < 1e-12 else c_s
            coeff_s = c_m / c_s
            for off_m, off_s in ((row_m, row_s), (row_m + n_m, row_s + n_s)):
                gn_m = pm_m.global_nodes[off_m]
                gn_s = pm_s.global_nodes[off_s]
                constraints[_dof(gn_s, _U_X)] = [(_dof(gn_m, _U_X), 1.0)]
                constraints[_dof(gn_s, _U_S)] = [(_dof(gn_m, _U_S), coeff_s)]
                constraints[_dof(gn_s, _W)] = [(_dof(gn_m, _W), 1.0)]
                constraints[_dof(gn_s, _BETA_S)] = [(_dof(gn_m, _BETA_S), coeff_s)]
                constraints[_dof(gn_s, _BETA_X)] = [(_dof(gn_m, _BETA_X), 1.0)]
    return constraints


def _build_basis_transform_constraints(
    panel_maps: list[_PanelGlobalMap],
    panels: list[Any],
    endpoint_clusters: list[list[tuple[int, str, np.ndarray]]],
) -> _Constraints:
    """
    Build physically correct 2-D kinematic compatibility MPCs.

    For each junction cluster the master panel (first in cluster) defines a local
    (u_s, w) basis.  All slave panels' endpoint DOFs are expressed in terms of the
    master's DOFs via the 2-D rotation:

      u_s_s = (ŝ_s·ŝ_m) u_s_m + (ŝ_s·n̂_m) w_m
      w_s   = (n̂_s·ŝ_m) u_s_m + (n̂_s·n̂_m) w_m

    where n̂ = CCW_90(ŝ) is the panel's out-of-plane direction in the (Y,Z) plane.

    Spanwise (u_x) and drilling (β_x) DOFs are shared 1:1 since X is common to all
    panels.  The tangential rotation β_s is projected by (ŝ_s·ŝ_m).
    """
    constraints: _Constraints = {}
    for cluster in endpoint_clusters:
        if len(cluster) < 2:
            continue
        pi_m, end_m, _ = cluster[0]
        pm_m = panel_maps[pi_m]
        n_m = len(pm_m.s_nodes)
        row_m = 0 if end_m == "start" else n_m - 1
        nodes_m = np.asarray(getattr(panels[pi_m], "nodes"), dtype=float)
        t_m = _panel_end_tangent(nodes_m, end_m)
        n_m_hat = _normal_2d(t_m)

        for pi_s, end_s, _ in cluster[1:]:
            pm_s = panel_maps[pi_s]
            n_s = len(pm_s.s_nodes)
            row_s = 0 if end_s == "start" else n_s - 1
            nodes_s = np.asarray(getattr(panels[pi_s], "nodes"), dtype=float)
            t_s = _panel_end_tangent(nodes_s, end_s)
            n_s_hat = _normal_2d(t_s)

            # 2×2 rotation matrix entries: R = [[ts_tm, ts_nm],[ns_tm, ns_nm]]
            # q_slave = R * q_master   for (u_s, w) block.
            ts_tm = float(np.dot(t_s, t_m))
            ts_nm = float(np.dot(t_s, n_m_hat))
            ns_tm = float(np.dot(n_s_hat, t_m))
            ns_nm = float(np.dot(n_s_hat, n_m_hat))

            for off_m, off_s in ((row_m, row_s), (row_m + n_m, row_s + n_s)):
                gn_m = pm_m.global_nodes[off_m]
                gn_s = pm_s.global_nodes[off_s]

                # Spanwise: u_x is common to all panels (no rotation of X-axis).
                constraints[_dof(gn_s, _U_X)] = [(_dof(gn_m, _U_X), 1.0)]

                # 2-D in-plane displacement: u_s_s = ts_tm*u_s_m + ts_nm*w_m
                us_terms = [(_dof(gn_m, _U_S), ts_tm)]
                if abs(ts_nm) > 1e-12:
                    us_terms.append((_dof(gn_m, _W), ts_nm))
                constraints[_dof(gn_s, _U_S)] = us_terms

                # 2-D out-of-plane: w_s = ns_tm*u_s_m + ns_nm*w_m
                w_terms = [(_dof(gn_m, _U_S), ns_tm)]
                if abs(ns_nm) > 1e-12:
                    w_terms.append((_dof(gn_m, _W), ns_nm))
                constraints[_dof(gn_s, _W)] = w_terms

                # Rotation about span: β_x is common.
                constraints[_dof(gn_s, _BETA_X)] = [(_dof(gn_m, _BETA_X), 1.0)]

                # Tangential rotation: project β_s by the tangent dot product.
                # β_s represents rotation about the panel tangent; its (Y,Z) component
                # along the slave tangent is (ŝ_s·ŝ_m)*β_s_m.
                constraints[_dof(gn_s, _BETA_S)] = [(_dof(gn_m, _BETA_S), ts_tm)]

    return constraints


@dataclass
class _PanelGlobalMap:
    s_nodes: np.ndarray
    elements: list[list[int]]
    global_nodes: list[int]  # local node -> global node id
    panel_label: str
    panel_index: int


def solve_global_coupled_mitc4(
    panels: list[Any],
    Nx_panels: list[np.ndarray],
    Nxy_panels: list[np.ndarray],
    *,
    n_elements_per_panel: int = 10,
    endpoint_tol: float = 1e-6,
    bc_mode: str = "legacy",
    interface_constraint_mode: str = "shared",
) -> tuple[list[list[ShellPanelResultants]], list[dict]]:
    """
    Assemble and solve all panel strips in one global system with shared endpoint nodes.
    """
    panel_maps: list[_PanelGlobalMap] = []
    endpoints: list[tuple[int, str, np.ndarray]] = []

    # Build local strip meshes per panel.
    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(_PanelGlobalMap(np.array([]), [], [], str(getattr(p, "label", f"panel_{pi}")), pi))
            continue
        s_nodes, elems = _panel_mesh(s_panel, n_elements_per_panel)
        n_local_nodes = 2 * len(s_nodes)
        global_nodes = [-1] * n_local_nodes
        panel_maps.append(_PanelGlobalMap(s_nodes, elems, global_nodes, str(getattr(p, "label", f"panel_{pi}")), pi))
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        ep = _panel_endpoints(nodes_yz)
        endpoints.append((pi, "start", ep["start"]))
        endpoints.append((pi, "end", ep["end"]))

    # Interface topology from endpoint clusters.
    clusters = _cluster_points(endpoints, endpoint_tol)
    endpoint_cluster_index: dict[tuple[int, str], int] = {}
    for cluster_id, c in enumerate(clusters):
        for pi, which, _ in c:
            endpoint_cluster_index[(pi, which)] = cluster_id

    next_global_node = 0
    if interface_constraint_mode == "shared":
        endpoint_global_node: dict[tuple[int, str, str], int] = {}
        for c in clusters:
            g_bottom = next_global_node
            g_top = next_global_node + 1
            next_global_node += 2
            for pi, which, _ in c:
                endpoint_global_node[(pi, which, "bottom")] = g_bottom
                endpoint_global_node[(pi, which, "top")] = g_top
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            n = len(pm.s_nodes)
            pm.global_nodes[0] = endpoint_global_node[(pm.panel_index, "start", "bottom")]
            pm.global_nodes[n - 1] = endpoint_global_node[(pm.panel_index, "end", "bottom")]
            pm.global_nodes[n] = endpoint_global_node[(pm.panel_index, "start", "top")]
            pm.global_nodes[2 * n - 1] = endpoint_global_node[(pm.panel_index, "end", "top")]
            for li in range(1, n - 1):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1
            for li in range(n + 1, 2 * n - 1):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1
    else:
        for pm in panel_maps:
            if len(pm.s_nodes) == 0:
                continue
            for li in range(2 * len(pm.s_nodes)):
                pm.global_nodes[li] = next_global_node
                next_global_node += 1

    n_gdof = next_global_node * _NDOF_NODE
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    f_global = np.zeros(n_gdof)
    panel_load_totals: list[dict] = []

    # Assemble K and f.
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            panel_load_totals.append({})
            continue
        p = panels[pi]
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        # Build ABD per panel from laminate
        from lib.laminate_clpt import abd_stack  # type: ignore[import-untyped]
        A_mat, B_mat, D_mat = abd_stack(p.lam.build_plies())
        ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])
        thickness = float(p.lam.t)
        nu = float(p.lam.nu)
        E_p = float(p.lam.E)
        G_eff = E_p / (2.0 * (1.0 + nu))

        s_nodes = pm.s_nodes
        Nx_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nx_panels[pi], dtype=float))
        Nxy_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nxy_panels[pi], dtype=float))
        fx_total = 0.0
        fs_total = 0.0

        for e in pm.elements:
            i0, i1, i2, i3 = e
            L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
            if L_s < 1e-30:
                continue
            Ke = mitc4_stiffness(L_s, 1.0, ABD, thickness, G_eff=G_eff)
            local_nodes = [i0, i1, i2, i3]
            g_dofs: list[int] = []
            for ln in local_nodes:
                gn = pm.global_nodes[ln]
                for d in range(_NDOF_NODE):
                    g_dofs.append(_dof(gn, d))
            for a, ga in enumerate(g_dofs):
                for b, gb in enumerate(g_dofs):
                    rows.append(ga)
                    cols.append(gb)
                    vals.append(float(Ke[a, b]))

            Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
            Nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])
            f_node_x = Nx_avg * L_s / 2.0
            f_node_s = Nxy_avg * L_s / 2.0
            for ln in (i2, i3):
                gn = pm.global_nodes[ln]
                f_global[_dof(gn, _U_X)] += f_node_x
                f_global[_dof(gn, _U_S)] += f_node_s
            fx_total += 2.0 * f_node_x
            fs_total += 2.0 * f_node_s

        panel_load_totals.append({"Fx_total": fx_total, "Fs_total": fs_total})

        # Donnell curvature correction: q_n(s) = Nx(s) · κ(s) [N/m²]
        # Mirror the per-panel _assemble_curvature_loads + _panel_curvature logic so
        # the global solve sees the same constitutive load path as solve_panel_mitc4.
        if len(nodes_yz) >= 3:
            # Curvature κ at each s-panel node (tangent-angle derivative).
            y_p, z_p = nodes_yz[:, 0], nodes_yz[:, 1]
            s_panel_full = np.asarray(p.s, dtype=float)
            dy_p, dz_p = np.diff(y_p), np.diff(z_p)
            ds_p = np.maximum(np.hypot(dy_p, dz_p), 1e-30)
            theta_p = np.arctan2(dz_p, dy_p)
            kappa_p = np.zeros(len(nodes_yz))
            for ii in range(1, len(nodes_yz) - 1):
                dth = theta_p[ii] - theta_p[ii - 1]
                dth = (dth + np.pi) % (2 * np.pi) - np.pi
                arc = 0.5 * (ds_p[ii - 1] + ds_p[ii])
                kappa_p[ii] = dth / arc if arc > 1e-30 else 0.0
            if len(kappa_p) > 1:
                kappa_p[0] = kappa_p[1]
                kappa_p[-1] = kappa_p[-2]
            # Interpolate κ onto the solver s_nodes.
            kappa_nodes = np.interp(s_nodes, s_panel_full, kappa_p)
            # Consistent nodal load for each element.
            for e in pm.elements:
                i0, i1, i2, i3 = e
                L_s = float(abs(s_nodes[i1] - s_nodes[i0]))
                if L_s < 1e-30:
                    continue
                Nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
                kappa_avg = 0.5 * (kappa_nodes[i0] + kappa_nodes[i1])
                q_n = Nx_avg * kappa_avg          # [N/m²]
                f_node_w = q_n * L_s * 1.0 / 4.0  # L_x = 1 for unit slice
                for ln in (i0, i1, i2, i3):
                    gn = pm.global_nodes[ln]
                    f_global[_dof(gn, _W)] += f_node_w

    K_global = sp.coo_matrix((vals, (rows, cols)), shape=(n_gdof, n_gdof)).tocsr()

    # Build compatibility constraints based on selected mode.
    if interface_constraint_mode == "shared":
        constraints: _Constraints = {}
    elif interface_constraint_mode == "transformed_basis":
        constraints = _build_basis_transform_constraints(panel_maps, panels, clusters)
    else:
        # "transformed" mode: legacy scalar-ratio constraints.
        constraints = _build_transform_constraints(panel_maps, panels, clusters)

    # Global supports: minimal RBM + optional legacy endpoint support.
    # In transformed_basis mode, slave endpoint DOFs are constrained via MPC;
    # applying a direct BC to them would conflict with the constraint (the constraint
    # is silently skipped when a slave is fixed).  We therefore identify slave endpoint
    # DOFs and skip the direct BC so the MPC propagates the master's BC instead.
    _slave_endpoint_dofs: set[int] = set()
    if interface_constraint_mode == "transformed_basis":
        for slave_dof in constraints:
            _slave_endpoint_dofs.add(slave_dof)

    fixed: set[int] = set()
    if next_global_node > 1:
        fixed.add(_dof(0, _U_X))
        fixed.add(_dof(0, _U_S))
        fixed.add(_dof(1, _U_X))
    for pm in panel_maps:
        if len(pm.s_nodes) == 0:
            continue
        n = len(pm.s_nodes)
        if bc_mode == "legacy":
            support_nodes = (0, n - 1, n, 2 * n - 1)
        else:
            support_nodes = (0, n - 1)
        for ln in support_nodes:
            gn = pm.global_nodes[ln]
            for dof_type in (_W, _BETA_S, _BETA_X):
                dof = _dof(gn, dof_type)
                if dof not in _slave_endpoint_dofs:
                    fixed.add(dof)

    # Apply panel-level static consistency correction to match integrated targets.
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            continue
        p = panels[pi]
        s_nodes = pm.s_nodes
        Nx_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nx_panels[pi], dtype=float))
        Nxy_nodes = np.interp(s_nodes, np.asarray(p.s, dtype=float), np.asarray(Nxy_panels[pi], dtype=float))
        target_fx = float(np.trapezoid(Nx_nodes, s_nodes))
        target_fs = float(np.trapezoid(Nxy_nodes, s_nodes))
        panel_load_totals[pi]["Fx_target"] = target_fx
        panel_load_totals[pi]["Fs_target"] = target_fs
        d_fx = target_fx - float(panel_load_totals[pi].get("Fx_total", 0.0))
        d_fs = target_fs - float(panel_load_totals[pi].get("Fs_total", 0.0))
        if abs(d_fx) > 0.0 or abs(d_fs) > 0.0:
            n = len(s_nodes)
            w = np.zeros(n)
            for i in range(n):
                if i == 0:
                    w[i] = 0.5 * abs(s_nodes[1] - s_nodes[0]) if n > 1 else 1.0
                elif i == n - 1:
                    w[i] = 0.5 * abs(s_nodes[-1] - s_nodes[-2])
                else:
                    w[i] = 0.5 * abs(s_nodes[i] - s_nodes[i - 1]) + 0.5 * abs(s_nodes[i + 1] - s_nodes[i])
            w_sum = float(np.sum(w))
            w = np.ones(n) if w_sum < 1e-30 else w
            w_sum = float(np.sum(w))
            for i in range(n):
                frac = float(w[i] / w_sum)
                gn = pm.global_nodes[i + n]
                f_global[_dof(gn, _U_X)] += d_fx * frac
                f_global[_dof(gn, _U_S)] += d_fs * frac
            panel_load_totals[pi]["Fx_total"] = target_fx
            panel_load_totals[pi]["Fs_total"] = target_fs
            panel_load_totals[pi]["Fx_correction"] = d_fx
            panel_load_totals[pi]["Fs_correction"] = d_fs

    fixed_sorted = sorted(fixed)
    constrained_slaves = set(constraints.keys())
    # Each constraint value is a list of (master_dof, coeff) pairs.
    constrained_masters = {m for terms in constraints.values() for m, _ in terms}
    all_idx = np.arange(n_gdof)
    independent = np.array([i for i in all_idx if i not in fixed and i not in constrained_slaves], dtype=int)
    red_pos = {dof: i for i, dof in enumerate(independent.tolist())}
    t_rows: list[int] = []
    t_cols: list[int] = []
    t_vals: list[float] = []
    for dof in independent.tolist():
        t_rows.append(dof)
        t_cols.append(red_pos[dof])
        t_vals.append(1.0)
    for slave, terms in constraints.items():
        if slave in fixed:
            continue
        for master, coeff in terms:
            if master in red_pos:
                t_rows.append(slave)
                t_cols.append(red_pos[master])
                t_vals.append(float(coeff))
    T = sp.coo_matrix((t_vals, (t_rows, t_cols)), shape=(n_gdof, len(independent))).tocsr()
    d_global = np.zeros(n_gdof)
    if len(independent) > 0:
        K_rr = (T.T @ K_global @ T).tocsc()
        f_r = np.asarray(T.T @ f_global, dtype=float)
        q = spla.spsolve(K_rr, f_r)
        d_global = np.asarray(T @ q, dtype=float)
    r_full = np.asarray(K_global @ d_global - f_global, dtype=float)

    # Recover per-panel results + diagnostics.
    all_panel_results: list[list[ShellPanelResultants]] = []
    all_panel_diag: list[dict] = []
    for pi, pm in enumerate(panel_maps):
        if len(pm.s_nodes) == 0:
            all_panel_results.append([])
            all_panel_diag.append({})
            continue
        p = panels[pi]
        from lib.laminate_clpt import abd_stack  # type: ignore[import-untyped]
        A_mat, B_mat, D_mat = abd_stack(p.lam.build_plies())
        ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])
        panel_label = pm.panel_label
        results: list[ShellPanelResultants] = []
        edge_start_acc: list[float] = []
        edge_end_acc: list[float] = []
        edge_start_s_acc: list[float] = []
        edge_end_s_acc: list[float] = []
        edge_start_tx_acc: list[float] = []
        edge_end_tx_acc: list[float] = []
        edge_start_ts_acc: list[float] = []
        edge_end_ts_acc: list[float] = []
        prov = {
            "Nx": FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane x-resultant"),
            "Ny": FieldProvenance(ProvenanceKind.MITC4, "MITC4 membrane s-resultant"),
            "Nxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 in-plane shear resultant"),
            "Mx": FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (x-axis)"),
            "My": FieldProvenance(ProvenanceKind.MITC4, "MITC4 bending moment (s-axis)"),
            "Mxy": FieldProvenance(ProvenanceKind.MITC4, "MITC4 twisting moment"),
            "Qx": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
            "Qy": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / future"),
        }
        for station_idx, e in enumerate(pm.elements):
            i0, i1, i2, i3 = e
            L_s = float(abs(pm.s_nodes[i1] - pm.s_nodes[i0]))
            d_elem = np.zeros(20)
            for local_i, ln in enumerate([i0, i1, i2, i3]):
                gn = pm.global_nodes[ln]
                for dd in range(_NDOF_NODE):
                    d_elem[local_i * _NDOF_NODE + dd] = d_global[_dof(gn, dd)]
            res = mitc4_resultants(d_elem, L_s, 1.0, ABD)
            edge = mitc4_edge_resultants(d_elem, L_s, 1.0, ABD)
            edge_start_acc.append(float(edge["start"]["Nx"]))
            edge_end_acc.append(float(edge["end"]["Nx"]))
            edge_start_s_acc.append(float(edge["start"]["Nxy"]))
            edge_end_s_acc.append(float(edge["end"]["Nxy"]))
            edge_int_start = mitc4_edge_shear_traction_integrated(
                d_elem, L_s, 1.0, ABD, edge="start", gauss_n=3
            )
            edge_int_end = mitc4_edge_shear_traction_integrated(
                d_elem, L_s, 1.0, ABD, edge="end", gauss_n=3
            )
            edge_start_tx_acc.append(float(edge_int_start["Tx_edge_int"]))
            edge_start_ts_acc.append(float(edge_int_start["Ts_edge_int"]))
            edge_end_tx_acc.append(float(edge_int_end["Tx_edge_int"]))
            edge_end_ts_acc.append(float(edge_int_end["Ts_edge_int"]))
            results.append(
                ShellPanelResultants(
                    Nx=res["Nx"],
                    Ny=res["Ny"],
                    Nxy=res["Nxy"],
                    Mx=res["Mx"],
                    My=res["My"],
                    Mxy=res["Mxy"],
                    Qx=None,
                    Qy=None,
                    provenance=dict(prov),
                    sigma_xx_pa=res["Nx"] / max(ABD[0, 0], 1e-30),
                    tau_xy_pa=res["Nxy"],
                    q_n_per_m=res["Nxy"],
                    thickness_m=0.0,
                    panel_label=panel_label,
                    panel_index=pi,
                    station_index=station_idx,
                )
            )
        all_panel_results.append(results)

        n = len(pm.s_nodes)
        start_nodes = [pm.global_nodes[0], pm.global_nodes[n]]
        end_nodes = [pm.global_nodes[n - 1], pm.global_nodes[2 * n - 1]]
        start_fx = float(sum(r_full[_dof(gn, _U_X)] for gn in start_nodes))
        start_fs = float(sum(r_full[_dof(gn, _U_S)] for gn in start_nodes))
        end_fx = float(sum(r_full[_dof(gn, _U_X)] for gn in end_nodes))
        end_fs = float(sum(r_full[_dof(gn, _U_S)] for gn in end_nodes))
        edge_start_nx = float(edge_start_acc[0]) if edge_start_acc else 0.0
        edge_end_nx = float(edge_end_acc[-1]) if edge_end_acc else 0.0
        edge_start_nxy = float(edge_start_s_acc[0]) if edge_start_s_acc else 0.0
        edge_end_nxy = float(edge_end_s_acc[-1]) if edge_end_s_acc else 0.0
        edge_start_nxy_int = float(edge_start_tx_acc[0]) if edge_start_tx_acc else 0.0
        edge_end_nxy_int = float(edge_end_tx_acc[-1]) if edge_end_tx_acc else 0.0
        edge_start_ts_int = float(edge_start_ts_acc[0]) if edge_start_ts_acc else 0.0
        edge_end_ts_int = float(edge_end_ts_acc[-1]) if edge_end_ts_acc else 0.0
        # Blend edge-resultant recovery with nodal interface-force recovery to
        # stabilize secondary continuity diagnostics at multi-way junctions.
        alpha_force = 0.75
        field_start_nx = float(alpha_force * start_fx + (1.0 - alpha_force) * edge_start_nx)
        field_end_nx = float(alpha_force * end_fx + (1.0 - alpha_force) * edge_end_nx)
        # Nxy secondary field is now taken from edge line-integration directly.
        field_start_nxy = edge_start_nxy_int
        field_end_nxy = edge_end_nxy_int
        all_panel_diag.append(
            {
                "endpoint_cluster_index": {
                    "start": int(endpoint_cluster_index.get((pi, "start"), -1)),
                    "end": int(endpoint_cluster_index.get((pi, "end"), -1)),
                },
                "panel_end_tangent": {
                    "start": _panel_end_tangent(nodes_yz, "start").tolist(),
                    "end": _panel_end_tangent(nodes_yz, "end").tolist(),
                },
                "boundary_reaction_set": {
                    "start": {"Fx": start_fx, "Fs": start_fs},
                    "end": {"Fx": end_fx, "Fs": end_fs},
                },
                "interface_field_set": {
                    "start": {"Nx": field_start_nx, "Nxy": field_start_nxy},
                    "end": {"Nx": field_end_nx, "Nxy": field_end_nxy},
                },
                "interface_edge_set": {
                    "start": {
                        "Nx": edge_start_nx,
                        "Nxy": edge_start_nxy,
                        "Nxy_int": edge_start_nxy_int,
                        "Tx_int": edge_start_nxy_int,
                        "Ts_int": edge_start_ts_int,
                    },
                    "end": {
                        "Nx": edge_end_nx,
                        "Nxy": edge_end_nxy,
                        "Nxy_int": edge_end_nxy_int,
                        "Tx_int": edge_end_nxy_int,
                        "Ts_int": edge_end_ts_int,
                    },
                },
                "load_totals": panel_load_totals[pi],
                "residual": {
                    "free_res_rel": float(np.linalg.norm(r_full[independent]) / max(np.linalg.norm(f_global[independent]), 1e-12)),
                    "global_force_mismatch_rel": float(np.linalg.norm(r_full) / max(np.linalg.norm(f_global), 1e-12)),
                },
                "constraint_stats": {
                    "n_reduced_dofs": int(len(independent)),
                    "n_slave_dofs": int(len(constrained_slaves)),
                    "n_master_dofs": int(len(constrained_masters)),
                },
                "solver": "global_coupled",
            }
        )
    return all_panel_results, all_panel_diag
