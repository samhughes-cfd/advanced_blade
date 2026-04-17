"""
Strip-graph stiffness-weighted first moments for transverse-shear screening.

This module complements :func:`~blade_precompute.section_properties.engine.interlaminar_recovery.recover_interlaminar`,
which applies **one** scalar ``EIy`` / ``EIz`` and **laminar** ``Q_axial(z)`` per edge without resolving how
global ``Vy``, ``Vz`` partition across the midsurface **line graph**.

Here we walk a **spanning tree** of the merged midsurface graph (see :func:`build_line_mesh`), accumulate
stiffness-weighted first-moment increments

    dS_z ≈ E_ax (b t L) z_rel,   dS_y ≈ E_ax (b t L) y_rel

with ``(y_rel, z_rel)`` taken about the **elastic centroid** (same bending axes as ``K6`` assembly in
:class:`~blade_precompute.section_properties.engine.solver.MidsurfaceSectionSolver`), and scale the
closed-form interlaminar envelope per edge. This is a **discrete thin-wall style bookkeeping** step, not
full multicell shear flow with twist compatibility.

Torque ``T`` and multicell circulations
--------------------------------------
``StripShearFlowSummary.q_torque_add`` is reserved for a future Bredt-style torque overlay on closed cells.
It is currently zero: coupling torque into ply-level ``σ_13``/``σ_23`` screening needs an explicit
cell-boundary walk and twist compatibility across cells, which is **not** implemented here. Multicell
circulations should use dedicated thin-wall solvers or higher-fidelity shell models.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.types import SectionSolveResult
from .geometry import SectionDefinition
from .interlaminar_recovery import (
    EdgeInterlaminarResult,
    InterfaceIFI,
    SectionInterlaminarResult,
    _governing_strengths,
    _recover_edge,
)
from .elements import StripElementData
from .laminate import LaminateDefinition
from .mesh import LineMesh
from .solver import _t_eff


def _edge_dS(
    section: SectionDefinition,
    fe: StripElementData,
    mesh: LineMesh,
    e: int,
    y_e: float,
    z_e: float,
) -> tuple[float, float]:
    """Stiffness-weighted first-moment increments (Vz / Vy channels) for edge ``e``."""
    i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
    y0, z0 = float(mesh.nodes[i0, 0]), float(mesh.nodes[i0, 1])
    y1, z1 = float(mesh.nodes[i1, 0]), float(mesh.nodes[i1, 1])
    y_m = 0.5 * (y0 + y1)
    z_m = 0.5 * (z0 + z1)
    y_r = y_m - y_e
    z_r = z_m - z_e
    si = int(fe.subcomp_idx[e])
    t = _t_eff(section, si)
    dvol = float(fe.b[e] * t * fe.L[e])
    w = float(fe.E_axial[e]) * dvol
    return w * z_r, w * y_r


def _mst_and_chords(n_n: int, edges_uv: NDArray[np.int32]) -> tuple[NDArray[np.bool_], List[int]]:
    """Kruskal MST on ``edges_uv`` (equal weights); return ``is_tree_edge``, ``chord_edge_indices``."""
    n_e = int(edges_uv.shape[0])
    parent = np.arange(n_n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    order = list(range(n_e))
    is_tree = np.zeros(n_e, dtype=bool)
    chords: List[int] = []
    for e in order:
        u, v = int(edges_uv[e, 0]), int(edges_uv[e, 1])
        if union(u, v):
            is_tree[e] = True
        else:
            chords.append(e)
    return is_tree, chords


def _adj_from_tree(
    n_n: int, edges_uv: NDArray[np.int32], is_tree: NDArray[np.bool_]
) -> List[List[Tuple[int, int]]]:
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_n)]
    for e in range(int(edges_uv.shape[0])):
        if not bool(is_tree[e]):
            continue
        u, v = int(edges_uv[e, 0]), int(edges_uv[e, 1])
        adj[u].append((v, e))
        adj[v].append((u, e))
    return adj


def _dfs_Q_on_tree(
    adj: List[List[Tuple[int, int]]],
    root: int,
    mesh: LineMesh,
    fe: StripElementData,
    section: SectionDefinition,
    y_e: float,
    z_e: float,
    Qz: NDArray[np.float64],
    Qy: NDArray[np.float64],
) -> None:
    """Fill ``Qz`` / ``Qy`` for tree edges reachable from ``root``."""

    def dfs(u: int, p: int) -> tuple[float, float]:
        acc_z = 0.0
        acc_y = 0.0
        for v, eid in adj[u]:
            if v == p:
                continue
            sub_z, sub_y = dfs(v, u)
            dz, dy = _edge_dS(section, fe, mesh, eid, y_e, z_e)
            qz_e = sub_z + dz
            qy_e = sub_y + dy
            Qz[eid] = qz_e
            Qy[eid] = qy_e
            acc_z += qz_e
            acc_y += qy_e
        return acc_z, acc_y

    dfs(root, -1)


def _collect_tree_component(start: int, tree_adj: List[List[Tuple[int, int]]], seen: NDArray[np.bool_]) -> List[int]:
    dq = deque([start])
    nodes: List[int] = []
    while dq:
        u = int(dq.popleft())
        if bool(seen[u]):
            continue
        seen[u] = True
        nodes.append(u)
        for v, _ in tree_adj[u]:
            if not bool(seen[v]):
                dq.append(v)
    return nodes


def _pick_root_subset(adj: List[List[Tuple[int, int]]], verts: List[int]) -> int:
    vset = set(verts)
    best = verts[0]
    best_deg = 10**9
    for u in verts:
        d = sum(1 for v, _ in adj[u] if v in vset)
        if d < best_deg:
            best_deg = d
            best = u
    return int(best)


@dataclass
class StripShearFlowSummary:
    """Diagnostics for strip-equilibrium scaling."""

    Qz_edge: NDArray[np.float64]
    Qy_edge: NDArray[np.float64]
    q_torque_add: NDArray[np.float64]
    scale13: NDArray[np.float64]
    scale23: NDArray[np.float64]
    #: ``T_applied + Vy (z_load - z_sc) - Vz (y_load - y_sc)`` with ``loads_at`` selecting ``(y_load,z_load)``.
    t_equivalent: float


def compute_strip_shear_flow_summary(
    mesh: LineMesh,
    fe: StripElementData,
    section: SectionDefinition,
    result: SectionSolveResult,
    vy: float,
    vz: float,
    t_applied: float,
    *,
    loads_at: str = "elastic",
) -> StripShearFlowSummary:
    """
    Build per-edge first-moment accumulators and interlaminar scale factors.

    ``loads_at`` controls the transport of ``(Vy, Vz)`` to the shear centre:
    ``"elastic"`` (default) applies the lever rule about ``result.elastic_center``.
    """
    n_e = int(mesh.edges.shape[0])
    if n_e == 0:
        empty = np.zeros(0, dtype=np.float64)
        return StripShearFlowSummary(empty, empty, empty, empty, empty, 0.0)

    y_e, z_e = float(result.elastic_center[0]), float(result.elastic_center[1])
    y_s, z_s = float(result.shear_center[0]), float(result.shear_center[1])
    loads_at_l = (loads_at or "elastic").strip().lower()
    if loads_at_l == "elastic":
        y_l, z_l = y_e, z_e
    elif loads_at_l == "shear":
        y_l, z_l = y_s, z_s
    else:
        raise ValueError("loads_at must be 'elastic' or 'shear'.")

    t_tot = float(
        t_applied + vy * (z_l - z_s) - vz * (y_l - y_s)
    )

    n_n = int(mesh.nodes.shape[0])
    is_tree, chords = _mst_and_chords(n_n, mesh.edges)
    tree_adj = _adj_from_tree(n_n, mesh.edges, is_tree)
    Qz = np.zeros(n_e, dtype=np.float64)
    Qy = np.zeros(n_e, dtype=np.float64)
    comp_seen = np.zeros(n_n, dtype=bool)
    for ni in range(n_n):
        if comp_seen[ni]:
            continue
        comp = _collect_tree_component(ni, tree_adj, comp_seen)
        root = _pick_root_subset(tree_adj, comp)
        _dfs_Q_on_tree(tree_adj, root, mesh, fe, section, y_e, z_e, Qz, Qy)

    q_torque = np.zeros(n_e, dtype=np.float64)

    # Chord edges: inherit average of adjacent tree-edge scalars (weak closure)
    for e in chords:
        u, v = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        inc_u = [Qz[ei] for _, ei in tree_adj[u] if ei != e]
        inc_v = [Qz[ei] for _, ei in tree_adj[v] if ei != e]
        qzu = float(np.mean(np.abs(inc_u))) if inc_u else 0.0
        qzv = float(np.mean(np.abs(inc_v))) if inc_v else 0.0
        Qz[e] = 0.5 * (qzu + qzv)
        inc_u = [Qy[ei] for _, ei in tree_adj[u] if ei != e]
        inc_v = [Qy[ei] for _, ei in tree_adj[v] if ei != e]
        qyu = float(np.mean(np.abs(inc_u))) if inc_u else 0.0
        qyv = float(np.mean(np.abs(inc_v))) if inc_v else 0.0
        Qy[e] = 0.5 * (qyu + qyv)

    q_ref_z = float(np.max(np.maximum(np.abs(Qz), 1e-30)))
    q_ref_y = float(np.maximum(np.max(np.abs(Qy)), 1e-30))
    scale13 = np.abs(Qz) / q_ref_z
    scale23 = np.abs(Qy) / q_ref_y
    return StripShearFlowSummary(
        Qz_edge=Qz,
        Qy_edge=Qy,
        q_torque_add=q_torque,
        scale13=scale13,
        scale23=scale23,
        t_equivalent=float(t_tot),
    )


def recover_interlaminar_strip_equilibrium(
    comp_edge_indices: List[int],
    lams: List[LaminateDefinition],
    mesh: LineMesh,
    fe: StripElementData,
    section: SectionDefinition,
    result: SectionSolveResult,
    vy: float,
    vz: float,
    eiy: float,
    eiz: float,
    t: float = 0.0,
    *,
    loads_at: str = "elastic",
) -> tuple[SectionInterlaminarResult, StripShearFlowSummary]:
    """
    Interlaminar screening with strip-graph scaling of ``recover_interlaminar`` envelopes.

    Returns the scaled :class:`SectionInterlaminarResult` plus a :class:`StripShearFlowSummary`
    for inspection. Torque ``t`` is combined with the transport of ``(Vy, Vz)`` to the shear centre.
    """
    summ = compute_strip_shear_flow_summary(mesh, fe, section, result, vy, vz, t, loads_at=loads_at)

    edge_results: List[EdgeInterlaminarResult] = []
    ifi_global = 0.0
    crit_edge = -1
    crit_z = 0.0

    for e_idx, lam in zip(comp_edge_indices, lams):
        base = _recover_edge(e_idx, lam, vy, vz, eiy, eiz)
        s13 = float(summ.scale13[e_idx]) if e_idx < summ.scale13.size else 1.0
        s23 = float(summ.scale23[e_idx]) if e_idx < summ.scale23.size else 1.0
        s13_g, s23_g = _governing_strengths(lam)
        interfaces: List[InterfaceIFI] = []
        ifi_max = 0.0
        z_crit = 0.0
        for ifc in base.interfaces:
            sig13 = ifc.sigma_13 * s13
            sig23 = ifc.sigma_23 * s23
            ifi = (sig13 / s13_g) ** 2 + (sig23 / s23_g) ** 2
            interfaces.append(
                InterfaceIFI(
                    z_interface=ifc.z_interface,
                    sigma_13=float(sig13),
                    sigma_23=float(sig23),
                    IFI=float(ifi),
                )
            )
            if ifi > ifi_max:
                ifi_max = ifi
                z_crit = ifc.z_interface
        edge_results.append(
            EdgeInterlaminarResult(edge_idx=e_idx, interfaces=interfaces, IFI_max=float(ifi_max), z_critical=z_crit)
        )
        if ifi_max > ifi_global:
            ifi_global = ifi_max
            crit_edge = e_idx
            crit_z = z_crit

    return (
        SectionInterlaminarResult(
            edge_results=edge_results,
            IFI_global=float(ifi_global),
            critical_edge=crit_edge,
            critical_z=crit_z,
        ),
        summ,
    )
