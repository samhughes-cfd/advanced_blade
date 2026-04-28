"""
beam_model/assembly.py
======================
Global assembly, boundary conditions, external load vector (7 DOFs / node).
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .element import (
    _lagrange_gauss,
    _shape,
    element_energy_gradient,
    element_hessian_fd,
    element_stiffness_geometric_stress,
    element_stiffness_material,
    gp_z_coord,
)
from .interp import interp_K7
from .kinematics import quat_to_rotmat
from ..core.types import BeamModel, BeamLoads, BoundaryCondition, NodeState, SectionStation


def dof_base(node_id: int) -> int:
    return 7 * int(node_id)


def scatter_el_vec(
    el_dofs: Sequence[int] | NDArray[np.int64],
    v_el: NDArray[np.float64],
    v_glob: NDArray[np.float64],
) -> None:
    idx = el_dofs if isinstance(el_dofs, np.ndarray) else np.asarray(el_dofs, dtype=np.int64)
    v_glob[idx] += v_el


def scatter_el_mat(
    el_dofs: Sequence[int] | NDArray[np.int64],
    K_el: NDArray[np.float64],
    K_glob: NDArray[np.float64],
) -> None:
    idx = el_dofs if isinstance(el_dofs, np.ndarray) else np.asarray(el_dofs, dtype=np.int64)
    K_glob[np.ix_(idx, idx)] += K_el


def element_dofs(el) -> List[int]:
    i, j = el.node_ids
    return [dof_base(i) + k for k in range(7)] + [dof_base(j) + k for k in range(7)]


def fixed_dof_set(bcs: List[BoundaryCondition]) -> Set[int]:
    s: Set[int] = set()
    for bc in bcs:
        b = dof_base(bc.node_id)
        for d in bc.fixed_dofs:
            s.add(b + int(d))
    return s


def external_load_vector(
    model: BeamModel,
    loads: BeamLoads,
    n_gauss: int,
    nodes: List[NodeState] | None = None,
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    F = np.zeros(ndof, dtype=np.float64)
    n = model.n_nodes
    Ff = np.asarray(loads.nodal_F, dtype=np.float64)
    Mm = np.asarray(loads.nodal_M, dtype=np.float64)
    if Ff.shape != (n, 3) or Mm.shape != (n, 3):
        raise ValueError("nodal_F and nodal_M must have shape (n_nodes, 3).")
    use_corot = getattr(loads, "frame", "undeformed_global") == "node_corotated"
    for inode in range(n):
        if use_corot:
            if nodes is None:
                raise ValueError(
                    "nodes are required when BeamLoads.frame == 'node_corotated'."
                )
            R = quat_to_rotmat(nodes[inode].q)
            F[7 * inode : 7 * inode + 3] = R @ Ff[inode]
            F[7 * inode + 3 : 7 * inode + 6] = R @ Mm[inode]
        else:
            F[7 * inode : 7 * inode + 3] = Ff[inode]
            F[7 * inode + 3 : 7 * inode + 6] = Mm[inode]

    dq = loads.distributed_q
    if dq is not None:
        q_tab = np.asarray(dq, dtype=np.float64)
        xi_w, w_w = _lagrange_gauss(n_gauss)
        for e_id, el in enumerate(model.elements):
            if q_tab.ndim == 1:
                qv = q_tab
            elif q_tab.ndim == 2 and q_tab.shape[0] == len(model.elements):
                qv = q_tab[e_id]
            else:
                raise ValueError("distributed_q must be (3,) or (n_elem, 3).")
            L0 = el.L0
            i, j = el.node_ids
            Q1 = np.zeros(3, dtype=np.float64)
            Q2 = np.zeros(3, dtype=np.float64)
            for xi, w in zip(xi_w, w_w):
                N1, N2, _, _ = _shape(float(xi))
                fac = 0.5 * L0 * float(w)
                Q1 += fac * N1 * qv
                Q2 += fac * N2 * qv
            F[7 * i : 7 * i + 3] += Q1
            F[7 * j : 7 * j + 3] += Q2

    dmz = loads.distributed_mz
    if dmz is not None:
        mz = np.asarray(dmz, dtype=np.float64)
        for e_id, el in enumerate(model.elements):
            L0 = el.L0
            i, j = el.node_ids
            m = float(mz) if mz.ndim == 0 else float(mz[e_id] if mz.shape[0] > e_id else mz[0])
            lump = 0.5 * L0 * m
            # Distributed torsion acts on rotational DOF 3 (about beam/local-x),
            # not on the warping amplitude DOF 6.
            F[7 * i + 3] += lump
            F[7 * j + 3] += lump

    return F


def _nodes_with_bump(
    base: List[NodeState], node_id: int, k_dof: int, h: float
) -> List[NodeState]:
    from .kinematics import update_orientation

    out = [ns.copy() for ns in base]
    p = out[node_id]
    if k_dof < 3:
        p.x = p.x.copy()
        p.x[k_dof] += h
    elif k_dof < 6:
        e = np.zeros(3, dtype=np.float64)
        e[k_dof - 3] = 1.0
        p.q = update_orientation(p.q, h * e)
    else:
        p.psi = float(p.psi) + h
    return out


def external_load_jacobian_fd(
    model: BeamModel,
    loads: BeamLoads,
    nodes: List[NodeState],
    n_gauss: int,
    eps: float,
) -> NDArray[np.float64]:
    """
    ``-∂F_ext/∂q`` from central differences (used in the tangent for follower / corotated loads).
    """
    from .kinematics import update_orientation

    ndof = 7 * model.n_nodes
    h = max(float(eps), 1e-10)
    J = np.zeros((ndof, ndof), dtype=np.float64)
    for n in range(model.n_nodes):
        for k in range(7):
            col = 7 * n + k
            snap_p = _nodes_with_bump(nodes, n, k, h)
            snap_m = _nodes_with_bump(nodes, n, k, -h)
            Fp = external_load_vector(model, loads, n_gauss, nodes=snap_p)
            Fm = external_load_vector(model, loads, n_gauss, nodes=snap_m)
            dF = (Fp - Fm) / (2.0 * h)
            J[:, col] = dF
    return -J


def _precompute_K7_gp(
    model: BeamModel,
    stations: List[SectionStation],
    n_gauss: int,
) -> List[List[NDArray[np.float64]]]:
    """Precompute K7 at every element Gauss point (constant for a fixed design during NR)."""
    xi_pts, _ = _lagrange_gauss(n_gauss)
    result: List[List[NDArray[np.float64]]] = []
    for el in model.elements:
        el_gps: List[NDArray[np.float64]] = []
        for xi in xi_pts:
            zg = gp_z_coord(model, el, float(xi))
            K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
            el_gps.append(K7)
        result.append(el_gps)
    return result


def assemble_gradient(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    K7_gp: List[List[NDArray[np.float64]]] | None = None,
    use_cs: bool = True,
    h_cs: float = 1e-20,
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    g = np.zeros(ndof, dtype=np.float64)
    el_dof_idx = [np.array(element_dofs(el), dtype=np.int64) for el in model.elements]
    for ei, el in enumerate(model.elements):
        K7_el = K7_gp[ei] if K7_gp is not None else None
        ge = element_energy_gradient(
            model, el, nodes, stations, n_gauss, fd_h, K7_el, use_cs=use_cs, h_cs=h_cs
        )
        scatter_el_vec(el_dof_idx[ei], ge, g)
    return g


def assemble_geometric_stiffness(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    jac_eps: float,
    spin_eps: float,
    warping_eps: float,
    K7_gp: List[List[NDArray[np.float64]]] | None = None,
) -> NDArray[np.float64]:
    """Assemble only the stress (initial) geometric stiffness from ``rᵀ d²e``."""
    ndof = 7 * model.n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    el_dof_idx = [np.array(element_dofs(el), dtype=np.int64) for el in model.elements]
    for ei, el in enumerate(model.elements):
        K7_el = K7_gp[ei] if K7_gp is not None else None
        Ke = element_stiffness_geometric_stress(
            model, el, nodes, stations, n_gauss, jac_eps, K7_el
        )
        scatter_el_mat(el_dof_idx[ei], Ke, K)
    if spin_eps > 0.0:
        for n in range(model.n_nodes):
            for k in range(3):
                d = 7 * n + 3 + k
                K[d, d] += spin_eps
    if warping_eps > 0.0:
        for n in range(model.n_nodes):
            d = 7 * n + 6
            K[d, d] += warping_eps
    return K


def assemble_hessian(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    hess_eps: float,
    full_fd: bool,
    jac_eps: float,
    spin_eps: float,
    warping_eps: float,
    K7_gp: List[List[NDArray[np.float64]]] | None = None,
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    el_dof_idx = [np.array(element_dofs(el), dtype=np.int64) for el in model.elements]
    for ei, el in enumerate(model.elements):
        K7_el = K7_gp[ei] if K7_gp is not None else None
        if full_fd:
            Ke = element_hessian_fd(model, el, nodes, stations, n_gauss, fd_h, hess_eps, K7_el)
        else:
            Km = element_stiffness_material(
                model, el, nodes, stations, n_gauss, fd_h, jac_eps, K7_el
            )
            Kg = element_stiffness_geometric_stress(
                model, el, nodes, stations, n_gauss, jac_eps, K7_el
            )
            Ke = Km + Kg
        scatter_el_mat(el_dof_idx[ei], Ke, K)
    if spin_eps > 0.0:
        for n in range(model.n_nodes):
            for k in range(3):
                d = 7 * n + 3 + k
                K[d, d] += spin_eps
    if warping_eps > 0.0:
        for n in range(model.n_nodes):
            d = 7 * n + 6
            K[d, d] += warping_eps
    return K


def reduce_linear_system(
    K: NDArray[np.float64],
    rhs: NDArray[np.float64],
    fixed: Set[int],
    free: NDArray[np.int64] | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    ndof = rhs.shape[0]
    if free is not None:
        free = np.asarray(free, dtype=np.int64).ravel()
    else:
        free = np.array([i for i in range(ndof) if i not in fixed], dtype=np.int64)
    K_ff = K[np.ix_(free, free)]
    rhs_f = rhs[free]
    return K_ff, rhs_f, free


def expand_solution(free_idx: NDArray[np.int64], du_f: NDArray[np.float64], ndof: int) -> NDArray[np.float64]:
    du = np.zeros(ndof, dtype=np.float64)
    du[free_idx] = du_f
    return du
