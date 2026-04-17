"""
beam_model/assembly.py
======================
Global assembly, boundary conditions, external load vector (7 DOFs / node).
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .element import (
    _lagrange_gauss,
    _shape,
    element_energy_gradient,
    element_hessian_fd,
    element_stiffness_material,
)
from ..core.types import BeamModel, BeamLoads, BoundaryCondition, NodeState, SectionStation


def dof_base(node_id: int) -> int:
    return 7 * int(node_id)


def scatter_el_vec(el_dofs: List[int], v_el: NDArray[np.float64], v_glob: NDArray[np.float64]) -> None:
    for a, idx in enumerate(el_dofs):
        v_glob[idx] += v_el[a]


def scatter_el_mat(
    el_dofs: List[int],
    K_el: NDArray[np.float64],
    K_glob: NDArray[np.float64],
) -> None:
    for a, ia in enumerate(el_dofs):
        for b, ib in enumerate(el_dofs):
            K_glob[ia, ib] += K_el[a, b]


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
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    F = np.zeros(ndof, dtype=np.float64)
    n = model.n_nodes
    Ff = np.asarray(loads.nodal_F, dtype=np.float64)
    Mm = np.asarray(loads.nodal_M, dtype=np.float64)
    if Ff.shape != (n, 3) or Mm.shape != (n, 3):
        raise ValueError("nodal_F and nodal_M must have shape (n_nodes, 3).")
    for inode in range(n):
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
            F[7 * i + 6] += lump
            F[7 * j + 6] += lump

    return F


def assemble_gradient(
    model: BeamModel,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    g = np.zeros(ndof, dtype=np.float64)
    for el in model.elements:
        ge = element_energy_gradient(model, el, nodes, stations, n_gauss, fd_h)
        scatter_el_vec(element_dofs(el), ge, g)
    return g


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
) -> NDArray[np.float64]:
    ndof = 7 * model.n_nodes
    K = np.zeros((ndof, ndof), dtype=np.float64)
    for el in model.elements:
        if full_fd:
            Ke = element_hessian_fd(model, el, nodes, stations, n_gauss, fd_h, hess_eps)
        else:
            Ke = element_stiffness_material(
                model, el, nodes, stations, n_gauss, fd_h, jac_eps
            )
        scatter_el_mat(element_dofs(el), Ke, K)
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    ndof = rhs.shape[0]
    free = np.array([i for i in range(ndof) if i not in fixed], dtype=np.int64)
    K_ff = K[np.ix_(free, free)]
    rhs_f = rhs[free]
    return K_ff, rhs_f, free


def expand_solution(free_idx: NDArray[np.int64], du_f: NDArray[np.float64], ndof: int) -> NDArray[np.float64]:
    du = np.zeros(ndof, dtype=np.float64)
    du[free_idx] = du_f
    return du
