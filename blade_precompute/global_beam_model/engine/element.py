"""
beam_model/element.py
=====================
Two-node Simo–Reissner beam element with Vlasov warping (7 DOFs / node).

Internal energy (Gauss)::

    U_e ≈ (L₀/2) Σ_g w_g [ ½ e₇ᵀ K7(z_g) e₇ ],   e₇ = [e_sec; χ]

``χ = dψ/ds − χ₀`` with linear interpolation of ``ψ`` along the element.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from . import constitutive
from .interp import interp_K7
from .kinematics import (
    quat_to_rotmat,
    relative_rotation_vector,
    slerp,
    update_orientation,
)
from ..core.types import BeamElement, BeamModel, NodeState, SectionStation


def _clone_states(nodes: List[NodeState]) -> List[NodeState]:
    return [n.copy() for n in nodes]


def _shape(xi: float) -> Tuple[float, float, float, float]:
    N1 = 0.5 * (1.0 - xi)
    N2 = 0.5 * (1.0 + xi)
    dN1 = -0.5
    dN2 = 0.5
    return N1, N2, dN1, dN2


def _lagrange_gauss(n: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)
    if n == 3:
        a = np.sqrt(3.0 / 5.0)
        return np.array([-a, 0.0, a], dtype=np.float64), np.array(
            [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64
        )
    if n == 4:
        a1 = np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        a2 = np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-a2, -a1, a1, a2], dtype=np.float64), np.array(
            [w2, w1, w1, w2], dtype=np.float64
        )
    raise NotImplementedError("Only 1- to 4-point Gauss rules are implemented.")


def element_gauss_shape_matrix(
    n_gauss: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Gauss-point natural coordinates, quadrature weights, and axial Lagrange
    shape rows N(ξ) for a 2-node line element (same ``_shape`` as assembly).

    Returns
    -------
    xi, w, N_mat
        ``xi``, ``w`` length ``n_gauss``; ``N_mat`` shape ``(n_gauss, 2)`` with
        ``[N1, N2]`` per row for use in GP→nodal recovery (see nodal_result_projector).
    """
    xi_w, w_w = _lagrange_gauss(n_gauss)
    N_mat = np.zeros((n_gauss, 2), dtype=np.float64)
    for g, xi in enumerate(xi_w):
        N1, N2, _, _ = _shape(float(xi))
        N_mat[g, 0] = N1
        N_mat[g, 1] = N2
    return xi_w, w_w, N_mat


def infer_z_node(model: BeamModel) -> NDArray[np.float64]:
    """Cumulative arc-length at nodes (chain along element list)."""
    if model.z_node is not None:
        return np.asarray(model.z_node, dtype=np.float64).ravel()
    n = model.n_nodes
    z = np.zeros(n, dtype=np.float64)
    for el in model.elements:
        i, j = el.node_ids
        z[j] = z[i] + el.L0
    return z


def gp_z_coord(model: BeamModel, el: BeamElement, xi: float) -> float:
    ztab = infer_z_node(model)
    i, j = el.node_ids
    N1, N2, _, _ = _shape(xi)
    return float(N1 * ztab[i] + N2 * ztab[j])


def _interp_kappa0(
    model: BeamModel,
    el: BeamElement,
    xi: float,
) -> NDArray[np.float64]:
    if model.kappa0_node is None:
        return np.zeros(3, dtype=np.float64)
    i, j = el.node_ids
    N1, N2, _, _ = _shape(xi)
    k0 = N1 * model.kappa0_node[i] + N2 * model.kappa0_node[j]
    return np.asarray(k0, dtype=np.float64).reshape(3)


def _interp_chi0(model: BeamModel, el: BeamElement, xi: float) -> float:
    if model.chi0_node is None:
        return 0.0
    i, j = el.node_ids
    N1, N2, _, _ = _shape(xi)
    return float(N1 * model.chi0_node[i] + N2 * model.chi0_node[j])


def _R_interp(q1: NDArray[np.float64], q2: NDArray[np.float64], xi: float) -> NDArray[np.float64]:
    t = 0.5 * (xi + 1.0)
    qg = slerp(q1, q2, float(t))
    return quat_to_rotmat(qg)


def gp_strains_mechanical(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    xi: float,
    fd_h: float,
) -> NDArray[np.float64]:
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    L0 = el.L0
    x_prime = (x2 - x1) / L0
    R = _R_interp(q1, q2, xi)
    phi = relative_rotation_vector(q1, q2)
    Om = phi / L0
    k0 = _interp_kappa0(model, el, float(xi))
    xp = x_prime / max(float(np.linalg.norm(x_prime)), 1e-14)
    Gam, _ = constitutive.reissner_strains(xp, R, np.zeros((3, 3), dtype=np.float64))
    return constitutive.section_strain_mechanical(Gam, Om, k0)


def chi_strain(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    xi: float,
) -> float:
    i, j = el.node_ids
    L0 = el.L0
    N1, N2, dN1, dN2 = _shape(float(xi))
    dpsi_ds = (2.0 / L0) * (dN1 * nodes[i].psi + dN2 * nodes[j].psi)
    chi0 = _interp_chi0(model, el, float(xi))
    return float(dpsi_ds - chi0)


def e7_vector(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    xi: float,
    fd_h: float,
) -> NDArray[np.float64]:
    e6 = gp_strains_mechanical(model, el, nodes, float(xi), fd_h)
    chi = chi_strain(model, el, nodes, float(xi))
    return constitutive.strain_vector_seven(e6, chi)


def element_energy_gradient(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> NDArray[np.float64]:
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    L0 = el.L0
    xi_w, w_w = _lagrange_gauss(n_gauss)
    g = np.zeros(14, dtype=np.float64)
    jac_eps = 1e-7
    for xi, w in zip(xi_w, w_w):
        zg = gp_z_coord(model, el, float(xi))
        K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
        e0 = e7_vector(model, el, nodes, float(xi), fd_h)
        r_nat = constitutive.section_resultants_natural(K7, e0)
        fac = 0.5 * L0 * float(w)
        for k in range(3):
            xp = x1.copy()
            xm = x1.copy()
            xp[k] += jac_eps
            xm[k] -= jac_eps
            ns = _clone_states(nodes)
            ns[i].x = xp
            ep = e7_vector(model, el, ns, float(xi), fd_h)
            ns[i].x = xm
            em = e7_vector(model, el, ns, float(xi), fd_h)
            de = (ep - em) / (2.0 * jac_eps)
            g[k] += fac * float(np.dot(de, r_nat))
        for k in range(3):
            ek = np.zeros(3)
            ek[k] = 1.0
            qp = update_orientation(q1, jac_eps * ek)
            qm = update_orientation(q1, -jac_eps * ek)
            ns = _clone_states(nodes)
            ns[i].q = qp
            ep = e7_vector(model, el, ns, float(xi), fd_h)
            ns[i].q = qm
            em = e7_vector(model, el, ns, float(xi), fd_h)
            de = (ep - em) / (2.0 * jac_eps)
            g[3 + k] += fac * float(np.dot(de, r_nat))
        for k in range(3):
            xp = x2.copy()
            xm = x2.copy()
            xp[k] += jac_eps
            xm[k] -= jac_eps
            ns = _clone_states(nodes)
            ns[j].x = xp
            ep = e7_vector(model, el, ns, float(xi), fd_h)
            ns[j].x = xm
            em = e7_vector(model, el, ns, float(xi), fd_h)
            de = (ep - em) / (2.0 * jac_eps)
            g[7 + k] += fac * float(np.dot(de, r_nat))
        for k in range(3):
            ek = np.zeros(3)
            ek[k] = 1.0
            qp = update_orientation(q2, jac_eps * ek)
            qm = update_orientation(q2, -jac_eps * ek)
            ns = _clone_states(nodes)
            ns[j].q = qp
            ep = e7_vector(model, el, ns, float(xi), fd_h)
            ns[j].q = qm
            em = e7_vector(model, el, ns, float(xi), fd_h)
            de = (ep - em) / (2.0 * jac_eps)
            g[10 + k] += fac * float(np.dot(de, r_nat))
        for psi_idx, node_idx in ((6, i), (13, j)):
            ns = _clone_states(nodes)
            ns[node_idx].psi += jac_eps
            ep = e7_vector(model, el, ns, float(xi), fd_h)
            ns[node_idx].psi -= 2.0 * jac_eps
            em = e7_vector(model, el, ns, float(xi), fd_h)
            de = (ep - em) / (2.0 * jac_eps)
            g[psi_idx] += fac * float(np.dot(de, r_nat))
    return g


def _perturb_nodes(
    nodes: List[NodeState],
    el: BeamElement,
    col: int,
    sign: float,
    eps: float,
) -> List[NodeState]:
    i, j = el.node_ids
    dx1 = np.zeros(3, dtype=np.float64)
    th1 = np.zeros(3, dtype=np.float64)
    dpsi1 = 0.0
    dx2 = np.zeros(3, dtype=np.float64)
    th2 = np.zeros(3, dtype=np.float64)
    dpsi2 = 0.0
    s = sign * eps
    if col < 3:
        dx1[col] = s
    elif col < 6:
        th1[col - 3] = s
    elif col == 6:
        dpsi1 = s
    elif col < 10:
        dx2[col - 7] = s
    elif col < 13:
        th2[col - 10] = s
    else:
        dpsi2 = s
    ns = _clone_states(nodes)
    ns[i].x = ns[i].x + dx1
    ns[i].q = update_orientation(ns[i].q, th1)
    ns[i].psi += dpsi1
    ns[j].x = ns[j].x + dx2
    ns[j].q = update_orientation(ns[j].q, th2)
    ns[j].psi += dpsi2
    return ns


def element_stiffness_material(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    jac_eps: float,
) -> NDArray[np.float64]:
    L0 = el.L0
    xi_w, w_w = _lagrange_gauss(n_gauss)
    K = np.zeros((14, 14), dtype=np.float64)
    for xi, w in zip(xi_w, w_w):
        zg = gp_z_coord(model, el, float(xi))
        K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
        B = np.zeros((7, 14), dtype=np.float64)
        for col in range(14):
            n_plus = _perturb_nodes(nodes, el, col, +1.0, jac_eps)
            n_minus = _perturb_nodes(nodes, el, col, -1.0, jac_eps)
            ep = e7_vector(model, el, n_plus, float(xi), fd_h)
            em = e7_vector(model, el, n_minus, float(xi), fd_h)
            B[:, col] = (ep - em) / (2.0 * jac_eps)
        fac = 0.5 * L0 * float(w)
        K += fac * (B.T @ K7 @ B)
    K = 0.5 * (K + K.T)
    return K


def element_hessian_fd(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    hess_eps: float,
) -> NDArray[np.float64]:
    K = np.zeros((14, 14), dtype=np.float64)
    for col in range(14):
        g_p = element_energy_gradient(
            model, el, _perturb_nodes(nodes, el, col, +1.0, hess_eps), stations, n_gauss, fd_h
        )
        g_m = element_energy_gradient(
            model, el, _perturb_nodes(nodes, el, col, -1.0, hess_eps), stations, n_gauss, fd_h
        )
        K[:, col] = (g_p - g_m) / (2.0 * hess_eps)
    return 0.5 * (K + K.T)


def element_gp_cache(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    xi_w, _, _ = element_gauss_shape_matrix(n_gauss)
    zs = np.zeros(n_gauss, dtype=np.float64)
    es = np.zeros((n_gauss, 7), dtype=np.float64)
    Rs = np.zeros((n_gauss, 7), dtype=np.float64)
    for g, xi in enumerate(xi_w):
        zg = gp_z_coord(model, el, float(xi))
        zs[g] = zg
        K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
        e = e7_vector(model, el, nodes, float(xi), fd_h)
        es[g] = e
        Rs[g] = constitutive.section_resultants_seven(K7, e)
    return zs, es, Rs
