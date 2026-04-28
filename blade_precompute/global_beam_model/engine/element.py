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
    skew,
    exp_so3,
    exp_so3_cs,
    relative_rotation_vector_cs,
    update_orientation_cs,
    _quat_to_rotmat_cs,
    _cs_norm,
)
from ..core.types import BeamElement, BeamModel, NodeState, SectionStation


def _shape(xi: float) -> Tuple[float, float, float, float]:
    N1 = 0.5 * (1.0 - xi)
    N2 = 0.5 * (1.0 + xi)
    dN1 = -0.5
    dN2 = 0.5
    return N1, N2, dN1, dN2


def _lagrange_gauss(n: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if n < 1:
        raise ValueError("Gauss order must be >= 1.")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n <= 4:
        if n == 2:
            a = 1.0 / np.sqrt(3.0)
            return np.array([-a, a], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64)
        if n == 3:
            a = np.sqrt(3.0 / 5.0)
            return np.array([-a, 0.0, a], dtype=np.float64), np.array(
                [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64
            )
        a1 = np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        a2 = np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-a2, -a1, a1, a2], dtype=np.float64), np.array(
            [w2, w1, w1, w2], dtype=np.float64
        )
    from numpy.polynomial import legendre as _leg

    x, w = _leg.leggauss(n)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


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


def _gp_strains_from_endpoints(
    x1: NDArray[np.float64],
    q1: NDArray[np.float64],
    x2: NDArray[np.float64],
    q2: NDArray[np.float64],
    L0: float,
    xi: float,
    kappa0: NDArray[np.float64],
    R_for_strain: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Element mechanical 6-strain; if ``R_for_strain`` is None, uses ``_R_interp(q1, q2, xi)``."""
    x_prime = (x2 - x1) / L0
    R = R_for_strain if R_for_strain is not None else _R_interp(q1, q2, float(xi))
    phi = relative_rotation_vector(q1, q2)
    Om = phi / L0
    xp = x_prime / max(float(np.linalg.norm(x_prime)), 1e-14)
    Gam, _ = constitutive.reissner_strains(xp, R, np.zeros((3, 3), dtype=np.float64))
    return constitutive.section_strain_mechanical(Gam, Om, kappa0)


def e7_from_endpoint_states(
    x1: NDArray[np.float64],
    q1: NDArray[np.float64],
    psi1: float,
    x2: NDArray[np.float64],
    q2: NDArray[np.float64],
    psi2: float,
    L0: float,
    kappa0: NDArray[np.float64],
    chi0: float,
    dN1: float,
    dN2: float,
    R_for_strain: NDArray[np.float64] | None,
    xi: float,
) -> NDArray[np.float64]:
    """
    Full 7-strain at one Gauss point from two endpoint DOFs only (no full node list).

    ``R_for_strain`` is the precomputed Gauss rotation when quaternions match the
    base state (translations and warping FD); use None when ``q1``/``q2`` are
    orientation-perturbed so ``_R_interp`` is evaluated on the perturbed quats.
    """
    e6 = _gp_strains_from_endpoints(x1, q1, x2, q2, L0, float(xi), kappa0, R_for_strain)
    dpsi_ds = (2.0 / L0) * (dN1 * float(psi1) + dN2 * float(psi2))
    return constitutive.strain_vector_seven(e6, float(dpsi_ds - chi0))


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
    k0 = _interp_kappa0(model, el, float(xi))
    return _gp_strains_from_endpoints(
        x1, q1, x2, q2, L0, float(xi), k0, None
    )


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
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    L0 = el.L0
    xi_f = float(xi)
    N1, N2, dN1, dN2 = _shape(xi_f)
    if model.kappa0_node is None:
        kappa0 = np.zeros(3, dtype=np.float64)
    else:
        k0i = np.asarray(model.kappa0_node[i], dtype=np.float64).ravel()[:3]
        k0j = np.asarray(model.kappa0_node[j], dtype=np.float64).ravel()[:3]
        kappa0 = (N1 * k0i + N2 * k0j).astype(np.float64, copy=False)
    chi0v = 0.0
    if model.chi0_node is not None:
        chi0v = float(N1 * float(model.chi0_node[i]) + N2 * float(model.chi0_node[j]))
    return e7_from_endpoint_states(
        x1, q1, float(nodes[i].psi),
        x2, q2, float(nodes[j].psi),
        L0, kappa0, chi0v, dN1, dN2, None, xi_f,
    )


def _interp_kappa0_N(
    model: BeamModel, el: BeamElement, N1: float, N2: float
) -> NDArray[np.float64]:
    if model.kappa0_node is None:
        return np.zeros(3, dtype=np.float64)
    i, j = el.node_ids
    k0i = np.asarray(model.kappa0_node[i], dtype=np.float64).ravel()[:3]
    k0j = np.asarray(model.kappa0_node[j], dtype=np.float64).ravel()[:3]
    return (N1 * k0i + N2 * k0j).astype(np.float64, copy=False)


def _interp_chi0_N(model: BeamModel, el: BeamElement, N1: float, N2: float) -> float:
    if model.chi0_node is None:
        return 0.0
    i, j = el.node_ids
    return float(N1 * float(model.chi0_node[i]) + N2 * float(model.chi0_node[j]))


def _tangent_op(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """T(φ): 3×3 left-trivialized tangent map; spatial spin = T(φ) δφ for R = exp([φ]_×)."""
    t = float(np.linalg.norm(phi))
    K = skew(phi)
    if t < 1e-9:
        return np.eye(3, dtype=np.float64) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    c = (1.0 - np.cos(t)) / (t * t)
    s = (t - np.sin(t)) / (t ** 3)
    return np.eye(3, dtype=np.float64) + c * K + s * (K @ K)


def _tangent_op_inv(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """T_inv(φ): inverse of _tangent_op. Satisfies T_inv(φ) T(φ) = I."""
    t = float(np.linalg.norm(phi))
    K = skew(phi)
    if t < 1e-9:
        return np.eye(3, dtype=np.float64) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half_t = 0.5 * t
    cot = np.cos(half_t) / max(np.sin(half_t), 1e-30)
    c = (1.0 - 0.5 * t * cot) / (t * t)
    return np.eye(3, dtype=np.float64) - 0.5 * K + c * (K @ K)


def _analytical_B_gp(
    x1: NDArray[np.float64],
    q1: NDArray[np.float64],
    x2: NDArray[np.float64],
    q2: NDArray[np.float64],
    L0: float,
    dN1: float,
    dN2: float,
    xi_f: float,
) -> NDArray[np.float64]:
    """
    Analytical (7, 14) strain-displacement B-matrix at one Gauss point.

    DOF layout per column block: [x1(0:3) | θ1(3:6) | ψ1(6) | x2(7:10) | θ2(10:13) | ψ2(13)]
    Row layout after P_SECTION: [Γ₁ | Ω₂ | Ω₃ | Ω₁ | Γ₂ | Γ₃ | χ]

    Derivation: left-invariant spatial spin parameterisation.
      φ = relative_rotation_vector(q1, q2) = log(R1^T R2)
      ∂φ/∂θ1 = -T_inv(φ) R1^T,   ∂φ/∂θ2 = +T_inv(φ) R1^T
      R_gp = R1 exp(t_s [φ]_×),  t_s = (ξ+1)/2
      δR_gp (node-1 spin) = [e_k − t_s R1 C e_k]_× R_gp,  C = T(t_s φ) T_inv(φ) R1^T
      δΓ = R_gp^T [ξ_t]_× δθ_gp  (using R_gp^T [ξ_t]_× = [R_gp^T ξ_t]_× R_gp^T)
    """
    t_s = 0.5 * (xi_f + 1.0)

    d = x2 - x1
    L = float(np.linalg.norm(d))
    L_inv = 1.0 / max(L, 1e-14)
    xi_t = d * L_inv
    P_proj = np.eye(3, dtype=np.float64) - np.outer(xi_t, xi_t)

    phi = relative_rotation_vector(q1, q2)
    R1 = quat_to_rotmat(q1)
    R_s = exp_so3(t_s * phi)
    R_gp = R1 @ R_s

    T_inv = _tangent_op_inv(phi)
    T_ts = _tangent_op(t_s * phi)
    C = T_ts @ T_inv @ R1.T  # (3,3): maps spatial spin → effective local increment

    xi_t_local = R_gp.T @ xi_t  # R_gp^T ξ_t

    dGam_dx = R_gp.T @ P_proj * L_inv  # ∂Γ/∂x (sign applied per node below)
    ts_skew_xtl_RsT_C = t_s * skew(xi_t_local) @ R_s.T @ C

    # ∂Γ/∂θ1 = R_gp^T [ξ_t]_× − t_s [R_gp^T ξ_t]_× R_s^T C
    dGam_dth1 = R_gp.T @ skew(xi_t) - ts_skew_xtl_RsT_C
    dGam_dth2 = ts_skew_xtl_RsT_C  # ∂Γ/∂θ2

    dOm_dth = (1.0 / L0) * (T_inv @ R1.T)  # ±∂Ω/∂θ (sign per node)

    # Pre-permutation (6×14): rows [Γ₁,Γ₂,Γ₃ | Ω₁,Ω₂,Ω₃]
    J6 = np.zeros((6, 14), dtype=np.float64)
    J6[0:3, 0:3] = -dGam_dx
    J6[0:3, 3:6] = dGam_dth1
    J6[0:3, 7:10] = dGam_dx
    J6[0:3, 10:13] = dGam_dth2
    J6[3:6, 3:6] = -dOm_dth
    J6[3:6, 10:13] = dOm_dth

    B = np.zeros((7, 14), dtype=np.float64)
    B[0:6, :] = constitutive.P_SECTION @ J6
    B[6, 6] = (2.0 / L0) * dN1   # ∂χ/∂ψ1
    B[6, 13] = (2.0 / L0) * dN2  # ∂χ/∂ψ2
    return B


def _apply_bump_cs(
    k: int,
    h: float,
    x1: NDArray[np.float64],
    q1: NDArray[np.float64],
    p1: float,
    x2: NDArray[np.float64],
    q2: NDArray[np.float64],
    p2: float,
) -> tuple:
    """Complex-step bump of magnitude h on DOF k; returns complex endpoint states."""
    ih = 1j * h
    x1c = x1.astype(complex)
    q1c = q1.astype(complex)
    p1c = complex(p1)
    x2c = x2.astype(complex)
    q2c = q2.astype(complex)
    p2c = complex(p2)
    if k < 3:
        x1c = x1c.copy()
        x1c[k] += ih
    elif k < 6:
        dth = np.zeros(3, dtype=complex)
        dth[k - 3] = ih
        q1c = update_orientation_cs(q1, dth)
    elif k == 6:
        p1c += ih
    elif k < 10:
        x2c = x2c.copy()
        x2c[k - 7] += ih
    elif k < 13:
        dth = np.zeros(3, dtype=complex)
        dth[k - 10] = ih
        q2c = update_orientation_cs(q2, dth)
    else:
        p2c += ih
    return x1c, q1c, p1c, x2c, q2c, p2c


def _cs_e7(
    x1c: np.ndarray,
    q1c: np.ndarray,
    p1c: complex,
    x2c: np.ndarray,
    q2c: np.ndarray,
    p2c: complex,
    L0: float,
    kappa0: NDArray[np.float64],
    chi0: float,
    dN1: float,
    dN2: float,
    xi_f: float,
) -> np.ndarray:
    """7-strain vector in complex arithmetic for complex-step gradient."""
    t_s = 0.5 * (xi_f + 1.0)
    d = x2c - x1c
    L = _cs_norm(d)
    xi_t = d / L
    phi_c = relative_rotation_vector_cs(q1c, q2c)
    R1_c = _quat_to_rotmat_cs(q1c)
    R_gp_c = R1_c @ exp_so3_cs(t_s * phi_c)
    e1 = np.array([1.0 + 0j, 0j, 0j])
    Gamma = R_gp_c.T @ xi_t - e1
    Omega_mech = phi_c / L0 - np.asarray(kappa0, dtype=complex)
    v6 = np.concatenate([Gamma, Omega_mech])
    e6 = constitutive.P_SECTION.astype(complex) @ v6
    chi = (2.0 / L0) * (dN1 * p1c + dN2 * p2c) - chi0
    return np.concatenate([e6, [chi]])


def _reuse_R_gp_for_fd_col(col: int) -> bool:
    """Columns where Gauss frame rotation is unchanged from the base (reuse ``R_gp``)."""
    return col in (0, 1, 2, 6, 7, 8, 9, 13)


def _apply_endpoint_bump(
    col: int,
    sign: float,
    eps: float,
    x1: NDArray[np.float64],
    q1: NDArray[np.float64],
    psi1: float,
    x2: NDArray[np.float64],
    q2: NDArray[np.float64],
    psi2: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], float]:
    s = sign * float(eps)
    dx1 = np.zeros(3, dtype=np.float64)
    th1 = np.zeros(3, dtype=np.float64)
    dpsi1 = 0.0
    dx2 = np.zeros(3, dtype=np.float64)
    th2 = np.zeros(3, dtype=np.float64)
    dpsi2 = 0.0
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
    return (
        x1 + dx1,
        update_orientation(q1, th1),
        float(psi1 + dpsi1),
        x2 + dx2,
        update_orientation(q2, th2),
        float(psi2 + dpsi2),
    )


def element_energy_gradient(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    K7_gp_el: List[NDArray[np.float64]] | None = None,
    use_cs: bool = True,
    h_cs: float = 1e-20,
) -> NDArray[np.float64]:
    if use_cs:
        return element_energy_gradient_cs(model, el, nodes, stations, n_gauss, h_cs, K7_gp_el)
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    p1, p2 = float(nodes[i].psi), float(nodes[j].psi)
    L0 = el.L0
    ztab = infer_z_node(model)
    xi_w, w_w = _lagrange_gauss(n_gauss)
    g = np.zeros(14, dtype=np.float64)
    jac_eps = float(fd_h) if float(fd_h) > 0.0 else 1e-7
    for gi, (xi, w) in enumerate(zip(xi_w, w_w)):
        xi_f = float(xi)
        N1, N2, dN1, dN2 = _shape(xi_f)
        kappa0 = _interp_kappa0_N(model, el, N1, N2)
        chi0v = _interp_chi0_N(model, el, N1, N2)
        zg = float(N1 * ztab[i] + N2 * ztab[j])
        if K7_gp_el is not None:
            K7 = K7_gp_el[gi]
        else:
            K7 = interp_K7(np.array([zg], dtype=np.float64), stations)[0]
        e0 = e7_from_endpoint_states(
            x1, q1, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
        )
        r_nat = constitutive.section_resultants_natural(K7, e0)
        R_gp = _R_interp(q1, q2, xi_f)
        fac = 0.5 * L0 * float(w)
        for k in range(3):
            xp, xm = x1.copy(), x1.copy()
            xp[k] += jac_eps
            xm[k] -= jac_eps
            ep = e7_from_endpoint_states(
                xp, q1, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
            )
            em = e7_from_endpoint_states(
                xm, q1, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
            )
            g[k] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
        for k in range(3):
            ek = np.zeros(3, dtype=np.float64)
            ek[k] = 1.0
            qp = update_orientation(q1, jac_eps * ek)
            qm = update_orientation(q1, -jac_eps * ek)
            ep = e7_from_endpoint_states(
                x1, qp, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
            )
            em = e7_from_endpoint_states(
                x1, qm, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
            )
            g[3 + k] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
        for k in range(3):
            xp, xm = x2.copy(), x2.copy()
            xp[k] += jac_eps
            xm[k] -= jac_eps
            ep = e7_from_endpoint_states(
                x1, q1, p1, xp, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
            )
            em = e7_from_endpoint_states(
                x1, q1, p1, xm, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
            )
            g[7 + k] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
        for k in range(3):
            ek = np.zeros(3, dtype=np.float64)
            ek[k] = 1.0
            qp = update_orientation(q2, jac_eps * ek)
            qm = update_orientation(q2, -jac_eps * ek)
            ep = e7_from_endpoint_states(
                x1, q1, p1, x2, qp, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
            )
            em = e7_from_endpoint_states(
                x1, q1, p1, x2, qm, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
            )
            g[10 + k] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
        ep = e7_from_endpoint_states(
            x1, q1, p1 + jac_eps, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
        )
        em = e7_from_endpoint_states(
            x1, q1, p1 - jac_eps, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
        )
        g[6] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
        ep = e7_from_endpoint_states(
            x1, q1, p1, x2, q2, p2 + jac_eps, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
        )
        em = e7_from_endpoint_states(
            x1, q1, p1, x2, q2, p2 - jac_eps, L0, kappa0, chi0v, dN1, dN2, R_gp, xi_f
        )
        g[13] += fac * float(np.dot((ep - em) / (2.0 * jac_eps), r_nat))
    return g


def element_energy_gradient_cs(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    h_cs: float = 1e-20,
    K7_gp_el: List[NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """Machine-precision element gradient via complex-step differentiation."""
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    p1, p2 = float(nodes[i].psi), float(nodes[j].psi)
    L0 = el.L0
    ztab = infer_z_node(model)
    xi_w, w_w = _lagrange_gauss(n_gauss)
    g = np.zeros(14, dtype=np.float64)
    for gi, (xi, w) in enumerate(zip(xi_w, w_w)):
        xi_f = float(xi)
        N1, N2, dN1, dN2 = _shape(xi_f)
        kappa0 = _interp_kappa0_N(model, el, N1, N2)
        chi0v = _interp_chi0_N(model, el, N1, N2)
        zg = float(N1 * ztab[i] + N2 * ztab[j])
        K7 = K7_gp_el[gi] if K7_gp_el is not None else interp_K7(
            np.array([zg], dtype=np.float64), stations
        )[0]
        fac = 0.5 * L0 * float(w)
        for k in range(14):
            x1c, q1c, p1c, x2c, q2c, p2c = _apply_bump_cs(
                k, h_cs, x1, q1, p1, x2, q2, p2
            )
            e7c = _cs_e7(x1c, q1c, p1c, x2c, q2c, p2c,
                         L0, kappa0, chi0v, dN1, dN2, xi_f)
            Ue = 0.5 * np.sum(e7c * (K7 @ e7c))
            g[k] += fac * float(np.imag(Ue)) / h_cs
    return g


def _perturb_nodes(
    nodes: List[NodeState],
    el: BeamElement,
    col: int,
    sign: float,
    eps: float,
) -> List[NodeState]:
    """Perturb one element endpoint; only the affected node is copied (not the full list)."""
    i, j = el.node_ids
    n = len(nodes)
    dx1 = np.zeros(3, dtype=np.float64)
    th1 = np.zeros(3, dtype=np.float64)
    dpsi1 = 0.0
    dx2 = np.zeros(3, dtype=np.float64)
    th2 = np.zeros(3, dtype=np.float64)
    dpsi2 = 0.0
    s = sign * float(eps)
    if col < 3:
        dx1[int(col)] = s
    elif col < 6:
        th1[int(col) - 3] = s
    elif col == 6:
        dpsi1 = s
    elif col < 10:
        dx2[int(col) - 7] = s
    elif col < 13:
        th2[int(col) - 10] = s
    else:
        dpsi2 = s
    out: List[NodeState] = [nodes[k] for k in range(n)]
    if col < 7:
        out[i] = nodes[i].copy()
        out[i].x = out[i].x + dx1
        out[i].q = update_orientation(out[i].q, th1)
        out[i].psi = float(out[i].psi) + dpsi1
    else:
        out[j] = nodes[j].copy()
        out[j].x = out[j].x + dx2
        out[j].q = update_orientation(out[j].q, th2)
        out[j].psi = float(out[j].psi) + dpsi2
    return out


def element_stiffness_material(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    jac_eps: float,
    K7_gp_el: List[NDArray[np.float64]] | None = None,
    _use_fd_B: bool = False,
) -> NDArray[np.float64]:
    i, j = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[j].x, nodes[j].q
    p1, p2 = float(nodes[i].psi), float(nodes[j].psi)
    L0 = el.L0
    ztab = infer_z_node(model)
    xi_w, w_w = _lagrange_gauss(n_gauss)
    K = np.zeros((14, 14), dtype=np.float64)
    for gi, (xi, w) in enumerate(zip(xi_w, w_w)):
        xi_f = float(xi)
        N1, N2, dN1, dN2 = _shape(xi_f)
        kappa0 = _interp_kappa0_N(model, el, N1, N2)
        chi0v = _interp_chi0_N(model, el, N1, N2)
        zg = float(N1 * ztab[i] + N2 * ztab[j])
        K7 = K7_gp_el[gi] if K7_gp_el is not None else interp_K7(
            np.array([zg], dtype=np.float64), stations
        )[0]
        fac = 0.5 * L0 * float(w)
        if _use_fd_B:
            R_gp = _R_interp(q1, q2, xi_f)
            B = np.zeros((7, 14), dtype=np.float64)
            for col in range(14):
                r_use = R_gp if _reuse_R_gp_for_fd_col(col) else None
                x1p, q1p, p1p, x2p, q2p, p2p = _apply_endpoint_bump(
                    col, 1.0, jac_eps, x1, q1, p1, x2, q2, p2
                )
                x1m, q1m, p1m, x2m, q2m, p2m = _apply_endpoint_bump(
                    col, -1.0, jac_eps, x1, q1, p1, x2, q2, p2
                )
                ep = e7_from_endpoint_states(
                    x1p, q1p, p1p, x2p, q2p, p2p, L0, kappa0, chi0v, dN1, dN2, r_use, xi_f
                )
                em = e7_from_endpoint_states(
                    x1m, q1m, p1m, x2m, q2m, p2m, L0, kappa0, chi0v, dN1, dN2, r_use, xi_f
                )
                B[:, col] = (ep - em) / (2.0 * jac_eps)
        else:
            B = _analytical_B_gp(x1, q1, x2, q2, L0, dN1, dN2, xi_f)
        K += fac * (B.T @ K7 @ B)
    K = 0.5 * (K + K.T)
    return K


def element_stiffness_geometric_stress(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    jac_eps: float,
    K7_gp_el: List[NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    """
    Stress (initial) geometric stiffness: ``Σ_m r_m ∂²e_m/∂q∂qᵀ`` with constant ``K7`` and
    ``r_nat = K7 @ e7`` (native resultant order).  Summed with ``Bᵀ K7 B`` this matches the
    consistent Hessian of ``½ eᵀ K7 e`` for nonlinear ``e7(q)``.
    """
    i, jn = el.node_ids
    x1, q1 = nodes[i].x, nodes[i].q
    x2, q2 = nodes[jn].x, nodes[jn].q
    p1, p2 = float(nodes[i].psi), float(nodes[jn].psi)
    L0 = el.L0
    ztab = infer_z_node(model)
    xi_w, w_w = _lagrange_gauss(n_gauss)
    h = max(float(jac_eps), 1e-10)
    Kg = np.zeros((14, 14), dtype=np.float64)
    for gi, (xi, w) in enumerate(zip(xi_w, w_w)):
        xi_f = float(xi)
        N1, N2, dN1, dN2 = _shape(xi_f)
        kappa0 = _interp_kappa0_N(model, el, N1, N2)
        chi0v = _interp_chi0_N(model, el, N1, N2)
        zg = float(N1 * ztab[i] + N2 * ztab[jn])
        K7 = K7_gp_el[gi] if K7_gp_el is not None else interp_K7(
            np.array([zg], dtype=np.float64), stations
        )[0]
        e0 = e7_from_endpoint_states(
            x1, q1, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f
        )
        r_nat = constitutive.section_resultants_natural(K7, e0)
        fac = 0.5 * L0 * float(w)
        for ii in range(14):
            x1p, q1p, p1p, x2p, q2p, p2p = _apply_endpoint_bump(
                ii, 1.0, h, x1, q1, p1, x2, q2, p2
            )
            x1m, q1m, p1m, x2m, q2m, p2m = _apply_endpoint_bump(
                ii, -1.0, h, x1, q1, p1, x2, q2, p2
            )
            Bp = _analytical_B_gp(x1p, q1p, x2p, q2p, L0, dN1, dN2, xi_f)
            Bm = _analytical_B_gp(x1m, q1m, x2m, q2m, L0, dN1, dN2, xi_f)
            dB_dqi = (Bp - Bm) / (2.0 * h)
            Kg[ii, :] += fac * (r_nat @ dB_dqi)
    Kg = 0.5 * (Kg + Kg.T)
    return Kg


def element_hessian_fd(
    model: BeamModel,
    el: BeamElement,
    nodes: List[NodeState],
    stations: List[SectionStation],
    n_gauss: int,
    fd_h: float,
    hess_eps: float,
    K7_gp_el: List[NDArray[np.float64]] | None = None,
) -> NDArray[np.float64]:
    K = np.zeros((14, 14), dtype=np.float64)
    for col in range(14):
        g_p = element_energy_gradient(
            model, el, _perturb_nodes(nodes, el, col, +1.0, hess_eps), stations, n_gauss, fd_h, K7_gp_el
        )
        g_m = element_energy_gradient(
            model, el, _perturb_nodes(nodes, el, col, -1.0, hess_eps), stations, n_gauss, fd_h, K7_gp_el
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
