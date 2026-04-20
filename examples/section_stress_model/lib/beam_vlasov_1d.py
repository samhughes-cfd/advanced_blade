"""
1D spanwise equilibrium on x in [0, L] (root x=0, tip x=L).

- **Cantilever** statics: axial force, shears, bending moments, torque from distributed
  line loads by integration toward the tip (tip free).
- **Warping / non-uniform torsion**: 4th-order ODE on the twist angle ``phi(x)`` about x:

      (EI_omega(x) phi'')'' - (GJ(x) phi')' = m_x(x)

  For **uniform** ``EI_omega``, ``GJ`` this reduces to the common form

      EI_omega phi'''' - GJ phi'' = m_x(x).

  Discretized with second-order finite differences on a uniform grid; **bimoment**
  (engineering convention used here):

      B(x) ≈ -EI_omega(x) * phi''(x).

Distributed loads are in the **blade B frame** (edge / flap / axial / torque about x).
Output arrays are in **B** components for ``V_edge, V_flap, M_edge, M_flap`` — pass through
``blade_frames.resultants_B_to_S`` for each station to obtain ``Vy, Vz, My, Mz``.
"""

from __future__ import annotations

import numpy as np


def _trapz_tail(x: np.ndarray, f: np.ndarray, x_from: float) -> float:
    """Integral of f(x) from x_from to x[-1] (trapezoid)."""
    mask = x >= x_from - 1e-15
    if np.count_nonzero(mask) < 2:
        return 0.0
    xs = x[mask]
    fs = f[mask]
    return float(np.trapezoid(fs, xs))


def cantilever_resultants_B(
    x: np.ndarray,
    q_axial: np.ndarray,
    p_edge: np.ndarray,
    p_flap: np.ndarray,
    m_torque_x: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Cantilever fixed at x=0, free at x=L. Loads per unit length in B frame.

    Returns N, V_edge, V_flap, M_edge, M_flap, T (all [N] or [N·m] per unit conventions).
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    q_axial = np.asarray(q_axial, dtype=float).ravel()
    p_edge = np.asarray(p_edge, dtype=float).ravel()
    p_flap = np.asarray(p_flap, dtype=float).ravel()
    m_torque_x = np.asarray(m_torque_x, dtype=float).ravel()

    N = np.array([_trapz_tail(x, q_axial, x[i]) for i in range(n)])
    V_edge = np.array([_trapz_tail(x, p_edge, x[i]) for i in range(n)])
    V_flap = np.array([_trapz_tail(x, p_flap, x[i]) for i in range(n)])
    T = np.array([_trapz_tail(x, m_torque_x, x[i]) for i in range(n)])

    M_flap = np.array([_trapz_tail(x, V_flap, x[i]) for i in range(n)])
    M_edge = np.array([_trapz_tail(x, V_edge, x[i]) for i in range(n)])

    return {
        "N": N,
        "V_edge": V_edge,
        "V_flap": V_flap,
        "M_edge": M_edge,
        "M_flap": M_flap,
        "T": T,
    }


def solve_nonuniform_torsion_fd(
    x: np.ndarray,
    EI_omega: np.ndarray | float,
    GJ: np.ndarray | float,
    m_x: np.ndarray,
    *,
    phi0: float = 0.0,
    dphi0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve ``EI_omega phi'''' - GJ phi'' = m_x(x)`` with **clamped root** ``phi(0)=phi0``,
    ``phi'(0)=dphi0`` and **free tip**: ``phi''(L)=0``, ``phi'''(L)=0`` (FD).

    Uniform spacing; ``EI_omega``, ``GJ`` per node.

    Returns ``phi``, ``phi_double_prime``, ``B`` (bimoment ≈ -EI_omega * phi'').
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 7:
        raise ValueError("need at least 7 nodes for FD torsion solve")
    h = float(x[1] - x[0])
    if not np.allclose(np.diff(x), h, rtol=1e-6):
        raise ValueError("uniform x grid required for this FD solver")

    m_x = np.asarray(m_x, dtype=float).ravel()
    EIw = np.broadcast_to(np.asarray(EI_omega, dtype=float), (n,))
    gj = np.broadcast_to(np.asarray(GJ, dtype=float), (n,))

    inv_h2 = 1.0 / (h * h)
    inv_h4 = 1.0 / (h**4)
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    phi0 = float(phi0)
    phi1 = phi0 + h * float(dphi0)

    # Interior i = 2 .. n-3
    A[:, :] = 0.0
    rhs[:] = 0.0
    for i in range(2, n - 2):
        ei, gi = float(EIw[i]), float(gj[i])
        A[i, i - 2] = ei * inv_h4
        A[i, i - 1] = -4.0 * ei * inv_h4 - gi * inv_h2
        A[i, i] = 6.0 * ei * inv_h4 + 2.0 * gi * inv_h2
        A[i, i + 1] = -4.0 * ei * inv_h4 - gi * inv_h2
        A[i, i + 2] = ei * inv_h4
        rhs[i] = float(m_x[i])
    for i in range(2, n - 2):
        for j in (0, 1):
            if A[i, j] != 0.0:
                phi_b = phi0 if j == 0 else phi1
                rhs[i] -= A[i, j] * phi_b
                A[i, j] = 0.0

    # Row 0: phi[0] = phi0
    A[0, 0] = 1.0
    rhs[0] = phi0
    # Row 1: phi[1] = phi1 (from phi' at root)
    A[1, 1] = 1.0
    rhs[1] = phi1
    # Free tip
    A[n - 2, n - 3] = 1.0 * inv_h2
    A[n - 2, n - 2] = -2.0 * inv_h2
    A[n - 2, n - 1] = 1.0 * inv_h2
    rhs[n - 2] = 0.0
    A[n - 1, n - 4] = -1.0 / (h**3)
    A[n - 1, n - 3] = 3.0 / (h**3)
    A[n - 1, n - 2] = -3.0 / (h**3)
    A[n - 1, n - 1] = 1.0 / (h**3)
    rhs[n - 1] = 0.0

    phi = np.linalg.solve(A, rhs)
    p2 = np.zeros(n)
    for i in range(1, n - 1):
        p2[i] = (phi[i + 1] - 2 * phi[i] + phi[i - 1]) * inv_h2
    p2[0] = (phi[2] - 2 * phi[1] + phi[0]) * inv_h2
    p2[-1] = (phi[-1] - 2 * phi[-2] + phi[-3]) * inv_h2
    B = -EIw * p2
    return phi, p2, B


def solve_span_equilibrium(
    x: np.ndarray,
    q_axial: np.ndarray,
    p_edge: np.ndarray,
    p_flap: np.ndarray,
    m_torque_x: np.ndarray,
    EI_omega: np.ndarray | float,
    GJ: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """
    Combined cantilever resultants in B plus warping ``B_warp`` from non-uniform torsion ODE.

    The same ``m_torque_x`` drives both **Saint-Venant-type** resultant ``T(x)`` (from
    equilibrium) and the **warping** ODE; in a full theory these are split — here ``T`` is
    retained from statics and ``B`` from the warping equation for section post-processing.
    """
    res = cantilever_resultants_B(x, q_axial, p_edge, p_flap, m_torque_x)
    phi, p2, Bw = solve_nonuniform_torsion_fd(x, EI_omega, GJ, m_torque_x)
    res["phi_twist"] = phi
    res["B"] = Bw
    return res
