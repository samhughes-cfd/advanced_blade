"""
MITC4 flat-shell element for panel-level stress recovery.

Reference: Bathe & Dvorkin (1985), "A four-node plate bending element based on
Mindlin/Reissner plate theory and a mixed interpolation of tensorial components".

Local coordinate frame per panel strip
---------------------------------------
  ξ ∈ [−1, 1]  — along panel contour tangent (s direction, length L_s)
  η ∈ [−1, 1]  — along beam-axis / span (x direction, length L_x = 1 for unit slice)
  ζ            — through-thickness normal (not discretised)

Node ordering (counter-clockwise):
  1: (ξ=−1, η=−1)   2: (ξ=+1, η=−1)
  3: (ξ=+1, η=+1)   4: (ξ=−1, η=+1)

DOFs per node (5), total 20 per element:
  [u_x, u_s, w, β_s, β_x]
  u_x : displacement along beam axis (span)
  u_s : displacement along contour tangent
  w   : out-of-plane (normal to shell surface)
  β_s : Mindlin rotation about contour tangent  (≈ −∂w/∂x in thin limit)
  β_x : Mindlin rotation about beam axis        (≈  ∂w/∂s in thin limit)

DOF ordering inside the 20-vector:
  [node1_ux, node1_us, node1_w, node1_βs, node1_βx,
   node2_ux, …, node4_βx]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Gauss quadrature rules
# ---------------------------------------------------------------------------

def _gauss(n: int) -> tuple[NDArray, NDArray]:
    """Points and weights for n-point 1-D Gauss-Legendre on [−1, 1]."""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    if n == 2:
        p = 1.0 / np.sqrt(3.0)
        return np.array([-p, p]), np.array([1.0, 1.0])
    if n == 3:
        p = np.sqrt(3.0 / 5.0)
        return np.array([-p, 0.0, p]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    if n == 4:
        # 4-point Gauss-Legendre on [-1, 1]
        p1 = np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
        p2 = np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-p2, -p1, p1, p2]), np.array([w2, w1, w1, w2])
    raise ValueError(f"Gauss order {n} not implemented (use 1, 2, 3, or 4)")


# ---------------------------------------------------------------------------
# Bilinear shape functions
# ---------------------------------------------------------------------------

_XI_NODES = np.array([-1.0, 1.0, 1.0, -1.0])
_ET_NODES = np.array([-1.0, -1.0, 1.0, 1.0])


def _shape(xi: float, eta: float) -> NDArray:
    """N[I] = ¼(1 + ξ_I ξ)(1 + η_I η)  for I=0…3."""
    return 0.25 * (1.0 + _XI_NODES * xi) * (1.0 + _ET_NODES * eta)


def _dshape(xi: float, eta: float) -> NDArray:
    """dN/dξ  (row 0) and dN/dη  (row 1), shape (2, 4)."""
    dNdxi = 0.25 * _XI_NODES * (1.0 + _ET_NODES * eta)
    dNdet = 0.25 * _ET_NODES * (1.0 + _XI_NODES * xi)
    return np.stack([dNdxi, dNdet], axis=0)


# ---------------------------------------------------------------------------
# B-matrix construction
# ---------------------------------------------------------------------------

def _b_membrane(xi: float, eta: float, L_s: float, L_x: float) -> NDArray:
    """
    Membrane B matrix (3 × 20).

    Strains: [ε_xx, ε_ss, γ_xs]  from u_x, u_s DOFs.
    Jacobian maps (ξ, η) → (x, s): ∂/∂x = (2/L_x)∂/∂η, ∂/∂s = (2/L_s)∂/∂ξ.
    """
    dN = _dshape(xi, eta)          # (2, 4)
    dNds = dN[0] * (2.0 / L_s)    # ∂N/∂s
    dNdx = dN[1] * (2.0 / L_x)    # ∂N/∂x

    B = np.zeros((3, 20))
    for I in range(4):
        base = I * 5
        # u_x col → ε_xx = ∂u_x/∂x
        B[0, base + 0] = dNdx[I]
        # u_s col → ε_ss = ∂u_s/∂s
        B[1, base + 1] = dNds[I]
        # γ_xs = ∂u_x/∂s + ∂u_s/∂x
        B[2, base + 0] = dNds[I]
        B[2, base + 1] = dNdx[I]
    return B


def _b_bending(xi: float, eta: float, L_s: float, L_x: float) -> NDArray:
    """
    Bending B matrix (3 × 20).

    Curvatures: [κ_xx, κ_ss, κ_xs]  from β_s, β_x DOFs.
      κ_xx = ∂β_s/∂x,  κ_ss = −∂β_x/∂s,  κ_xs = ∂β_s/∂s − ∂β_x/∂x
    Sign convention: β_s rotates about s-axis, β_x about x-axis.
    """
    dN = _dshape(xi, eta)
    dNds = dN[0] * (2.0 / L_s)
    dNdx = dN[1] * (2.0 / L_x)

    B = np.zeros((3, 20))
    for I in range(4):
        base = I * 5
        # β_s (col base+3): κ_xx = ∂β_s/∂x,  κ_xs += ∂β_s/∂s
        B[0, base + 3] = dNdx[I]
        B[2, base + 3] = dNds[I]
        # β_x (col base+4): κ_ss = −∂β_x/∂s,  κ_xs += −∂β_x/∂x
        B[1, base + 4] = -dNds[I]
        B[2, base + 4] = -dNdx[I]
    return B


def _b_shear_mitc(xi: float, eta: float, L_s: float, L_x: float) -> NDArray:
    """
    MITC4 mixed-interpolated transverse shear B matrix (2 × 20).

    Shear strains: [γ_xn, γ_sn]
      γ_xn = ∂w/∂x − β_s   (shear in the x–n plane)
      γ_sn = ∂w/∂s + β_x   (shear in the s–n plane)

    MITC tying (Bathe & Dvorkin 1985):
      γ̃_xn(ξ,η) = ½(1+η) γ_xn(ξ,+1) + ½(1−η) γ_xn(ξ,−1)
      γ̃_sn(ξ,η) = ½(1+ξ) γ_sn(+1,η) + ½(1−ξ) γ_sn(−1,η)

    Each tying-point evaluation uses the standard Mindlin shear expression.
    """
    B = np.zeros((2, 20))

    # -- γ_xn  (tying along A=η=+1, B=η=−1, any ξ) ---
    for sign_eta, coeff in ((+1.0, 0.5 * (1.0 + eta)), (-1.0, 0.5 * (1.0 - eta))):
        dN_tp = _dshape(xi, sign_eta)
        dNdx_tp = dN_tp[1] * (2.0 / L_x)
        N_tp = _shape(xi, sign_eta)
        for I in range(4):
            base = I * 5
            B[0, base + 2] += coeff * dNdx_tp[I]   # ∂w/∂x at tying point
            B[0, base + 3] -= coeff * N_tp[I]       # −β_s at tying point

    # -- γ_sn  (tying along C=ξ=+1, D=ξ=−1, any η) ---
    for sign_xi, coeff in ((+1.0, 0.5 * (1.0 + xi)), (-1.0, 0.5 * (1.0 - xi))):
        dN_tp = _dshape(sign_xi, eta)
        dNds_tp = dN_tp[0] * (2.0 / L_s)
        N_tp = _shape(sign_xi, eta)
        for I in range(4):
            base = I * 5
            B[1, base + 2] += coeff * dNds_tp[I]   # ∂w/∂s at tying point
            B[1, base + 4] += coeff * N_tp[I]       # +β_x at tying point

    return B


# ---------------------------------------------------------------------------
# Element stiffness
# ---------------------------------------------------------------------------

def mitc4_stiffness(
    L_s: float,
    L_x: float,
    ABD: NDArray,
    thickness: float,
    ks: float = 5.0 / 6.0,
    G_eff: float | None = None,
) -> NDArray:
    """
    20×20 MITC4 element stiffness matrix.

    Parameters
    ----------
    L_s      : element length along contour (s) direction [m]
    L_x      : element length along span (x) direction [m]; use 1.0 for unit slice
    ABD      : 6×6 laminate stiffness matrix [[A,B],[B,D]] [N/m, N, N·m]
    thickness: wall thickness [m] (used for H_s = ks * G_eff * thickness)
    ks       : shear correction factor (5/6 default)
    G_eff    : effective transverse shear modulus [Pa]; if None, estimated from A matrix
    """
    A_mat = ABD[:3, :3]
    B_mat = ABD[:3, 3:]
    D_mat = ABD[3:, 3:]

    if G_eff is None:
        # Approximate from in-plane shear stiffness A66 / thickness
        G_eff = float(A_mat[2, 2]) / max(thickness, 1e-30)
    H_s = ks * G_eff * thickness   # transverse shear stiffness per unit area

    Jdet = (L_s / 2.0) * (L_x / 2.0)   # constant for rectangular element

    K = np.zeros((20, 20))

    # 3×3 Gauss for membrane + bending
    gp3, gw3 = _gauss(3)
    for i, (xi, wi) in enumerate(zip(gp3, gw3)):
        for j, (eta, wj) in enumerate(zip(gp3, gw3)):
            Bm = _b_membrane(xi, eta, L_s, L_x)
            Bb = _b_bending(xi, eta, L_s, L_x)
            w = wi * wj * Jdet
            K += w * (Bm.T @ A_mat @ Bm
                      + Bm.T @ B_mat @ Bb
                      + Bb.T @ B_mat.T @ Bm
                      + Bb.T @ D_mat @ Bb)

    # 2×2 Gauss for transverse shear (MITC-interpolated, no locking)
    gp2, gw2 = _gauss(2)
    H2 = np.array([[H_s, 0.0], [0.0, H_s]])
    for i, (xi, wi) in enumerate(zip(gp2, gw2)):
        for j, (eta, wj) in enumerate(zip(gp2, gw2)):
            Bs = _b_shear_mitc(xi, eta, L_s, L_x)
            w = wi * wj * Jdet
            K += w * Bs.T @ H2 @ Bs

    return K


# ---------------------------------------------------------------------------
# Resultant recovery
# ---------------------------------------------------------------------------

def mitc4_resultants(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
) -> dict[str, float]:
    """
    Recover shell resultants at element centroid (ξ=0, η=0).

    Returns dict with keys: Nx, Ny, Nxy, Mx, My, Mxy
    (all in N/m or N·m/m depending on whether membrane or bending).
    """
    Bm = _b_membrane(0.0, 0.0, L_s, L_x)
    Bb = _b_bending(0.0, 0.0, L_s, L_x)

    eps0 = Bm @ d_elem   # [ε_xx, ε_ss, γ_xs]
    kappa = Bb @ d_elem  # [κ_xx, κ_ss, κ_xs]

    A_mat = ABD[:3, :3]
    B_mat = ABD[:3, 3:]
    D_mat = ABD[3:, 3:]

    N_vec = A_mat @ eps0 + B_mat @ kappa
    M_vec = B_mat.T @ eps0 + D_mat @ kappa

    return {
        "Nx": float(N_vec[0]),
        "Ny": float(N_vec[1]),
        "Nxy": float(N_vec[2]),
        "Mx": float(M_vec[0]),
        "My": float(M_vec[1]),
        "Mxy": float(M_vec[2]),
    }


def mitc4_resultants_at(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    *,
    xi: float,
    eta: float,
) -> dict[str, float]:
    """
    Recover shell resultants at an arbitrary parent-space location (xi, eta).
    """
    Bm = _b_membrane(xi, eta, L_s, L_x)
    Bb = _b_bending(xi, eta, L_s, L_x)
    eps0 = Bm @ d_elem
    kappa = Bb @ d_elem
    A_mat = ABD[:3, :3]
    B_mat = ABD[:3, 3:]
    D_mat = ABD[3:, 3:]
    N_vec = A_mat @ eps0 + B_mat @ kappa
    M_vec = B_mat.T @ eps0 + D_mat @ kappa
    return {
        "Nx": float(N_vec[0]),
        "Ny": float(N_vec[1]),
        "Nxy": float(N_vec[2]),
        "Mx": float(M_vec[0]),
        "My": float(M_vec[1]),
        "Mxy": float(M_vec[2]),
    }


def mitc4_edge_resultants(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
) -> dict[str, dict[str, float]]:
    """
    Recover membrane resultants at edge midpoints for strip interface diagnostics.

    Returns keys:
      start -> xi=-1, eta=0
      end   -> xi=+1, eta=0
    """
    start = mitc4_resultants_at(d_elem, L_s, L_x, ABD, xi=-1.0, eta=0.0)
    end = mitc4_resultants_at(d_elem, L_s, L_x, ABD, xi=1.0, eta=0.0)
    return {
        "start": {"Nx": float(start["Nx"]), "Nxy": float(start["Nxy"])},
        "end": {"Nx": float(end["Nx"]), "Nxy": float(end["Nxy"])},
    }


def mitc4_edge_shear_traction_integrated(
    d_elem: NDArray,
    L_s: float,
    L_x: float,
    ABD: NDArray,
    *,
    edge: str,
    gauss_n: int = 4,
) -> dict[str, object]:
    """
    Line-integrate membrane traction components on interface edges (xi=±1).

    Returns mean traction components over the edge in local panel frame:
      - Nxy_edge_int: x-directed traction component (shear continuity driver)
      - Tx_edge_int: alias of Nxy_edge_int for explicit diagnostics
      - Ts_edge_int: s-directed traction component
      - Tx_gps: per-Gauss-point Tx values (list of floats, length gauss_n)
      - Ts_gps: per-Gauss-point Ts values (list of floats, length gauss_n)
    """
    if edge not in ("start", "end"):
        raise ValueError("edge must be 'start' or 'end'")
    xi = -1.0 if edge == "start" else 1.0
    normal_sign = -1.0 if edge == "start" else 1.0
    gp, gw = _gauss(gauss_n)
    j_edge = L_x / 2.0
    tx_int = 0.0
    ts_int = 0.0
    tx_gps: list[float] = []
    ts_gps: list[float] = []
    for eta, w in zip(gp, gw):
        res = mitc4_resultants_at(d_elem, L_s, L_x, ABD, xi=xi, eta=float(eta))
        # Membrane resultant tensor in local (x,s) frame
        # [ Nx  Nxy ]
        # [ Nxy Ny  ]
        nx = float(res["Nx"])
        ny = float(res["Ny"])
        nxy = float(res["Nxy"])
        nvec = np.array([0.0, normal_sign], dtype=float)
        nmat = np.array([[nx, nxy], [nxy, ny]], dtype=float)
        tvec = nmat @ nvec
        tx_gp = float(tvec[0])
        ts_gp = float(tvec[1])
        tx_gps.append(tx_gp)
        ts_gps.append(ts_gp)
        tx_int += float(w) * j_edge * tx_gp
        ts_int += float(w) * j_edge * ts_gp
    edge_len = max(L_x, 1e-30)
    tx_mean = tx_int / edge_len
    ts_mean = ts_int / edge_len
    return {
        "Nxy_edge_int": float(tx_mean),
        "Tx_edge_int": float(tx_mean),
        "Ts_edge_int": float(ts_mean),
        "Tx_gps": tx_gps,
        "Ts_gps": ts_gps,
    }
