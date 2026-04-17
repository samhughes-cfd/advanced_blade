"""
beam_model/constitutive.py
==========================
Reissner strains, section ordering, and seven-vector resultants.

Spatial Reissner strains (parameter ``s`` = reference arc length)::

    Γ = Λᵀ x′ − e₁,   [Ω]_× = Λᵀ dΛ/ds,  Ω = axial([Ω]_×)

Mechanical curvature subtracts prescribed ``κ₀`` in the **same** frame as
``Ω`` (material / convected triad at the Gauss point)::

    Ω_mech = Ω − κ₀

Section six-vector (``section_model``)::

    e_sec = P @ [Γ; Ω_mech]

Seven-vector constitutive law::

    [r_sec; B] = K7 @ [e_sec; χ],   χ = dψ/ds − χ₀

Beam resultant order (first six)::

    [N, Vy, Vz, My, Mz, T, B]

``σ_warp(ξ) = −E ω(ξ) d²ψ/dz²`` in downstream ``section_model`` uses ``B`` from
column 6 with the sign convention implied by ``K7`` assembly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

P_SECTION: NDArray[np.float64] = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
    dtype=np.float64,
)

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def reissner_strains(
    x_prime: NDArray[np.float64],
    R: NDArray[np.float64],
    dR_ds0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    Rt = R.T
    Gamma = Rt @ x_prime - E1
    Omega_mat = Rt @ dR_ds0
    Omega_mat = 0.5 * (Omega_mat - Omega_mat.T)
    Omega = np.array(
        [Omega_mat[2, 1], Omega_mat[0, 2], Omega_mat[1, 0]], dtype=np.float64
    )
    return Gamma, Omega


def section_strain_vector(Gamma: NDArray[np.float64], Omega: NDArray[np.float64]) -> NDArray[np.float64]:
    v = np.concatenate([Gamma, Omega], axis=0)
    return P_SECTION @ v


def section_strain_mechanical(
    Gamma: NDArray[np.float64],
    Omega: NDArray[np.float64],
    kappa0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Six-vector with mechanical curvature ``Ω − κ₀``."""
    Om_m = Omega - np.asarray(kappa0, dtype=np.float64).reshape(3)
    return section_strain_vector(Gamma, Om_m)


def section_resultants(K6: NDArray[np.float64], e: NDArray[np.float64]) -> NDArray[np.float64]:
    return K6 @ e


def synthesize_K7(K6: NDArray[np.float64], K7: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Return full ``(7,7)`` stiffness, defaulting to decoupled warping diagonal."""
    if K7 is not None:
        K = np.asarray(K7, dtype=np.float64).reshape(7, 7)
        return K
    out = np.zeros((7, 7), dtype=np.float64)
    out[:6, :6] = K6
    g = float(max(K6[3, 3], 1e-6))
    out[6, 6] = g
    return out


def section_resultants_natural(K7f: NDArray[np.float64], e7: NDArray[np.float64]) -> NDArray[np.float64]:
    """``K7 @ e7`` in native ordering ``[N,My,Mz,T,Vy,Vz,B]`` (for energy gradients)."""
    return K7f @ np.asarray(e7, dtype=np.float64).reshape(7)


def section_resultants_seven(K7f: NDArray[np.float64], e7: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    ``r7 = K7 @ e7`` then reorder first six entries to
    ``[N, Vy, Vz, My, Mz, T]``; seventh is ``B``.
    """
    r = K7f @ np.asarray(e7, dtype=np.float64).reshape(7)
    N, My, Mz, T, Vy, Vz = r[:6]
    B = float(r[6])
    return np.array([N, Vy, Vz, My, Mz, T, B], dtype=np.float64)


def strain_vector_seven(e_sec: NDArray[np.float64], chi: float) -> NDArray[np.float64]:
    return np.concatenate([np.asarray(e_sec, dtype=np.float64).reshape(6), [float(chi)]], axis=0)


def resultants_to_recovery6(r7: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Map beam seven-vector ``[N,Vy,Vz,My,Mz,T,B]`` first six entries to
    ``section_model`` order ``[N,My,Mz,T,Vy,Vz]`` for :meth:`RecoveryModel.recover_stress`.
    """
    r = np.asarray(r7, dtype=np.float64).reshape(-1)
    if r.shape[0] < 6:
        raise ValueError("Need at least six resultants.")
    N, Vy, Vz, My, Mz, T = r[0], r[1], r[2], r[3], r[4], r[5]
    return np.array([N, My, Mz, T, Vy, Vz], dtype=np.float64)


def beam_resultants_to_section_recovery_order(r7: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Map beam seven-vector ``[N,Vy,Vz,My,Mz,T,B]`` to fused recovery column order
    ``[N,My,Mz,T,Vy,Vz,B]`` used by ``blade_utilities.recovery_operators`` /
    ``blade_utilities.stress_recovery``.
    """
    x = np.asarray(r7, dtype=np.float64)
    if x.shape[-1] != 7:
        raise ValueError("last axis must be length 7 (beam resultants).")
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0] = x[..., 0]
    out[..., 1] = x[..., 3]
    out[..., 2] = x[..., 4]
    out[..., 3] = x[..., 5]
    out[..., 4] = x[..., 1]
    out[..., 5] = x[..., 2]
    out[..., 6] = x[..., 6]
    return out
