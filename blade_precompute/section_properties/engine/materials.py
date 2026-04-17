"""
Orthotropic ply and isotropic metal definitions for midsurface section analysis.

Beam axis **x** (out of section plane); shell/laminate plane **1–2** lies in the
section **y–z** plane with **3** through-thickness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class IsotropicMaterial:
    name: str
    E: float
    nu: float
    rho: float
    sigma_allow: float  # von Mises allowable [Pa]


@dataclass
class OrthotropicPly:
    name: str
    E1: float
    E2: float
    G12: float
    nu12: float
    rho: float
    t_ply: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S12: float
    Zt: float
    S13: float
    S23: float


def plane_stress_Q(ply: OrthotropicPly) -> NDArray[np.float64]:
    """
    Stiffness matrix Q in ply material axes (1, 2), plane stress [Pa].

    Ordering: [σ11, σ22, τ12]^T = Q @ [ε11, ε22, γ12]^T.
    """
    e1, e2, g12, nu12 = ply.E1, ply.E2, ply.G12, ply.nu12
    nu21 = nu12 * e2 / e1
    denom = 1.0 - nu12 * nu21
    if denom <= 0:
        raise ValueError(f"Invalid elastic constants: 1 - nu12*nu21 = {denom}")
    q11 = e1 / denom
    q12 = nu12 * e2 / denom
    q22 = e2 / denom
    q66 = g12
    return np.array(
        [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]],
        dtype=np.float64,
    )


def plane_stress_Q_isotropic(E: float, nu: float) -> NDArray[np.float64]:
    """Plane-stress Q for isotropic material (same basis as ply)."""
    g = E / (2.0 * (1.0 + nu))
    q11 = E / (1.0 - nu * nu)
    q12 = nu * E / (1.0 - nu * nu)
    q22 = q11
    q66 = g
    return np.array(
        [[q11, q12, 0.0], [q12, q22, 0.0], [0.0, 0.0, q66]],
        dtype=np.float64,
    )


def shear_modulus_section(E: float, nu: float) -> float:
    """Shear modulus proxy for section Laplacian [Pa]."""
    return float(E / (2.0 * (1.0 + max(nu, 1e-9))))


def rotation_matrix_theta(theta_deg: float) -> NDArray[np.float64]:
    """Strain transformation matrix (Reuter vector), θ in degrees."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array(
        [
            [c * c, s * s, 2 * s * c],
            [s * s, c * c, -2 * s * c],
            [-s * c, s * c, c * c - s * s],
        ],
        dtype=np.float64,
    )
