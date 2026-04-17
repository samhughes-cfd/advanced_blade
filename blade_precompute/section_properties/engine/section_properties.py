"""
Integrated section properties from midsurface strip discretisation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .elements import StripElementData
from .geometry import SectionDefinition
from .mesh import LineMesh
from ..core.types import SectionSolveResult


def elastic_centroid(section: SectionDefinition, fe: StripElementData, mesh: LineMesh) -> NDArray[np.float64]:
    """Stiffness-weighted centroid in the y–z plane."""
    numy, numz, den = 0.0, 0.0, 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L = fe.b[e], fe.L[e]
        if sub.is_composite:
            A11 = fe.ABD[e, 0, 0]
            w = A11 * b * L
        else:
            w = fe.C_iso[e, 0, 0] * sub.thickness * b * L
        den += w
        numy += w * fe.y_mid[e]
        numz += w * fe.z_mid[e]
    if den <= 1e-30:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.array([numy / den, numz / den], dtype=np.float64)


def mass_centroid(section: SectionDefinition, fe: StripElementData) -> NDArray[np.float64]:
    numy, numz, den = 0.0, 0.0, 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L = fe.b[e], fe.L[e]
        if sub.is_composite:
            lam = sub.material
            rho = lam.equivalent_density()
            h = lam.total_thickness()
            w = rho * h * b * L
        else:
            mat = sub.material
            w = mat.rho * sub.thickness * b * L
        den += w
        numy += w * fe.y_mid[e]
        numz += w * fe.z_mid[e]
    if den <= 1e-30:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.array([numy / den, numz / den], dtype=np.float64)


def mass_per_length(section: SectionDefinition, fe: StripElementData) -> float:
    mu = 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L = fe.b[e], fe.L[e]
        if sub.is_composite:
            lam = sub.material
            mu += lam.equivalent_density() * lam.total_thickness() * b * L
        else:
            mu += sub.material.rho * sub.thickness * b * L
    return float(mu)


def section_area(section: SectionDefinition, fe: StripElementData) -> float:
    a = 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L = fe.b[e], fe.L[e]
        if sub.is_composite:
            a += sub.material.total_thickness() * b * L
        else:
            a += sub.thickness * b * L
    return float(a)


def shear_center_estimate(section: SectionDefinition, fe: StripElementData, y_e: float, z_e: float) -> NDArray[np.float64]:
    """
    Energy-style offset from elastic centroid (thin-walled line analogue).

    Uses transverse stiffness-weighted lever arm; falls back to elastic centre.
    """
    if fe.n_edges == 0:
        return np.array([y_e, z_e], dtype=np.float64)
    numy, numz, den = 0.0, 0.0, 0.0
    for e in range(fe.n_edges):
        w = fe.G[e] * fe.b[e] * fe.L[e]
        den += w
        numy += w * fe.y_mid[e]
        numz += w * fe.z_mid[e]
    if den <= 1e-30:
        return np.array([y_e, z_e], dtype=np.float64)
    return np.array([numy / den, numz / den], dtype=np.float64)


def assemble_M6(section: SectionDefinition, fe: StripElementData, y_m: float, z_m: float) -> NDArray[np.float64]:
    """Mass matrix in ``[ε0, κy, κz, γt, γsy, γsz]`` ordering (same as legacy)."""
    mu = mass_per_length(section, fe)
    Iy_m, Iz_m = 0.0, 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L = fe.b[e], fe.L[e]
        ym = fe.y_mid[e] - y_m
        zm = fe.z_mid[e] - z_m
        if sub.is_composite:
            lam = sub.material
            dm = lam.equivalent_density() * lam.total_thickness() * b * L
        else:
            dm = sub.material.rho * sub.thickness * b * L
        Iy_m += dm * zm**2
        Iz_m += dm * ym**2
    M6 = np.zeros((6, 6), dtype=np.float64)
    M6[0, 0] = mu
    M6[1, 1] = Iy_m
    M6[2, 2] = Iz_m
    M6[3, 3] = Iy_m + Iz_m
    M6[4, 4] = mu
    M6[5, 5] = mu
    return M6


def print_section_summary(res: SectionSolveResult) -> None:
    """Print key scalars and diagonal stiffnesses for quick inspection."""
    print("--- Section summary ---")
    print(f"area [m2]           : {res.area:.6g}")
    print(f"mass/length [kg/m]  : {res.mass_per_length:.6g}")
    print(f"elastic centre (y,z): {res.elastic_center}")
    print(f"mass centre (y,z)   : {res.mass_center}")
    print(f"shear centre (y,z)  : {res.shear_center}")
    print(f"K_ww                : {res.K_ww:.6g}")
    print("K6 diagonal:", np.diag(res.K6))
    print("K7 diagonal:", np.diag(res.K7))
