"""
Midsurface strip section solver: warping (1D graph), ``K6``/``K7``, Tier 1 bases.

Theory version tag: **midsurface-v1** — strip-wise CLPT, graph Laplacian warping,
Bernoulli-style membrane bending coupling. Not publication-grade Vlasov shell theory.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from .assembly import apply_pin_constraint, assemble_line_laplacian, build_warping_rhs_line
from .elements import StripElementData, build_strip_fe_data
from .geometry import SectionDefinition, SubcomponentGeometry
from .laminate import LaminateDefinition
from .mesh import LineMesh, build_line_mesh, subcomponents_by_type
from .section_properties import (
    assemble_M6,
    elastic_centroid,
    mass_centroid,
    mass_per_length,
    section_area,
    shear_center_estimate,
)
from ..core.types import SectionSolveResult, SectionSolverProtocol


def _t_eff(section: SectionDefinition, si: int) -> float:
    sub = section.subcomponents[si]
    if sub.is_composite:
        return float(sub.material.total_thickness())
    return float(max(sub.thickness, 1e-12))


def _eps6_mode(
    e: int,
    mode: int,
    fe: StripElementData,
    y_e: float,
    z_e: float,
    omega_edge: float,
    omega_norm_edge: float,
) -> NDArray[np.float64]:
    """Six-vector [ε1,ε2,γ12, κ1,κ2,κ6] in x–s shell axes (1=x beam, 2=tangent)."""
    yc, zc = fe.y_mid[e], fe.z_mid[e]
    out = np.zeros(6, dtype=np.float64)
    if mode == 0:
        out[0] = 1.0
    elif mode == 1:
        out[0] = -(zc - z_e)
    elif mode == 2:
        out[0] = (yc - y_e)
    elif mode == 3:
        out[2] = 1.0
    elif mode == 4:
        out[2] = fe.tz[e]
    elif mode == 5:
        out[2] = -fe.ty[e]
    elif mode == 6:
        out[0] = omega_norm_edge
    return out


def _assemble_K6_open_strip(
    section: SectionDefinition,
    fe: StripElementData,
    y_e: float,
    z_e: float,
) -> tuple[NDArray[np.float64], float]:
    """Return (K6, GJ_strip_scalar)."""
    K6 = np.zeros((6, 6), dtype=np.float64)
    EA = EIy = EIz = EIyz = 0.0
    GJ = 0.0
    GA = 0.0
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        sub = section.subcomponents[si]
        b, L, yc, zc = fe.b[e], fe.L[e], fe.y_mid[e], fe.z_mid[e]
        t = _t_eff(section, si)
        if sub.is_composite:
            ABD = fe.ABD[e]
            A11 = ABD[0, 0]
            EA += A11 * b * L
            EIy += A11 * b * L * (zc - z_e) ** 2
            EIz += A11 * b * L * (yc - y_e) ** 2
            EIyz += A11 * b * L * (yc - y_e) * (zc - z_e)
            GJ += fe.G[e] * b * t**3 / 3.0 * L
            GA += fe.G[e] * b * t * L
        else:
            c11 = fe.C_iso[e, 0, 0]
            EA += c11 * sub.thickness * b * L
            EIy += c11 * sub.thickness * b * L * (zc - z_e) ** 2
            EIz += c11 * sub.thickness * b * L * (yc - y_e) ** 2
            EIyz += c11 * sub.thickness * b * L * (yc - y_e) * (zc - z_e)
            GJ += fe.G[e] * b * sub.thickness**3 / 3.0 * L
            GA += fe.G[e] * b * sub.thickness * L
    alpha = 5.0 / 6.0
    K6[0, 0] = EA
    K6[1, 1] = EIy
    K6[2, 2] = EIz
    K6[1, 2] = K6[2, 1] = -EIyz
    K6[3, 3] = max(GJ, 1e-6)
    K6[4, 4] = alpha * max(GA, 1e-6)
    K6[5, 5] = alpha * max(GA, 1e-6)
    return K6, GJ


def _fix_iso_arrays(
    section: SectionDefinition,
    fe: StripElementData,
    iso_idx: List[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    from .materials import IsotropicMaterial

    n_iso = len(iso_idx)
    if n_iso == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    iso_t = np.zeros(n_iso, dtype=np.float64)
    iso_C = np.zeros((n_iso, 3, 3), dtype=np.float64)
    iso_sig = np.zeros(n_iso, dtype=np.float64)
    for k, si in enumerate(iso_idx):
        sub = section.subcomponents[si]
        assert isinstance(sub.material, IsotropicMaterial)
        iso_t[k] = sub.thickness
        iso_sig[k] = sub.material.sigma_allow
        e0 = next(e for e in range(fe.n_edges) if int(fe.subcomp_idx[e]) == si)
        iso_C[k] = fe.C_iso[e0]
    return iso_t, iso_C, iso_sig


def _pad_laminate_arrays(
    section: SectionDefinition,
    comp_indices: List[int],
) -> tuple[int, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    n_comp = len(comp_indices)
    if n_comp == 0:
        return (
            1,
            np.zeros((0, 6, 6), dtype=np.float64),
            np.zeros((0, 1, 3, 3), dtype=np.float64),
            np.zeros((0, 1, 3, 3), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
        )
    n_ply_max = 0
    for si in comp_indices:
        lam = section.subcomponents[si].material
        assert isinstance(lam, LaminateDefinition)
        n_ply_max = max(n_ply_max, len(lam.plies))
    if n_ply_max == 0:
        n_ply_max = 1
    ABD_inv = np.zeros((n_comp, 6, 6), dtype=np.float64)
    Q_bar = np.zeros((n_comp, n_ply_max, 3, 3), dtype=np.float64)
    T_ply = np.zeros((n_comp, n_ply_max, 3, 3), dtype=np.float64)
    z_ply = np.zeros((n_comp, n_ply_max), dtype=np.float64)
    Zt = np.zeros((n_comp, n_ply_max), dtype=np.float64)
    S13 = np.zeros((n_comp, n_ply_max), dtype=np.float64)
    S23 = np.zeros((n_comp, n_ply_max), dtype=np.float64)
    for k, si in enumerate(comp_indices):
        lam = section.subcomponents[si].material
        assert isinstance(lam, LaminateDefinition)
        abd = lam.build_ABD()
        ABD_inv[k] = np.linalg.inv(abd)
        qb = lam.build_Q_bar()
        tp = lam.build_T_ply()
        zp = lam.ply_depths()
        n_p = len(lam.plies)
        Q_bar[k, :n_p] = qb
        T_ply[k, :n_p] = tp
        z_ply[k, :n_p] = zp
        for p, (ply, _) in enumerate(lam.plies):
            Zt[k, p] = ply.Zt
            S13[k, p] = ply.S13
            S23[k, p] = ply.S23
    return n_ply_max, ABD_inv, Q_bar, T_ply, z_ply, Zt, S13, S23


def _composite_basis(
    section: SectionDefinition,
    fe: StripElementData,
    comp_indices: List[int],
    y_e: float,
    z_e: float,
    omega: NDArray[np.float64],
    mesh: LineMesh,
) -> NDArray[np.float64]:
    n_comp = len(comp_indices)
    if n_comp == 0:
        return np.zeros((0, 7, 6), dtype=np.float64)
    basis = np.zeros((n_comp, 7, 6), dtype=np.float64)
    om_max = float(np.max(np.abs(omega)) + 1e-30)
    for c, si in enumerate(comp_indices):
        total_L = 0.0
        acc = np.zeros((7, 6), dtype=np.float64)
        for e in range(fe.n_edges):
            if int(fe.subcomp_idx[e]) != si:
                continue
            i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
            o_mid = 0.5 * (omega[i0] + omega[i1]) / om_max
            L = fe.L[e]
            for m in range(7):
                eps6 = _eps6_mode(e, m, fe, y_e, z_e, o_mid, o_mid)
                n6 = fe.ABD[e] @ eps6
                acc[m] += n6 * L
            total_L += L
        if total_L > 1e-30:
            basis[c, :, :] = acc / total_L
    return basis


def _isotropic_basis(
    section: SectionDefinition,
    fe: StripElementData,
    iso_indices: List[int],
    y_e: float,
    z_e: float,
    omega: NDArray[np.float64],
    mesh: LineMesh,
) -> NDArray[np.float64]:
    n_iso = len(iso_indices)
    basis = np.zeros((n_iso, 7, 3), dtype=np.float64)
    om_max = float(np.max(np.abs(omega)) + 1e-30)
    for c, si in enumerate(iso_indices):
        total_L = 0.0
        acc = np.zeros((7, 3), dtype=np.float64)
        for e in range(fe.n_edges):
            if int(fe.subcomp_idx[e]) != si:
                continue
            i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
            o_mid = 0.5 * (omega[i0] + omega[i1]) / om_max
            L = fe.L[e]
            t = fe.t_membrane[e]
            C = fe.C_iso[e]
            for m in range(7):
                eps6 = _eps6_mode(e, m, fe, y_e, z_e, o_mid, o_mid)
                eps3 = eps6[0:3]
                sig = C @ eps3
                n_mem = t * sig * fe.b[e]
                acc[m] += n_mem * L
            total_L += L
        if total_L > 1e-30:
            basis[c, :, :] = acc / total_L
    return basis


def _k_w_and_Kww(
    section: SectionDefinition,
    fe: StripElementData,
    omega: NDArray[np.float64],
    mesh: LineMesh,
    y_e: float,
    z_e: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    om_max = float(np.max(np.abs(omega)) + 1e-30)
    k_w = np.zeros(6, dtype=np.float64)
    K_ww = 0.0
    Eomega = np.zeros(fe.n_edges, dtype=np.float64)
    for e in range(fe.n_edges):
        si = int(fe.subcomp_idx[e])
        t = _t_eff(section, si)
        i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        o_mid = 0.5 * (omega[i0] + omega[i1])
        o_n = o_mid / om_max
        b, L = fe.b[e], fe.L[e]
        Eax = fe.E_axial[e]
        dA = b * t * L
        K_ww += Eax * o_n**2 * dA
        Eomega[e] = Eax * o_mid
        for mode in range(6):
            eps6 = _eps6_mode(e, mode, fe, y_e, z_e, o_mid, o_n)
            eps_ax = eps6[0]
            k_w[mode] += Eax * o_n * eps_ax * dA
    return k_w, float(K_ww), Eomega


class MidsurfaceSectionSolver:
    """Default midsurface strip implementation of :class:`SectionSolverProtocol`."""

    merge_tolerance: float = 1e-6

    def solve_one(self, section: SectionDefinition) -> SectionSolveResult:
        mesh = build_line_mesh(section, self.merge_tolerance)
        fe = build_strip_fe_data(section, mesh)
        if fe.n_edges == 0:
            raise ValueError("Empty section: no midsurface edges.")

        y_e, z_e = elastic_centroid(section, fe, mesh)
        y_m, z_m = mass_centroid(section, fe)
        y_s, z_s = shear_center_estimate(section, fe, y_e, z_e)

        K = assemble_line_laplacian(mesh, fe)
        f_w = build_warping_rhs_line(mesh, fe, y_e, z_e)
        K_pin, f_pin = apply_pin_constraint(K, f_w, pin_node=0)
        lu = spla.splu(K_pin.tocsc())
        omega = lu.solve(f_pin)

        k_w, K_ww, E_omega_basis = _k_w_and_Kww(section, fe, omega, mesh, y_e, z_e)
        K6, _ = _assemble_K6_open_strip(section, fe, y_e, z_e)
        K7 = np.zeros((7, 7), dtype=np.float64)
        K7[0:6, 0:6] = K6
        K7[0:6, 6] = k_w
        K7[6, 0:6] = k_w
        K7[6, 6] = K_ww

        comp_idx, iso_idx = subcomponents_by_type(section)
        names_c = [section.subcomponents[i].name for i in comp_idx]
        names_i = [section.subcomponents[i].name for i in iso_idx]

        comp_basis = _composite_basis(section, fe, comp_idx, y_e, z_e, omega, mesh)
        iso_basis = _isotropic_basis(section, fe, iso_idx, y_e, z_e, omega, mesh)

        n_ply_max, ABD_inv, Q_bar, T_ply, z_ply, Zt, S13, S23 = _pad_laminate_arrays(section, comp_idx)

        n_iso = len(iso_idx)
        iso_t, iso_C, iso_sig = _fix_iso_arrays(section, fe, iso_idx)

        M6 = assemble_M6(section, fe, y_m, z_m)
        area = section_area(section, fe)
        mpl = mass_per_length(section, fe)

        if section.R_deformed is not None:
            warnings.warn(
                "Level-1 geometric correction only (R_deformed). Section shape change "
                "under large torsion (>~15 deg) is not resolved.",
                UserWarning,
                stacklevel=2,
            )

        return SectionSolveResult(
            K7=K7,
            K6=K6,
            M6=M6,
            warping_function=omega,
            K_ww=K_ww,
            k_w=k_w,
            composite_resultant_basis=comp_basis,
            isotropic_resultant_basis=iso_basis if n_iso else np.zeros((0, 7, 3)),
            composite_subcomp_names=names_c,
            isotropic_subcomp_names=names_i,
            ABD_inv=ABD_inv,
            Q_bar=Q_bar,
            T_ply=T_ply,
            z_ply=z_ply,
            iso_thickness=iso_t,
            iso_C=iso_C,
            iso_sigma_allow=iso_sig,
            Zt=Zt,
            S13=S13,
            S23=S23,
            area=area,
            mass_per_length=mpl,
            shear_center=np.array([y_s, z_s], dtype=np.float64),
            mass_center=np.array([y_m, z_m], dtype=np.float64),
            elastic_center=np.array([y_e, z_e], dtype=np.float64),
            E_omega_basis=E_omega_basis,
        )

    def solve(self, section_defs: List[SectionDefinition]) -> List[SectionSolveResult]:
        return [self.solve_one(s) for s in section_defs]
