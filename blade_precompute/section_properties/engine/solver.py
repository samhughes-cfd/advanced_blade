"""
Midsurface strip section solver: warping (1D graph), energy-consistent K7, CLPT bases.

Theory version tag: **midsurface-v2** — unified full-shell strip elements (composite
and isotropic), energy-integral K7, and a single subcomponent basis.

K7 assembly
-----------
Every strip edge carries a 6×6 laminate ABD (composite) or isotropic shell ABD
(A = Q·t, B = 0, D = Q·t³/12). The cross-section stiffness K7[m,n] is the bilinear
energy integral

    K7[m, n] = Σ_e  L_e · ε6_mode(e, m)^T @ ABD(e) @ ε6_mode(e, n)

where ε6_mode(e, m) is the 6-component CLPT strain vector in the strip frame for
generalised beam mode m (modes 0-5: axial/bending/shear/torsion, mode 6: warping).
This is **exactly dual** to the subcomponent resultant basis, ensuring K7_inv @ R
gives beam-mode amplitudes that project correctly back to ply stresses.
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


def _assemble_K7_strip(
    fe: StripElementData,
    y_e: float,
    z_e: float,
    omega: NDArray[np.float64],
    mesh: LineMesh,
) -> NDArray[np.float64]:
    """Energy-integral 7×7 cross-section stiffness.

    K7[m, n] = Σ_e  L_e · ε6_mode(e,m)^T @ ABD(e) @ ε6_mode(e,n)

    All 7 generalised beam modes (including warping, mode 6) and the full
    per-edge ABD (composite or isotropic shell) are used, making K7 exactly
    dual to the subcomponent resultant basis computed by :func:`_subcomponent_basis`.
    """
    om_max = float(np.max(np.abs(omega)) + 1e-30)
    K7 = np.zeros((7, 7), dtype=np.float64)
    for e in range(fe.n_edges):
        i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        o_mid = 0.5 * (omega[i0] + omega[i1]) / om_max
        L = fe.L[e]
        # Stack ε6 for all 7 modes into (7, 6) matrix
        eps6_modes = np.stack(
            [_eps6_mode(e, m, fe, y_e, z_e, o_mid, o_mid) for m in range(7)]
        )
        K7 += L * (eps6_modes @ fe.ABD[e] @ eps6_modes.T)
    return K7


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


def _subcomponent_basis(
    fe: StripElementData,
    sub_indices: List[int],
    y_e: float,
    z_e: float,
    omega: NDArray[np.float64],
    mesh: LineMesh,
) -> NDArray[np.float64]:
    """Length-averaged CLPT resultant basis per subcomponent.

    Returns
    -------
    basis : (n_sub, 7, 6)
        ``basis[c, m, :]`` = length-weighted average of ``ABD(e) @ ε6_mode(e, m)``
        over all edges belonging to subcomponent c.  Consistent with the energy
        form used by :func:`_assemble_K7_strip` because both use the same ABD and
        the same :func:`_eps6_mode` kernel.

    Replaces the former ``_composite_basis`` (shape (n,7,6)) and
    ``_isotropic_basis`` (shape (n,7,3)) with a single unified path.
    Isotropic edges now carry the full shell ABD so the output is always 6-wide.
    """
    n_sub = len(sub_indices)
    if n_sub == 0:
        return np.zeros((0, 7, 6), dtype=np.float64)
    basis = np.zeros((n_sub, 7, 6), dtype=np.float64)
    om_max = float(np.max(np.abs(omega)) + 1e-30)
    for c, si in enumerate(sub_indices):
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
                acc[m] += fe.ABD[e] @ eps6 * L
            total_L += L
        if total_L > 1e-30:
            basis[c] = acc / total_L
    return basis


def _e_omega_basis(
    fe: StripElementData,
    omega: NDArray[np.float64],
    mesh: LineMesh,
) -> NDArray[np.float64]:
    """Per-edge Eω product (E_axial × ω_mid) used in downstream homogenisation."""
    Eomega = np.zeros(fe.n_edges, dtype=np.float64)
    for e in range(fe.n_edges):
        i0, i1 = int(mesh.edges[e, 0]), int(mesh.edges[e, 1])
        o_mid = 0.5 * (omega[i0] + omega[i1])
        Eomega[e] = fe.E_axial[e] * o_mid
    return Eomega


def _build_iso_clpt_arrays(
    section: SectionDefinition,
    fe: StripElementData,
    iso_indices: List[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build ABD_inv, Q_bar, z_ply for isotropic subs (single outer-surface ply).

    The isotropic sub is treated as a single-ply shell evaluated at z = +t/2
    (outer surface), giving σ = N/t + 6M/t² which captures membrane + bending.

    Returns
    -------
    iso_ABD_inv : (n_iso, 6, 6)
    iso_Q_bar   : (n_iso, 1, 3, 3)
    iso_z_ply   : (n_iso, 1)
    """
    from .materials import IsotropicMaterial, build_isotropic_ABD, plane_stress_Q_isotropic

    n_iso = len(iso_indices)
    if n_iso == 0:
        return (
            np.zeros((0, 6, 6), dtype=np.float64),
            np.zeros((0, 1, 3, 3), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
        )
    iso_ABD_inv = np.zeros((n_iso, 6, 6), dtype=np.float64)
    iso_Q_bar = np.zeros((n_iso, 1, 3, 3), dtype=np.float64)
    iso_z_ply = np.zeros((n_iso, 1), dtype=np.float64)
    for k, si in enumerate(iso_indices):
        sub = section.subcomponents[si]
        assert isinstance(sub.material, IsotropicMaterial)
        mat = sub.material
        t = float(max(sub.thickness, 1e-12))
        ABD = build_isotropic_ABD(mat.E, mat.nu, t)
        iso_ABD_inv[k] = np.linalg.inv(ABD)
        iso_Q_bar[k, 0] = plane_stress_Q_isotropic(mat.E, mat.nu)
        iso_z_ply[k, 0] = t / 2.0  # outer surface: captures membrane + bending
    return iso_ABD_inv, iso_Q_bar, iso_z_ply


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

        # Principal sectorial coordinate normalisation: remove the E-weighted projection
        # of omega onto {1, -(z-z_e), (y-y_e)} so that k_w[0:3] = 0 and K7 is positive
        # definite. Without this, the pin at node 0 leaves an arbitrary constant that
        # couples the warping column of K7 to axial/bending, violating the Cauchy-Schwarz
        # bound and giving a negative K7 eigenvalue.
        _E_dA = np.array([
            fe.E_axial[e] * fe.b[e] * _t_eff(section, int(fe.subcomp_idx[e])) * fe.L[e]
            for e in range(fe.n_edges)
        ])
        _o_mid = 0.5 * (omega[mesh.edges[:, 0]] + omega[mesh.edges[:, 1]])
        _phi1 = -(fe.z_mid - z_e)
        _phi2 = fe.y_mid - y_e
        _ones = np.ones(fe.n_edges, dtype=np.float64)
        _G = np.array([
            [np.dot(_E_dA, _ones),           np.dot(_E_dA, _phi1),            np.dot(_E_dA, _phi2)],
            [np.dot(_E_dA, _phi1),           np.dot(_E_dA, _phi1 * _phi1),    np.dot(_E_dA, _phi1 * _phi2)],
            [np.dot(_E_dA, _phi2),           np.dot(_E_dA, _phi1 * _phi2),    np.dot(_E_dA, _phi2 * _phi2)],
        ])
        _rhs = np.array([
            np.dot(_E_dA, _o_mid),
            np.dot(_E_dA, _o_mid * _phi1),
            np.dot(_E_dA, _o_mid * _phi2),
        ])
        _coeffs = np.linalg.solve(_G, _rhs)
        _yn = mesh.nodes[:, 0]
        _zn = mesh.nodes[:, 1]
        omega -= _coeffs[0] + _coeffs[1] * (-(_zn - z_e)) + _coeffs[2] * (_yn - y_e)

        # Energy-consistent K7 from full shell ABD on all edges (Fix 4b)
        K7 = _assemble_K7_strip(fe, y_e, z_e, omega, mesh)
        K6 = K7[0:6, 0:6].copy()
        k_w = K7[0:6, 6].copy()
        K_ww = float(K7[6, 6])
        E_omega_basis = _e_omega_basis(fe, omega, mesh)

        comp_idx, iso_idx = subcomponents_by_type(section)
        names_c = [section.subcomponents[i].name for i in comp_idx]
        names_i = [section.subcomponents[i].name for i in iso_idx]

        # Unified basis for all subcomponents (Fix 4b)
        comp_basis = _subcomponent_basis(fe, comp_idx, y_e, z_e, omega, mesh)
        iso_basis = _subcomponent_basis(fe, iso_idx, y_e, z_e, omega, mesh)

        n_ply_max, ABD_inv, Q_bar, T_ply, z_ply, Zt, S13, S23 = _pad_laminate_arrays(section, comp_idx)

        n_iso = len(iso_idx)
        iso_t, iso_C, iso_sig = _fix_iso_arrays(section, fe, iso_idx)
        iso_ABD_inv, iso_Q_bar_clpt, iso_z_ply_clpt = _build_iso_clpt_arrays(section, fe, iso_idx)

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
            isotropic_resultant_basis=iso_basis if n_iso else np.zeros((0, 7, 6)),
            composite_subcomp_names=names_c,
            isotropic_subcomp_names=names_i,
            ABD_inv=ABD_inv,
            Q_bar=Q_bar,
            T_ply=T_ply,
            z_ply=z_ply,
            iso_thickness=iso_t,
            iso_C=iso_C,
            iso_sigma_allow=iso_sig,
            iso_ABD_inv=iso_ABD_inv,
            iso_Q_bar=iso_Q_bar_clpt,
            iso_z_ply=iso_z_ply_clpt,
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
