"""
Fused linear operators for Tier-1 + Tier-2 stress recovery.

Level-1 deformed-frame rotation: full 3×3 ``nodal_R`` acts on the symmetric
3×3 stress tensor embedded from plane-stress Voigt ``[σ11, σ22, τ12]``; the
returned Voigt triplet is the upper 2×2 block of ``R @ σ @ R.T`` in the same
canonical embedding (indices ``0,1`` = shell directions ``1,2``). This is a
small-angle / rigid-triad approximation consistent with
:func:`beam_model.engine.kinematics.rotmat_from_small_curvature`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.failure_criteria import tsai_wu_strength_tensors
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.core.types import SectionSolveResult

from ..core.types import RecoveryCacheStorage


def plane_stress_voigt_from_R(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Build ``(3, 3)`` matrix ``M`` with ``σ_voigt' = M @ σ_voigt`` where Voigt is
    ``[σ11, σ22, τ12]`` and ``σ`` is embedded as a symmetric ``(3, 3)`` tensor
    with zero out-of-plane rows/columns.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    B1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    B2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    B3 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    M = np.zeros((3, 3), dtype=np.float64)
    for a, B in enumerate([B1, B2, B3]):
        Sp = R @ B @ R.T
        M[0, a] = Sp[0, 0]
        M[1, a] = Sp[1, 1]
        M[2, a] = Sp[0, 1]
    return M


def _routing_indices(section0_subcomponents: Sequence[SubcomponentGeometry]) -> tuple[list[int], list[int]]:
    comp: list[int] = []
    iso: list[int] = []
    for i, sub in enumerate(section0_subcomponents):
        if sub.is_composite:
            comp.append(i)
        else:
            iso.append(i)
    return comp, iso


def _ply_strength_pad(
    section0_subcomponents: Sequence[SubcomponentGeometry],
    comp_idx: list[int],
    n_ply_max: int,
) -> tuple[NDArray[np.float64], ...]:
    Xt = np.zeros((len(comp_idx), n_ply_max), dtype=np.float64)
    Xc = np.zeros_like(Xt)
    Yt = np.zeros_like(Xt)
    Yc = np.zeros_like(Xt)
    S12 = np.zeros_like(Xt)
    for row, gi in enumerate(comp_idx):
        sub = section0_subcomponents[gi]
        assert isinstance(sub.material, LaminateDefinition)
        lam = sub.material
        n = len(lam.plies)
        for k, (ply, _) in enumerate(lam.plies):
            Xt[row, k] = ply.Xt
            Xc[row, k] = ply.Xc
            Yt[row, k] = ply.Yt
            Yc[row, k] = ply.Yc
            S12[row, k] = ply.S12
    return Xt, Xc, Yt, Yc, S12


def _ply_count_row(
    section0_subcomponents: Sequence[SubcomponentGeometry],
    comp_idx: list[int],
    n_s: int,
) -> NDArray[np.int32]:
    n_c = len(comp_idx)
    row = np.zeros((1, n_c), dtype=np.int32)
    for p, gi in enumerate(comp_idx):
        sub = section0_subcomponents[gi]
        assert isinstance(sub.material, LaminateDefinition)
        row[0, p] = int(len(sub.material.plies))
    return np.tile(row, (n_s, 1))


def build_recovery_cache(
    *,
    section_results: list[SectionSolveResult],
    z_stations: NDArray[np.float64],
    nodal_R: NDArray[np.float64] | None,
    section0_subcomponents: Sequence[SubcomponentGeometry],
    enable_tier3: bool = False,
) -> RecoveryCacheStorage:
    """
    Parameters
    ----------
    section_results
        One :class:`SectionSolveResult` per spanwise station (same topology).
    z_stations
        ``(n_s,)`` spanwise coordinates [m]; used for ``spanwise_dz`` and Tier-3 gradients.
    nodal_R
        ``(n_s, 3, 3)`` rotation per station, or ``None`` for identity.
    section0_subcomponents
        Subcomponents at the reference station (materials / ply counts).
    enable_tier3
        If true, allocate and fill ``L_rec_sec`` for section-frame ply stresses.
    """
    n_s = len(section_results)
    if n_s == 0:
        raise ValueError("section_results must be non-empty.")
    z_stations = np.asarray(z_stations, dtype=np.float64).ravel()
    if z_stations.shape[0] != n_s:
        raise ValueError("z_stations length must match section_results.")

    comp_idx, iso_idx = _routing_indices(section0_subcomponents)
    ref = section_results[0]
    n_comp = int(ref.composite_resultant_basis.shape[0])
    n_iso = int(ref.isotropic_resultant_basis.shape[0])
    n_ply_max = int(ref.Q_bar.shape[1])
    if n_comp != len(comp_idx):
        raise ValueError("Composite subcomponent count mismatch vs section0_subcomponents.")
    if n_iso != len(iso_idx):
        raise ValueError("Isotropic subcomponent count mismatch vs section0_subcomponents.")

    if nodal_R is None:
        R_stack = np.stack([np.eye(3, dtype=np.float64)] * n_s, axis=0)
    else:
        R_stack = np.asarray(nodal_R, dtype=np.float64)
        if R_stack.shape != (n_s, 3, 3):
            raise ValueError("nodal_R must have shape (n_s, 3, 3) or be None.")

    M_R = np.stack([plane_stress_voigt_from_R(R_stack[s]) for s in range(n_s)], axis=0)

    L_rec = np.zeros((n_s, n_comp, n_ply_max, 3, 7), dtype=np.float64)
    L_rec_sec = np.zeros((n_s, n_comp, n_ply_max, 3, 7), dtype=np.float64) if enable_tier3 else None

    for s in range(n_s):
        res = section_results[s]
        B = res.composite_resultant_basis
        Ainv = res.ABD_inv
        Qb = res.Q_bar
        Tp = res.T_ply
        zp = res.z_ply
        Mr = M_R[s]
        for p in range(n_comp):
            for k in range(n_ply_max):
                zpk = float(zp[p, k])
                Qk = Qb[p, k]
                Tk = Tp[p, k]
                for j in range(7):
                    N6 = B[p, j, :]
                    strain6 = Ainv[p] @ N6
                    eps0 = strain6[:3]
                    kap = strain6[3:6]
                    eps_k = eps0 + zpk * kap
                    sigma_sec = Qk @ eps_k
                    sigma_mat = Tk @ sigma_sec
                    sigma_out = Mr @ sigma_mat
                    L_rec[s, p, k, :, j] = sigma_out
                    if L_rec_sec is not None:
                        L_rec_sec[s, p, k, :, j] = sigma_sec

    L_iso = np.zeros((n_s, n_iso, 3, 7), dtype=np.float64)
    for s in range(n_s):
        res = section_results[s]
        Biso = res.isotropic_resultant_basis
        t = np.maximum(res.iso_thickness, 1e-18)
        Mr = M_R[s]
        for p in range(n_iso):
            for j in range(7):
                sig = Biso[p, j, :] / t[p]
                L_iso[s, p, :, j] = Mr @ sig

    Xt0, Xc0, Yt0, Yc0, S120 = _ply_strength_pad(section0_subcomponents, comp_idx, n_ply_max)
    f1, f2 = tsai_wu_strength_tensors(Xt0, Xc0, Yt0, Yc0, S120)
    F1 = np.broadcast_to(f1, (n_s,) + f1.shape).copy()
    F2 = np.broadcast_to(f2, (n_s,) + f2.shape).copy()

    sigma_allow_iso = np.stack([section_results[s].iso_sigma_allow for s in range(n_s)], axis=0)
    Zt = np.stack([section_results[s].Zt for s in range(n_s)], axis=0)
    S13 = np.stack([section_results[s].S13 for s in range(n_s)], axis=0)
    S23 = np.stack([section_results[s].S23 for s in range(n_s)], axis=0)

    spanwise_dz = np.diff(z_stations.astype(np.float64))
    z_ply_ref = section_results[0].z_ply.copy()

    names_c = list(ref.composite_subcomp_names)
    names_i = list(ref.isotropic_subcomp_names)
    ply_count = _ply_count_row(section0_subcomponents, comp_idx, n_s)

    K7 = np.stack([section_results[s].K7 for s in range(n_s)], axis=0)
    K6 = np.stack([section_results[s].K6 for s in range(n_s)], axis=0)
    M6 = np.stack([section_results[s].M6 for s in range(n_s)], axis=0)
    shear_center = np.stack([section_results[s].shear_center for s in range(n_s)], axis=0)
    mass_center = np.stack([section_results[s].mass_center for s in range(n_s)], axis=0)

    return RecoveryCacheStorage(
        L_rec=L_rec,
        L_iso=L_iso,
        L_rec_sec=L_rec_sec,
        F1=F1,
        F2=F2,
        sigma_allow_iso=sigma_allow_iso,
        Zt=Zt,
        S13=S13,
        S23=S23,
        spanwise_dz=spanwise_dz,
        z_stations=z_stations,
        z_ply_ref=z_ply_ref,
        composite_subcomp_idx=list(comp_idx),
        isotropic_subcomp_idx=list(iso_idx),
        composite_subcomp_names=names_c,
        isotropic_subcomp_names=names_i,
        ply_count=ply_count,
        K7=K7,
        K6=K6,
        M6=M6,
        shear_center=shear_center,
        mass_center=mass_center,
        enable_tier3=enable_tier3,
    )
