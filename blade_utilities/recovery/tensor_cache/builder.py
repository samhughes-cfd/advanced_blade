"""
Fused linear operators for Tier-1 + Tier-2 stress recovery (tensor cache).

Builds material-frame ply stress maps and isotropic membrane maps from
:class:`~blade_precompute.section_properties.core.types.SectionSolveResult` stacks.

Dependency boundary: this layer imports laminate geometry and failure helpers from
``blade_precompute.section_properties``; ``blade_precompute`` beam workflows import
``blade_utilities.recovery`` upward. There is no import cycle with ``global_beam_model``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry

from blade_utilities.recovery.core.cache_types import RecoveryCacheStorage
from blade_utilities.recovery.core.section_routing import (
    composite_and_isotropic_indices,
    ply_count_row,
    ply_strength_pad,
)
from blade_utilities.recovery.core.transforms import plane_stress_voigt_from_R


def build_recovery_cache(
    *,
    section_results: list[SectionSolveResult],
    z_stations: NDArray[np.float64],
    nodal_R: NDArray[np.float64] | None,
    section0_subcomponents: Sequence[SubcomponentGeometry],
) -> RecoveryCacheStorage:
    """
    Parameters
    ----------
    section_results
        One :class:`SectionSolveResult` per spanwise station (same topology).
    z_stations
        ``(n_s,)`` spanwise coordinates [m]; used for ``spanwise_dz``.
    nodal_R
        ``(n_s, 3, 3)`` rotation per station, or ``None`` for identity.
    section0_subcomponents
        Subcomponents at the reference station (materials / ply counts).
    """
    n_s = len(section_results)
    if n_s == 0:
        raise ValueError("section_results must be non-empty.")
    z_stations = np.asarray(z_stations, dtype=np.float64).ravel()
    if z_stations.shape[0] != n_s:
        raise ValueError("z_stations length must match section_results.")

    comp_idx, iso_idx = composite_and_isotropic_indices(section0_subcomponents)
    ref = section_results[0]
    n_comp = int(ref.composite_resultant_basis.shape[0])
    n_iso = int(ref.isotropic_resultant_basis.shape[0])
    n_ply_max = int(ref.Q_bar.shape[1])
    if n_comp != len(comp_idx):
        raise ValueError("Composite subcomponent count mismatch vs section0_subcomponents.")
    if n_iso != len(iso_idx):
        raise ValueError("Isotropic subcomponent count mismatch vs section0_subcomponents.")

    if nodal_R is None:
        r_stack = np.stack([np.eye(3, dtype=np.float64)] * n_s, axis=0)
    else:
        r_stack = np.asarray(nodal_R, dtype=np.float64)
        if r_stack.shape != (n_s, 3, 3):
            raise ValueError("nodal_R must have shape (n_s, 3, 3) or be None.")

    m_r = np.stack([plane_stress_voigt_from_R(r_stack[s]) for s in range(n_s)], axis=0)

    l_rec = np.zeros((n_s, n_comp, n_ply_max, 3, 7), dtype=np.float64)

    for s in range(n_s):
        res = section_results[s]
        b_mat = res.composite_resultant_basis
        ainv = res.ABD_inv
        qb = res.Q_bar
        tp = res.T_ply
        zp = res.z_ply
        mr = m_r[s]
        for p in range(n_comp):
            for k in range(n_ply_max):
                zpk = float(zp[p, k])
                qk = qb[p, k]
                tk = tp[p, k]
                for j in range(7):
                    n6 = b_mat[p, j, :]
                    strain6 = ainv[p] @ n6
                    eps0 = strain6[:3]
                    kap = strain6[3:6]
                    eps_k = eps0 + zpk * kap
                    sigma_sec = qk @ eps_k
                    sigma_mat = tk @ sigma_sec
                    sigma_out = mr @ sigma_mat
                    l_rec[s, p, k, :, j] = sigma_out

    l_iso = np.zeros((n_s, n_iso, 3, 7), dtype=np.float64)
    for s in range(n_s):
        res = section_results[s]
        biso = res.isotropic_resultant_basis
        t = np.maximum(res.iso_thickness, 1e-18)
        mr = m_r[s]
        for p in range(n_iso):
            for j in range(7):
                sig = biso[p, j, :] / t[p]
                l_iso[s, p, :, j] = mr @ sig

    xt0, xc0, yt0, yc0, s120 = ply_strength_pad(section0_subcomponents, comp_idx, n_ply_max)
    xt_full = np.broadcast_to(xt0, (n_s,) + xt0.shape).copy()
    xc_full = np.broadcast_to(xc0, (n_s,) + xc0.shape).copy()
    yt_full = np.broadcast_to(yt0, (n_s,) + yt0.shape).copy()
    yc_full = np.broadcast_to(yc0, (n_s,) + yc0.shape).copy()
    s12_full = np.broadcast_to(s120, (n_s,) + s120.shape).copy()

    sigma_allow_iso = np.stack([section_results[s].iso_sigma_allow for s in range(n_s)], axis=0)
    zt = np.stack([section_results[s].Zt for s in range(n_s)], axis=0)
    s13 = np.stack([section_results[s].S13 for s in range(n_s)], axis=0)
    s23 = np.stack([section_results[s].S23 for s in range(n_s)], axis=0)

    spanwise_dz = np.diff(z_stations.astype(np.float64))
    z_ply_ref = section_results[0].z_ply.copy()

    names_c = list(ref.composite_subcomp_names)
    names_i = list(ref.isotropic_subcomp_names)
    ply_count = ply_count_row(section0_subcomponents, comp_idx, n_s)

    k7 = np.stack([section_results[s].K7 for s in range(n_s)], axis=0)

    # L_rec / L_iso were built per unit beam STRAIN (mode j → ply stress). The public
    # RecoveryCache API takes beam RESULTANTS (forces/moments), so bake in K7_inv here
    # so that downstream callers can pass resultants directly.
    k7_inv = np.linalg.inv(k7)  # (n_s, 7, 7)
    l_rec = np.einsum("spkqm,smj->spkqj", l_rec, k7_inv, optimize=True)
    l_iso = np.einsum("spqm,smj->spqj", l_iso, k7_inv, optimize=True)

    k6 = np.stack([section_results[s].K6 for s in range(n_s)], axis=0)
    m6 = np.stack([section_results[s].M6 for s in range(n_s)], axis=0)
    shear_center = np.stack([section_results[s].shear_center for s in range(n_s)], axis=0)
    mass_center = np.stack([section_results[s].mass_center for s in range(n_s)], axis=0)

    return RecoveryCacheStorage(
        L_rec=l_rec,
        L_iso=l_iso,
        Xt=xt_full,
        Xc=xc_full,
        Yt=yt_full,
        Yc=yc_full,
        S12=s12_full,
        sigma_allow_iso=sigma_allow_iso,
        Zt=zt,
        S13=s13,
        S23=s23,
        spanwise_dz=spanwise_dz,
        z_stations=z_stations,
        z_ply_ref=z_ply_ref,
        composite_subcomp_idx=list(comp_idx),
        isotropic_subcomp_idx=list(iso_idx),
        composite_subcomp_names=names_c,
        isotropic_subcomp_names=names_i,
        ply_count=ply_count,
        K7=k7,
        K6=k6,
        M6=m6,
        shear_center=shear_center,
        mass_center=mass_center,
    )
