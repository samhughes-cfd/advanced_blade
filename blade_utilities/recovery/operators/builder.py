"""
Builders for runtime-only recovery operators.

Uses :class:`~blade_precompute.section_properties.core.types.SectionSolveResult` stacks
and shared routing helpers from :mod:`blade_utilities.recovery.core.section_routing`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry

from blade_utilities.recovery.core.section_routing import composite_and_isotropic_indices, ply_count_row
from blade_utilities.recovery.core.transforms import plane_stress_voigt_from_R
from blade_utilities.recovery.operators.types import RecoveryOperatorBundle


def _first_derivative_matrix(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64).ravel()
    n = int(z.shape[0])
    d_mat = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return d_mat
    if n == 2:
        dz = float(z[1] - z[0])
        if np.isclose(dz, 0.0):
            raise ValueError("z_stations must be strictly monotonic.")
        d_mat[0, 0] = -1.0 / dz
        d_mat[0, 1] = 1.0 / dz
        d_mat[1, 0] = -1.0 / dz
        d_mat[1, 1] = 1.0 / dz
        return d_mat

    def _weights(xa: float, xb: float, xc: float, x_eval: float) -> tuple[float, float, float]:
        w0 = (x_eval - xb + x_eval - xc) / ((xa - xb) * (xa - xc))
        w1 = (x_eval - xa + x_eval - xc) / ((xb - xa) * (xb - xc))
        w2 = (x_eval - xa + x_eval - xb) / ((xc - xa) * (xc - xb))
        return w0, w1, w2

    x0, x1, x2 = float(z[0]), float(z[1]), float(z[2])
    d_mat[0, 0], d_mat[0, 1], d_mat[0, 2] = _weights(x0, x1, x2, x0)

    for i in range(1, n - 1):
        xa, xb, xc = float(z[i - 1]), float(z[i]), float(z[i + 1])
        d_mat[i, i - 1], d_mat[i, i], d_mat[i, i + 1] = _weights(xa, xb, xc, xb)

    xa, xb, xc = float(z[-3]), float(z[-2]), float(z[-1])
    d_mat[-1, -3], d_mat[-1, -2], d_mat[-1, -1] = _weights(xa, xb, xc, xc)
    return d_mat


def _build_interlaminar_transfer(
    n_s: int, n_comp: int, n_ply_max: int
) -> NDArray[np.float64]:
    """
    Build a lightweight interface transfer operator.

    Output maps ``sigma_section[..., n_ply_max, 3]`` to ``[..., n_interface, 2]``.
    The current approximation distributes adjacent ply ``tau12`` equally to each
    interface and exposes two identical channels for downstream experimentation.
    """
    n_if = n_ply_max + 1
    g_if = np.zeros((n_s, n_comp, n_if, 2, n_ply_max, 3), dtype=np.float64)
    for s in range(n_s):
        for p in range(n_comp):
            for i in range(n_if):
                if i > 0:
                    g_if[s, p, i, :, i - 1, 2] += 0.5
                if i < n_ply_max:
                    g_if[s, p, i, :, i, 2] += 0.5
    return g_if


def build_recovery_operator_bundle(
    *,
    section_results: list[SectionSolveResult],
    z_stations: NDArray[np.float64],
    nodal_R: NDArray[np.float64] | None,
    section0_subcomponents: Sequence[SubcomponentGeometry],
    include_interlaminar_operator: bool = False,
) -> RecoveryOperatorBundle:
    """Build runtime-only fused operators for strain/stress and span derivatives."""
    n_s = len(section_results)
    if n_s == 0:
        raise ValueError("section_results must be non-empty.")
    z_stations = np.asarray(z_stations, dtype=np.float64).ravel()
    if z_stations.shape[0] != n_s:
        raise ValueError("z_stations length must match section_results.")

    comp_idx, _ = composite_and_isotropic_indices(section0_subcomponents)
    ref = section_results[0]
    n_comp = int(ref.composite_resultant_basis.shape[0])
    n_ply_max = int(ref.Q_bar.shape[1])
    if n_comp != len(comp_idx):
        raise ValueError("Composite subcomponent count mismatch vs section0_subcomponents.")

    if nodal_R is None:
        r_stack = np.stack([np.eye(3, dtype=np.float64)] * n_s, axis=0)
    else:
        r_stack = np.asarray(nodal_R, dtype=np.float64)
        if r_stack.shape != (n_s, 3, 3):
            raise ValueError("nodal_R must have shape (n_s, 3, 3) or be None.")
    m_voigt = np.stack([plane_stress_voigt_from_R(r_stack[s]) for s in range(n_s)], axis=0)

    h_eps = np.zeros((n_s, n_comp, 6, 7), dtype=np.float64)
    l_sec = np.zeros((n_s, n_comp, n_ply_max, 3, 7), dtype=np.float64)
    for s in range(n_s):
        res = section_results[s]
        b_mat = res.composite_resultant_basis
        ainv = res.ABD_inv
        qb = res.Q_bar
        zp = res.z_ply
        for p in range(n_comp):
            for j in range(7):
                n6 = b_mat[p, j, :]
                strain6 = ainv[p] @ n6
                h_eps[s, p, :, j] = strain6
                eps0 = strain6[:3]
                kap = strain6[3:6]
                for k in range(n_ply_max):
                    eps_k = eps0 + float(zp[p, k]) * kap
                    l_sec[s, p, k, :, j] = qb[p, k] @ eps_k

    d_z = _first_derivative_matrix(z_stations)
    g_if = (
        _build_interlaminar_transfer(n_s, n_comp, n_ply_max)
        if include_interlaminar_operator
        else None
    )
    return RecoveryOperatorBundle(
        H_eps=h_eps,
        L_sec=l_sec,
        M_voigt=m_voigt,
        D_z=d_z,
        G_if=g_if,
        z_stations=z_stations,
        composite_subcomp_idx=list(comp_idx),
        composite_subcomp_names=list(ref.composite_subcomp_names),
        ply_count=ply_count_row(section0_subcomponents, comp_idx, n_s),
    )
