"""Builders for runtime-only recovery operators."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition

from blade_utilities.stress_recovery.engine.builder import plane_stress_voigt_from_R

from ..core.types import RecoveryOperatorBundle


def _routing_indices(section0_subcomponents: Sequence[SubcomponentGeometry]) -> list[int]:
    comp: list[int] = []
    for i, sub in enumerate(section0_subcomponents):
        if sub.is_composite:
            comp.append(i)
    return comp


def _ply_count_row(
    section0_subcomponents: Sequence[SubcomponentGeometry],
    comp_idx: list[int],
    n_s: int,
) -> NDArray[np.int32]:
    row = np.zeros((1, len(comp_idx)), dtype=np.int32)
    for p, gi in enumerate(comp_idx):
        sub = section0_subcomponents[gi]
        assert isinstance(sub.material, LaminateDefinition)
        row[0, p] = int(len(sub.material.plies))
    return np.tile(row, (n_s, 1))


def _first_derivative_matrix(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=np.float64).ravel()
    n = int(z.shape[0])
    D = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return D
    if n == 2:
        dz = float(z[1] - z[0])
        if np.isclose(dz, 0.0):
            raise ValueError("z_stations must be strictly monotonic.")
        D[0, 0] = -1.0 / dz
        D[0, 1] = 1.0 / dz
        D[1, 0] = -1.0 / dz
        D[1, 1] = 1.0 / dz
        return D

    def _weights(xa: float, xb: float, xc: float, x_eval: float) -> tuple[float, float, float]:
        w0 = (x_eval - xb + x_eval - xc) / ((xa - xb) * (xa - xc))
        w1 = (x_eval - xa + x_eval - xc) / ((xb - xa) * (xb - xc))
        w2 = (x_eval - xa + x_eval - xb) / ((xc - xa) * (xc - xb))
        return w0, w1, w2

    x0, x1, x2 = float(z[0]), float(z[1]), float(z[2])
    D[0, 0], D[0, 1], D[0, 2] = _weights(x0, x1, x2, x0)

    for i in range(1, n - 1):
        xa, xb, xc = float(z[i - 1]), float(z[i]), float(z[i + 1])
        D[i, i - 1], D[i, i], D[i, i + 1] = _weights(xa, xb, xc, xb)

    xa, xb, xc = float(z[-3]), float(z[-2]), float(z[-1])
    D[-1, -3], D[-1, -2], D[-1, -1] = _weights(xa, xb, xc, xc)
    return D


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
    G_if = np.zeros((n_s, n_comp, n_if, 2, n_ply_max, 3), dtype=np.float64)
    for s in range(n_s):
        for p in range(n_comp):
            for i in range(n_if):
                if i > 0:
                    G_if[s, p, i, :, i - 1, 2] += 0.5
                if i < n_ply_max:
                    G_if[s, p, i, :, i, 2] += 0.5
    return G_if


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

    comp_idx = _routing_indices(section0_subcomponents)
    ref = section_results[0]
    n_comp = int(ref.composite_resultant_basis.shape[0])
    n_ply_max = int(ref.Q_bar.shape[1])
    if n_comp != len(comp_idx):
        raise ValueError("Composite subcomponent count mismatch vs section0_subcomponents.")

    if nodal_R is None:
        R_stack = np.stack([np.eye(3, dtype=np.float64)] * n_s, axis=0)
    else:
        R_stack = np.asarray(nodal_R, dtype=np.float64)
        if R_stack.shape != (n_s, 3, 3):
            raise ValueError("nodal_R must have shape (n_s, 3, 3) or be None.")
    M_voigt = np.stack([plane_stress_voigt_from_R(R_stack[s]) for s in range(n_s)], axis=0)

    H_eps = np.zeros((n_s, n_comp, 6, 7), dtype=np.float64)
    L_sec = np.zeros((n_s, n_comp, n_ply_max, 3, 7), dtype=np.float64)
    for s in range(n_s):
        res = section_results[s]
        B = res.composite_resultant_basis
        Ainv = res.ABD_inv
        Qb = res.Q_bar
        zp = res.z_ply
        for p in range(n_comp):
            for j in range(7):
                N6 = B[p, j, :]
                strain6 = Ainv[p] @ N6
                H_eps[s, p, :, j] = strain6
                eps0 = strain6[:3]
                kap = strain6[3:6]
                for k in range(n_ply_max):
                    eps_k = eps0 + float(zp[p, k]) * kap
                    L_sec[s, p, k, :, j] = Qb[p, k] @ eps_k

    D_z = _first_derivative_matrix(z_stations)
    G_if = (
        _build_interlaminar_transfer(n_s, n_comp, n_ply_max)
        if include_interlaminar_operator
        else None
    )
    return RecoveryOperatorBundle(
        H_eps=H_eps,
        L_sec=L_sec,
        M_voigt=M_voigt,
        D_z=D_z,
        G_if=G_if,
        z_stations=z_stations,
        composite_subcomp_idx=list(comp_idx),
        composite_subcomp_names=list(ref.composite_subcomp_names),
        ply_count=_ply_count_row(section0_subcomponents, comp_idx, n_s),
    )
