"""
Tier 3 interlaminar utilities.

- :func:`interlaminar_stress_recovery` / :func:`delamination_fi` — gradient-based
  recovery along the span (needs ``n_s >= 2``).
- Equilibrium-based first-moment recovery (:func:`recover_interlaminar`) —
  screening at a station for given ``Vy``, ``Vz``.

Limits of :func:`recover_interlaminar` vs strip equilibrium
-----------------------------------------------------------
:class:`~blade_precompute.section_properties.core.types.SectionSolveResult` carries
``K6`` (6×6 stiffness), ``K7`` (7×7 with warping), ply ``ABD_inv`` / ``Q_bar`` for CLPT,
and ``composite_resultant_basis`` ``(n_comp, 7, 6)`` mapping the seven midsurface strain
modes into subcomponent membrane/bending resultants. There is **no** per-edge row of
``K6`` that would give ``Vy``, ``Vz``, or ``T`` resolved strip-by-strip: transverse shear
is applied only through the **section-level** scalars ``EIy = K6[1,1]`` and
``EIz = K6[2,2]`` together with the laminate first moment ``Q_axial(z)``.

That path is equilibrium-based in the sense ``τ ∝ V·Q / EI``, but it does **not**
enforce **strip-graph** nodal equilibrium, **shear-centre** load decomposition beyond what
callers pass in ``(Vy, Vz)``, or **multicell** / **Bredt** circulations for torque ``T``.
For a strip-aware scaling of the same closed-form envelope, see
:mod:`blade_precompute.section_properties.engine.strip_shear_equilibrium`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

from .laminate import LaminateDefinition


def interlaminar_stress_recovery(
    sigma_inplane: NDArray[np.float64],
    z_stations: NDArray[np.float64],
    z_ply: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Recover ``[τ13, τ23, σ33]`` at ply interfaces.

    In-plane stress gradients use :func:`numpy.gradient` along span (axis 0).
    Through-thickness integration uses uniform ply thickness inferred from
    ``z_ply`` span.

    Parameters
    ----------
    sigma_inplane
        ``(n_s, n_comp, n_ply, 3)`` — ``[σ11, σ22, τ12]``.
    z_stations
        ``(n_s,)`` spanwise positions [m].
    z_ply
        ``(n_comp, n_ply)`` ply mid-ordinates [m].

    Returns
    -------
    NDArray
        ``(n_s, n_comp, n_ply + 1, 3)`` interface values bottom → top.
    """
    n_s, n_comp, n_ply, _ = sigma_inplane.shape
    n_if = n_ply + 1
    out = np.zeros((n_s, n_comp, n_if, 3), dtype=np.float64)
    if n_s < 2 or n_ply == 0:
        return out

    d11 = np.gradient(sigma_inplane[..., 0], z_stations, axis=0)
    d22 = np.gradient(sigma_inplane[..., 1], z_stations, axis=0)
    d12 = np.gradient(sigma_inplane[..., 2], z_stations, axis=0)

    for c in range(n_comp):
        zp = z_ply[c, :n_ply]
        zmin, zmax = float(np.min(zp)), float(np.max(zp))
        dz = (zmax - zmin) / max(n_ply, 1)
        if dz <= 0:
            dz = 1e-6
        for s in range(n_s):
            acc13 = 0.0
            acc23 = 0.0
            out[s, c, 0, :] = 0.0
            for j in range(1, n_if):
                k = j - 1
                kp = min(k, n_ply - 1)
                g13 = -(d11[s, c, kp] + d12[s, c, kp])
                g23 = -(d12[s, c, kp] + d22[s, c, kp])
                acc13 += g13 * dz
                acc23 += g23 * dz
                out[s, c, j, 0] = acc13
                out[s, c, j, 1] = acc23
                out[s, c, j, 2] = 0.0
    return out


def delamination_fi(
    sigma_interlaminar: NDArray[np.float64],
    Zt: NDArray[np.float64],
    S13: NDArray[np.float64],
    S23: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Delamination failure index at interfaces.

    ``⟨σ33⟩ = max(σ33, 0)`` (Macaulay bracket — tensile only).
    Strength arrays ``(n_comp, n_ply)`` are averaged to interface columns.
    """
    tau13 = sigma_interlaminar[..., 0]
    tau23 = sigma_interlaminar[..., 1]
    s33 = np.maximum(sigma_interlaminar[..., 2], 0.0)
    n_if = sigma_interlaminar.shape[-2]
    n_p = Zt.shape[1]
    Zi = np.zeros((Zt.shape[0], n_if), dtype=np.float64)
    S13i = np.zeros_like(Zi)
    S23i = np.zeros_like(Zi)
    Zi[:, 0] = Zt[:, 0]
    Zi[:, -1] = Zt[:, min(n_p - 1, 0)]
    if n_if > 2 and n_p > 1:
        Zi[:, 1:-1] = 0.5 * (Zt[:, :-1] + Zt[:, 1:])
        S13i[:, 1:-1] = 0.5 * (S13[:, :-1] + S13[:, 1:])
        S23i[:, 1:-1] = 0.5 * (S23[:, :-1] + S23[:, 1:])
    S13i[:, 0] = S13[:, 0]
    S13i[:, -1] = S13[:, min(n_p - 1, 0)]
    S23i[:, 0] = S23[:, 0]
    S23i[:, -1] = S23[:, min(n_p - 1, 0)]
    Zi = np.maximum(Zi, 1e-18)
    S13i = np.maximum(S13i, 1e-18)
    S23i = np.maximum(S23i, 1e-18)
    return (s33 / Zi) ** 2 + (tau13 / S13i) ** 2 + (tau23 / S23i) ** 2


# ---------------------------------------------------------------------------
# Equilibrium-based interlaminar screening (station loads)
# ---------------------------------------------------------------------------


@dataclass
class InterfaceIFI:
    """Interlaminar state at one ply interface."""

    z_interface: float
    sigma_13: float
    sigma_23: float
    IFI: float


@dataclass
class EdgeInterlaminarResult:
    """Interlaminar results for one composite strip edge."""

    edge_idx: int
    interfaces: List[InterfaceIFI]
    IFI_max: float
    z_critical: float


@dataclass
class SectionInterlaminarResult:
    """Interlaminar result for one station cross-section."""

    edge_results: List[EdgeInterlaminarResult]
    IFI_global: float
    critical_edge: int
    critical_z: float


def _Q_axial(lam: LaminateDefinition) -> NDArray[np.float64]:
    """
    First moment of axial stiffness at each interface (bottom surface Q=0).

    Returns ``(n_ply + 1,)`` in Pa·m².
    """
    z_ifaces = lam.ply_interfaces_z()
    q = np.zeros(len(z_ifaces))
    for k, (ply, angle_deg) in enumerate(lam.plies):
        theta = np.radians(angle_deg)
        c2 = np.cos(theta) ** 2
        s2 = np.sin(theta) ** 2
        e_ax = (
            ply.E1 * c2**2
            + ply.E2 * s2**2
            + 2.0 * (ply.E1 * ply.nu12 + 2.0 * ply.G12) * s2 * c2
        )
        z_bar = 0.5 * (z_ifaces[k] + z_ifaces[k + 1])
        q[k + 1] = q[k] + e_ax * z_bar * ply.t_ply
    return q


def _governing_strengths(lam: LaminateDefinition) -> tuple[float, float]:
    """Minimum S13, S23 across plies; fall back to S12 when unset."""
    s13 = min((ply.S13 if ply.S13 > 0.0 else ply.S12) for ply, _ in lam.plies)
    s23 = min((ply.S23 if ply.S23 > 0.0 else ply.S12) for ply, _ in lam.plies)
    return max(float(s13), 1e-3), max(float(s23), 1e-3)


def _recover_edge(
    edge_idx: int,
    lam: LaminateDefinition,
    vy: float,
    vz: float,
    eiy: float,
    eiz: float,
) -> EdgeInterlaminarResult:
    q_ax = _Q_axial(lam)
    z_ifaces = lam.ply_interfaces_z()
    s13_g, s23_g = _governing_strengths(lam)

    interfaces: List[InterfaceIFI] = []
    ifi_max = 0.0
    z_crit = float(z_ifaces[0])

    for k in range(len(z_ifaces)):
        s13 = vz * q_ax[k] / eiy if eiy > 1e-30 else 0.0
        s23 = vy * q_ax[k] / eiz if eiz > 1e-30 else 0.0
        ifi = (s13 / s13_g) ** 2 + (s23 / s23_g) ** 2
        interfaces.append(
            InterfaceIFI(
                z_interface=float(z_ifaces[k]),
                sigma_13=float(s13),
                sigma_23=float(s23),
                IFI=float(ifi),
            )
        )
        if ifi > ifi_max:
            ifi_max = ifi
            z_crit = float(z_ifaces[k])

    return EdgeInterlaminarResult(
        edge_idx=edge_idx,
        interfaces=interfaces,
        IFI_max=float(ifi_max),
        z_critical=z_crit,
    )


def recover_interlaminar(
    comp_edge_indices: List[int],
    lams: List[LaminateDefinition],
    vy: float,
    vz: float,
    eiy: float,
    eiz: float,
) -> SectionInterlaminarResult:
    """
    Recover interlaminar stresses for composite edges at one station.

    Parameters
    ----------
    comp_edge_indices
        Edge indices (labels only; not used in the closed-form Q(z) model).
    lams
        One laminate per edge index (same order).
    vy, vz
        Applied transverse shear forces [N].
    eiy, eiz
        Bending stiffnesses from ``K6`` diagonal [N·m²].
    """
    edge_results: List[EdgeInterlaminarResult] = []
    ifi_global = 0.0
    crit_edge = -1
    crit_z = 0.0

    for e_idx, lam in zip(comp_edge_indices, lams):
        res = _recover_edge(e_idx, lam, vy, vz, eiy, eiz)
        edge_results.append(res)
        if res.IFI_max > ifi_global:
            ifi_global = res.IFI_max
            crit_edge = e_idx
            crit_z = res.z_critical

    return SectionInterlaminarResult(
        edge_results=edge_results,
        IFI_global=float(ifi_global),
        critical_edge=crit_edge,
        critical_z=crit_z,
    )


def build_interlaminar_operators(
    lams: List[LaminateDefinition],
    eiy: float,
    eiz: float,
) -> NDArray[np.float64]:
    """
    Precompute R_il for vectorised use: σ_13 per unit Vz, σ_23 per unit Vy.

    Returns ``(n_edges, n_iface_max, 2)`` with ``[...,0] = Q/EIy``, ``[...,1] = Q/EIz``.
    """
    if not lams:
        return np.zeros((0, 0, 2), dtype=np.float64)

    n_iface_max = max(len(lam.plies) + 1 for lam in lams)
    r_il = np.zeros((len(lams), n_iface_max, 2), dtype=np.float64)

    for i, lam in enumerate(lams):
        q_ax = _Q_axial(lam)
        n_iface = len(q_ax)
        r_il[i, :n_iface, 0] = q_ax / eiy if eiy > 1e-30 else 0.0
        r_il[i, :n_iface, 1] = q_ax / eiz if eiz > 1e-30 else 0.0

    return r_il
