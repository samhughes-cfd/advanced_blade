"""
Closed-form orthotropic panel buckling for composite strip edges.

Critical load resultants (simply-supported orthotropic plate):
    N_x_cr  — Lekhnitskii / ESDU 80023, uniaxial compression
    N_xy_cr — Thielemann (1950), pure shear
    N_y_cr  — transverse compression

Interaction index (ESDU 02.03.02 style, extended with transverse compression):
    BI = R_c + R_cy + R_s**2    (BI >= 1.0 treated as buckled for the form used here)

Whitney (1987) D16/D26 knockdown applied to D11, D22.

Stress inputs are peak membrane values in the strip **section frame** (local 1 = beam axis,
local 2 = midsurface tangent): ``sigma_zz`` ≡ largest compressive ``σ11``, ``tau`` ≡ max ``|τ12|``,
``sigma_yy`` ≡ largest compressive ``σ22`` when supplied. These are mapped to ``N = σ t`` using
laminate thickness — a screening idealisation, not a full bending–stretching plate collapse analysis.

This module is **local panel** buckling in ``section_properties``. It is intentionally separate from
GBT **member** buckling in ``examples/section_beam_model`` (not run from precompute).

References
----------
Lekhnitskii, "Anisotropic Plates", 1968.
ESDU 80023 / 02.03.02.
Jones, "Mechanics of Composite Materials", ch. 5.
Whitney, "Structural Analysis of Laminated Anisotropic Plates", ch. 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from ..core.types import SectionSolveResult
from .clpt_recovery import clpt_ply_stresses_section_frame
from .elements import StripElementData
from .laminate import LaminateDefinition


@dataclass
class PanelBucklingResult:
    """Buckling result for one composite strip edge."""

    edge_idx: int
    a: float
    b: float
    N_x_cr: float
    N_xy_cr: float
    N_y_cr: float
    N_x_applied: float
    N_xy_applied: float
    N_y_applied: float
    BI: float
    R_c: float
    R_cy: float
    R_s: float
    m_critical: int
    buckled: bool


@dataclass
class PanelBucklingSectionResult:
    """Local orthotropic panel buckling summary for all composite edges at one station."""

    edge_results: List[PanelBucklingResult]
    BI_max: float
    critical_edge: int
    n_buckled: int


# Backward-compatible alias (distinct from GBT ``section_beam_model`` example naming).
SectionBucklingResult = PanelBucklingSectionResult


def composite_edge_panel_stresses_from_reference(
    result: SectionSolveResult,
    reference_forces_6: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Peak section-frame stresses per composite subcomponent for a reference resultant vector.

    Parameters
    ----------
    result
        Midsurface solve with populated ``composite_resultant_basis`` / CLPT arrays.
    reference_forces_6
        ``[N, My, Mz, T, Vy, Vz]`` consistent with ``result.K6`` row/column ordering.

    Returns
    -------
    NDArray
        ``(n_comp, 3)`` with columns ``[sigma_zz+, |tau12|_max, sigma_yy+]`` [Pa].
        Compressive normal stresses are returned as **positive** scalars for the buckling driver.
    """
    F6 = np.asarray(reference_forces_6, dtype=np.float64).reshape(6)
    basis = result.composite_resultant_basis
    if basis.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    strain7 = np.zeros(7, dtype=np.float64)
    strain7[:6] = np.linalg.solve(result.K6, F6)
    sub = np.einsum("cmj,m->cj", basis, strain7, optimize=True)
    sub_b = sub[np.newaxis, :, :]
    sig = clpt_ply_stresses_section_frame(
        sub_b,
        result.ABD_inv[np.newaxis, ...],
        result.Q_bar[np.newaxis, ...],
        result.z_ply[np.newaxis, ...],
    )[0]
    n_c, n_ply, _ = sig.shape
    out = np.zeros((n_c, 3), dtype=np.float64)
    for ic in range(n_c):
        s11 = sig[ic, :, 0]
        s22 = sig[ic, :, 1]
        t12 = sig[ic, :, 2]
        out[ic, 0] = max(0.0, -float(np.min(s11)))
        out[ic, 1] = float(np.max(np.abs(t12)))
        out[ic, 2] = max(0.0, -float(np.min(s22)))
    return out


def _bending_d_condensed(ABD: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Bending stiffness for buckling with membrane–bending coupling (3×3).

    ``D* = D − B A⁻¹ Bᵀ`` (static condensation of in-plane dofs) so asymmetric
    laminates use a physically consistent **bending** operator instead of
    uncoupled ``D`` alone.
    """
    abd = np.asarray(ABD, dtype=np.float64).reshape(6, 6)
    a3 = abd[0:3, 0:3]
    b3 = abd[0:3, 3:6]
    d3 = abd[3:6, 3:6]
    if np.linalg.norm(b3) < 1e-20 * (np.linalg.norm(d3) + 1.0):
        return d3
    reg = 1e-12 * (np.trace(a3) / 3.0 + 1.0) * np.eye(3, dtype=np.float64)
    a_inv = np.linalg.lstsq(a3 + reg, np.eye(3), rcond=None)[0]
    return d3 - b3 @ a_inv @ b3.T


def _D_effective(ABD: NDArray[np.float64]) -> Tuple[float, float, float, float]:
    """D11_eff, D12, D22_eff, D66 with Whitney D16/D26 knockdown (full 6×6 ABD)."""
    d_blk = ABD[3:6, 3:6]
    return _d_bending_whitney_knockdown(d_blk)


def _d_bending_whitney_knockdown(
    d_blk: NDArray[np.float64],
) -> Tuple[float, float, float, float]:
    d_blk = np.asarray(d_blk, dtype=np.float64).reshape(3, 3)
    d11 = float(d_blk[0, 0])
    d12 = float(d_blk[0, 1])
    d22 = float(d_blk[1, 1])
    d66 = float(d_blk[2, 2])
    d16 = float(d_blk[0, 2])
    d26 = float(d_blk[1, 2])
    if d66 > 1e-30:
        d11 = max(d11 - d16**2 / d66, 1e-30)
        d22 = max(d22 - d26**2 / d66, 1e-30)
    return d11, d12, d22, d66


def _Nx_cr(
    D11: float, D12: float, D22: float, D66: float, a: float, b: float, m_max: int = 20
) -> Tuple[float, int]:
    """Lekhnitskii N_x_cr — minimise over half-wave count m."""
    best = np.inf
    m_crit = 1
    for m in range(1, m_max + 1):
        mb_a = (m * b) / a
        a_mb = a / (m * b)
        ncr = (np.pi**2 / b**2) * (
            D11 * mb_a**2 + 2.0 * (D12 + 2.0 * D66) + D22 * a_mb**2
        ) / m**2
        if ncr < best:
            best = ncr
            m_crit = m
    return float(best), m_crit


def _Nxy_cr(D11: float, D12: float, D22: float, D66: float, a: float, b: float) -> float:
    """Thielemann N_xy_cr — orthotropic shear buckling."""
    ab = a / b
    k_iso = (5.34 + 4.0 / ab**2) if ab >= 1.0 else (4.0 + 5.34 / ab**2)
    d_eff = (D11 * D22) ** 0.25
    d_mid = D12 + 2.0 * D66
    ortho = (d_eff / np.sqrt(max(d_mid, 1e-30))) ** 0.5
    return float(k_iso * ortho * np.pi**2 / b**2 * np.sqrt(D11 * D22))


def _Ny_cr(D11: float, D12: float, D22: float, D66: float, a: float, b: float, m_max: int = 20) -> float:
    """Lekhnitskii N_y_cr — transverse compression."""
    best = np.inf
    for n in range(1, m_max + 1):
        na_b = (n * a) / b
        b_na = b / (n * a)
        ncr = (np.pi**2 / a**2) * (
            D22 * na_b**2 + 2.0 * (D12 + 2.0 * D66) + D11 * b_na**2
        ) / n**2
        if ncr < best:
            best = ncr
    return float(best)


def assess_panel_buckling_section(
    fe: StripElementData,
    comp_edge_indices: List[int],
    lams: List[LaminateDefinition],
    sigma_zz: NDArray[np.float64],
    tau: NDArray[np.float64],
    frame_spacing_m: float,
    sigma_yy: NDArray[np.float64] | None = None,
    m_max: int = 20,
    *,
    use_abd_bending_condensation: bool = True,
) -> PanelBucklingSectionResult:
    """
    Assess panel buckling for composite strip edges at one station.

    Parameters
    ----------
    fe
        Strip element data from :func:`build_strip_fe_data`.
    comp_edge_indices
        One midsurface edge index per entry in ``lams`` (same order).
    lams
        :class:`LaminateDefinition` for each composite edge.
    sigma_zz
        Peak compressive axial stress per edge [Pa] (positive = compression).
    tau
        Peak in-plane shear stress per edge [Pa].
    frame_spacing_m
        Spanwise panel length ``a`` [m].
    sigma_yy
        Optional peak compressive ``σ22`` per edge [Pa] (positive = compression).
    m_max
        Maximum half-wave count for minimisation along ``a``.
    use_abd_bending_condensation
        If True, build effective bending from ``D* = D − B A⁻¹ Bᵀ`` (full 6×6
        ABD) before the orthotropic Lekhnitskii/Thielemann closed forms; this
        better reflects asymmetric / coupled laminates. If False, use former
        treatment (``D`` block only, Whitney D16/D26 knockdown).
    """
    edge_results: List[PanelBucklingResult] = []
    bi_max = 0.0
    crit_edge = -1
    n_buckled = 0
    n = len(lams)
    syy_in = np.zeros(n, dtype=np.float64) if sigma_yy is None else np.asarray(sigma_yy, dtype=np.float64).reshape(n)

    for local_i, (e_idx, lam) in enumerate(zip(comp_edge_indices, lams)):
        abd = lam.build_ABD()
        if use_abd_bending_condensation:
            d11, d12, d22, d66 = _d_bending_whitney_knockdown(_bending_d_condensed(abd))
        else:
            d11, d12, d22, d66 = _D_effective(abd)

        b = float(fe.b[e_idx])
        a = frame_spacing_m

        if b < 1e-6 or a < 1e-6 or d11 < 1e-30:
            edge_results.append(
                PanelBucklingResult(
                    edge_idx=e_idx,
                    a=a,
                    b=b,
                    N_x_cr=1e30,
                    N_xy_cr=1e30,
                    N_y_cr=1e30,
                    N_x_applied=0.0,
                    N_xy_applied=0.0,
                    N_y_applied=0.0,
                    BI=0.0,
                    R_c=0.0,
                    R_cy=0.0,
                    R_s=0.0,
                    m_critical=1,
                    buckled=False,
                )
            )
            continue

        nx_cr, m_crit = _Nx_cr(d11, d12, d22, d66, a, b, m_max)
        nxy_cr = _Nxy_cr(d11, d12, d22, d66, a, b)
        ny_cr = _Ny_cr(d11, d12, d22, d66, a, b, m_max)

        t_eff = float(lam.total_thickness())
        n_x_app = max(float(sigma_zz[local_i]), 0.0) * t_eff
        n_xy_app = abs(float(tau[local_i])) * t_eff
        n_y_app = max(float(syy_in[local_i]), 0.0) * t_eff

        r_c = n_x_app / nx_cr if nx_cr > 1e-30 else 0.0
        r_s = n_xy_app / nxy_cr if nxy_cr > 1e-30 else 0.0
        r_cy = n_y_app / ny_cr if ny_cr > 1e-30 else 0.0
        bi = r_c + r_cy + r_s**2

        res = PanelBucklingResult(
            edge_idx=e_idx,
            a=a,
            b=b,
            N_x_cr=nx_cr,
            N_xy_cr=nxy_cr,
            N_y_cr=ny_cr,
            N_x_applied=n_x_app,
            N_xy_applied=n_xy_app,
            N_y_applied=n_y_app,
            BI=bi,
            R_c=r_c,
            R_cy=r_cy,
            R_s=r_s,
            m_critical=m_crit,
            buckled=(bi >= 1.0),
        )
        edge_results.append(res)

        if bi > bi_max:
            bi_max = bi
            crit_edge = e_idx
        if bi >= 1.0:
            n_buckled += 1

    return PanelBucklingSectionResult(
        edge_results=edge_results,
        BI_max=float(bi_max),
        critical_edge=crit_edge,
        n_buckled=n_buckled,
    )
