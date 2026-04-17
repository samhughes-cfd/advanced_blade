"""
Map GBT cross-section modal results to classical beam stiffness scalars and ``K6``.

Axis convention (matches :func:`section_stiffness_to_k6` with
:mod:`blade_precompute.section_properties.engine.solver`):

- ``EI_x`` / ``bending_x`` → ``K6[1, 1]`` (bending stiffness associated with ``κ_y``).
- ``EI_y`` / ``bending_y`` → ``K6[2, 2]`` (``κ_z``).
- ``GA_x`` → ``K6[4, 4]`` after Timoshenko factor ``5/6`` (shear strain ``γ_sy``).
- ``GA_y`` → ``K6[5, 5]`` (``γ_sz``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.core.types import SectionStation

from .modal import ModalResult, classical_export_indices
from .prebuckling import PreBucklingAnalysis, SectionLoads
from .section import CrossSection


@dataclass(frozen=True)
class SectionStiffness:
    EA: float
    EI_x: float
    EI_y: float
    GJ: float
    GA_x: float
    GA_y: float
    #: Bending–bending coupling from GBT: ``φ_xᵀ C φ_y`` (enters ``K6`` as ``-EIyz`` off-diagonal).
    EIyz: float = 0.0


def _shear_ga_from_strips(section: CrossSection) -> tuple[float, float]:
    """
    Aggregate strip shear stiffness into two section-axis scalars [N].

    Uses each strip's ``shear_stiffness()`` matrix (``2×2``) scaled by strip length
    and split between section ``y`` / ``z`` using the parent wall normal components.
    """
    ga_y = 0.0
    ga_z = 0.0
    for i in range(section.n_strips):
        s = section.get_strip(i)
        Ks = np.asarray(section.strip_shear_stiffness(i), dtype=np.float64)
        ds = float(s.length)
        if Ks.size == 0:
            continue
        if Ks.shape != (2, 2):
            smax = float(np.max(np.abs(Ks)))
        else:
            smax = float(np.linalg.eigvalsh(0.5 * (Ks + Ks.T) + 1e-30 * np.eye(2)).max())
        klen = smax * ds
        wn = section.walls[int(s.wall_id)].normal
        ny, nz = float(wn[0]), float(wn[1])
        ga_y += klen * ny**2
        ga_z += klen * nz**2
    return float(max(ga_y, 1e-12)), float(max(ga_z, 1e-12))


def gbt_to_beam_stiffness(
    result: ModalResult,
    selected_modes: ModalResult,
    *,
    section: CrossSection,
    loads: SectionLoads | None = None,
    strict_classical: bool = True,
) -> SectionStiffness:
    """
    Extract classical ``EA``, ``EI_*``, ``GJ``, ``GA_*`` for downstream beam models.

    ``EA`` is taken from :class:`PreBucklingAnalysis` strip integration (same as
    laminate extensional stiffness). ``EI_*`` and ``GJ`` use ``φᵀ C φ`` modal
    rigidities from ``selected_modes`` for the export-labelled bending and torsion
    modes (``M``-orthonormal modes satisfy ``modal_rigidity(k) == λ_k``).

    ``GA_x`` / ``GA_y`` are aggregated from strip shear stiffness matrices.

    ``EIyz`` is the cross-modal stiffness ``φ_{bending_x}ᵀ C φ_{bending_y}`` when both
    bending export modes are present; otherwise ``0``.
    """
    if result.n_dof != selected_modes.n_dof or result.C.shape != selected_modes.C.shape:
        raise ValueError("result and selected_modes must share the same section eigenproblem (C, n_dof).")
    if selected_modes.section is None:
        raise ValueError("selected_modes.section must be set (run CrossSectionModalAnalysis).")
    n = len(selected_modes.eigenvalues)
    labels_needed = ("axial", "bending_x", "bending_y", "torsion")
    found: dict[str, int] = {}
    for k in range(n):
        lab = selected_modes.classify_export_mode(k)
        if lab in labels_needed and lab not in found:
            found[lab] = int(k)

    if strict_classical:
        missing = [lb for lb in labels_needed if lb not in found]
        if missing:
            raise ValueError(
                "gbt_to_beam_stiffness: required export modes not found in selected_modes: "
                f"{missing}. Present labels: "
                f"{[selected_modes.classify_export_mode(k) for k in range(n)]}."
            )

    loads_use = loads if loads is not None else SectionLoads(N=-1.0)
    snap = PreBucklingAnalysis(section, loads_use).section_properties()
    EA = float(snap["EA"])

    def _rig(lab: str) -> float:
        if lab not in found:
            return 0.0
        j = int(found[lab])
        return float(selected_modes.modal_rigidity(j))

    EI_x = _rig("bending_x")
    EI_y = _rig("bending_y")
    GJ = _rig("torsion")
    GA_x, GA_y = _shear_ga_from_strips(section)
    if "bending_x" in found and "bending_y" in found:
        ix = int(found["bending_x"])
        iy = int(found["bending_y"])
        EIyz = float(selected_modes.modal_coupling(ix, iy))
    else:
        EIyz = 0.0
    return SectionStiffness(EA=EA, EI_x=EI_x, EI_y=EI_y, GJ=GJ, GA_x=GA_x, GA_y=GA_y, EIyz=EIyz)


def gbt_to_k7(
    result: ModalResult,
    K6: NDArray[np.float64],
    *,
    warping_index: int | None = None,
) -> NDArray[np.float64]:
    """
    Build a ``(7, 7)`` section stiffness with GBT-derived torsion–warping coupling.

    Copies ``K6`` into the leading block. Uses the full modal basis ``result`` (not
    the four-mode truncation) to pick a warping-like mode ``k_w`` and sets
    ``K7[3, 6] = K7[6, 3] = φ_torsionᵀ C φ_{k_w}``, ``K7[6, 6] = φ_{k_w}ᵀ C φ_{k_w}``.

    By default ``k_w`` is the mode index *not* among the classical export quartet
    that maximises ``|modal_coupling(torsion, k)|`` (tie-break: smaller eigenvalue).

    Other warping couplings (extension, bending, shear–warping) are left zero; only
    the torsion row/column is populated in the seventh DOF besides ``K7[6, 6]``.
    """
    K6 = np.asarray(K6, dtype=np.float64).reshape(6, 6)
    if result.section is None:
        raise ValueError("gbt_to_k7 requires ModalResult.section.")
    picks = classical_export_indices(result, result.section)
    k_t = int(picks["torsion"])
    classical_set = {int(v) for v in picks.values()}
    n = len(result.eigenvalues)

    K7 = np.zeros((7, 7), dtype=np.float64)
    K7[:6, :6] = K6

    if warping_index is not None:
        k_w = int(warping_index)
    else:
        best_k: int | None = None
        best_c = -1.0
        for k in range(n):
            if k in classical_set:
                continue
            c_abs = abs(result.modal_coupling(k_t, k))
            lam_k = float(result.eigenvalues[k])
            if best_k is None:
                best_k, best_c = k, c_abs
            elif c_abs > best_c + 1e-30 * max(best_c, 1.0):
                best_k, best_c = k, c_abs
            elif abs(c_abs - best_c) <= 1e-30 * max(best_c, 1.0) and best_k is not None:
                if lam_k < float(result.eigenvalues[best_k]):
                    best_k = k
        if best_k is None:
            g = float(max(K6[3, 3], 1e-6))
            K7[6, 6] = g
            return 0.5 * (K7 + K7.T)

        k_w = best_k

    c_tw = float(result.modal_coupling(k_t, k_w))
    k_ww = float(result.modal_rigidity(k_w))
    K7[3, 6] = K7[6, 3] = c_tw
    K7[6, 6] = float(max(k_ww, max(K6[3, 3], 1e-6), 1e-12))
    return 0.5 * (K7 + K7.T)


def section_stiffness_to_k6(st: SectionStiffness, *, EIyz: float | None = None) -> NDArray[np.float64]:
    """Build ``(6, 6)`` section stiffness ``K6`` in beam strain order (optional ``EIyz`` off-diagonal)."""
    eyz = float(st.EIyz if EIyz is None else EIyz)
    alpha = 5.0 / 6.0
    K6 = np.zeros((6, 6), dtype=np.float64)
    K6[0, 0] = st.EA
    K6[1, 1] = st.EI_x
    K6[2, 2] = st.EI_y
    K6[1, 2] = K6[2, 1] = -eyz
    K6[3, 3] = max(st.GJ, 1e-12)
    K6[4, 4] = alpha * max(st.GA_x, 1e-12)
    K6[5, 5] = alpha * max(st.GA_y, 1e-12)
    return K6


def section_stiffness_to_station(
    z: float,
    st: SectionStiffness,
    *,
    K7: NDArray[np.float64] | None = None,
) -> SectionStation:
    """Wrap ``SectionStiffness`` as a :class:`SectionStation` at arc coordinate ``z``."""
    return SectionStation(z=float(z), K6=section_stiffness_to_k6(st), K7=K7)
