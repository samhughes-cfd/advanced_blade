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

from .modal import ModalResult
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
    return SectionStiffness(EA=EA, EI_x=EI_x, EI_y=EI_y, GJ=GJ, GA_x=GA_x, GA_y=GA_y)


def section_stiffness_to_k6(st: SectionStiffness, *, EIyz: float = 0.0) -> NDArray[np.float64]:
    """Build decoupled ``(6, 6)`` section stiffness ``K6`` in beam strain order."""
    alpha = 5.0 / 6.0
    K6 = np.zeros((6, 6), dtype=np.float64)
    K6[0, 0] = st.EA
    K6[1, 1] = st.EI_x
    K6[2, 2] = st.EI_y
    K6[1, 2] = K6[2, 1] = -float(EIyz)
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
