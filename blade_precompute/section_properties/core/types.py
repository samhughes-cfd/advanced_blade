"""
Types for midsurface cross-section analysis (Vlasov extension + CLPT).

Beam axis **x**; section plane **(y, z)**.

Generalised beam resultants (``K6`` / ``K7`` coupling to six classical strains):

  index 0 : N   — axial force
  index 1 : My  — bending moment about y
  index 2 : Mz  — bending moment about z
  index 3 : T   — St. Venant torque
  index 4 : Vy  — shear force y
  index 5 : Vz  — shear force z

Mode index **6** is the restrained warping / bimoment-associated mode.

**Known limitation:** Level-1 geometric correction only (``R_deformed``);
large torsion (>~15°) needs shell-based section analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class MaterialProps:
    """Orthotropic block for legacy CST tooling (optional)."""

    mat_id: int
    name: str
    E1: float
    E2: float
    E3: float
    G12: float
    G13: float
    G23: float
    nu12: float
    nu13: float
    nu23: float
    rho: float
    E_axial: float = field(init=False, repr=False)
    G_section: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.E_axial = self.E1
        self.G_section = self.G23


@dataclass
class SectionSolveResult:
    K7: NDArray[np.float64]
    K6: NDArray[np.float64]
    M6: NDArray[np.float64]
    warping_function: NDArray[np.float64]
    K_ww: float
    k_w: NDArray[np.float64]
    composite_resultant_basis: NDArray[np.float64]
    isotropic_resultant_basis: NDArray[np.float64]
    composite_subcomp_names: List[str]
    isotropic_subcomp_names: List[str]
    ABD_inv: NDArray[np.float64]
    Q_bar: NDArray[np.float64]
    T_ply: NDArray[np.float64]
    z_ply: NDArray[np.float64]
    iso_thickness: NDArray[np.float64]
    iso_C: NDArray[np.float64]
    iso_sigma_allow: NDArray[np.float64]
    Zt: NDArray[np.float64]
    S13: NDArray[np.float64]
    S23: NDArray[np.float64]
    area: float
    mass_per_length: float
    shear_center: NDArray[np.float64]
    mass_center: NDArray[np.float64]
    elastic_center: NDArray[np.float64]
    E_omega_basis: NDArray[np.float64] | None = None


@runtime_checkable
class SectionSolverProtocol(Protocol):
    """Protocol so midsurface FE can be swapped for BECAS / VABS wrappers."""

    def solve(self, section_defs: List[object]) -> List[SectionSolveResult]: ...

    def solve_one(self, section_def: object) -> SectionSolveResult: ...


@dataclass
class SectionProps:
    """
    Legacy-compatible 6×6 section property bag for beam coupling.

    Prefer :class:`SectionSolveResult` for the midsurface pipeline.
    """

    K6: NDArray[np.float64]
    M6: NDArray[np.float64]
    elastic_center: NDArray[np.float64]
    mass_center: NDArray[np.float64]
    shear_center: NDArray[np.float64]
    EA: float
    EIy: float
    EIz: float
    EIyz: float
    GJ: float
    mu: float
    K7: NDArray[np.float64] | None = None
