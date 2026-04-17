"""
Section geometry: midsurface polylines with composite or isotropic assignments.

Routing uses ``SubcomponentGeometry.is_composite`` — no hard-coded part names.

**strip_width_m** multiplies per-unit-width ABD stiffness along each strip
(:math:`\\mathrm{N/m} \\times \\mathrm{m} = \\mathrm{N}` per strain).
If unset, defaults to ``thickness`` (documented heuristic; explicit width is
recommended for skins).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
from numpy.typing import NDArray

from .laminate import LaminateDefinition
from .materials import IsotropicMaterial

MaterialAssignment = Union[LaminateDefinition, IsotropicMaterial]


@dataclass
class SubcomponentGeometry:
    name: str
    midsurface_coords: NDArray[np.float64]  # (n_pts, 2) [y, z]
    material: MaterialAssignment
    thickness: float
    strip_width_m: float | None = None

    def __post_init__(self) -> None:
        self.midsurface_coords = np.asarray(self.midsurface_coords, dtype=np.float64)
        if self.midsurface_coords.ndim != 2 or self.midsurface_coords.shape[1] != 2:
            raise ValueError("midsurface_coords must have shape (n_pts, 2).")

    @property
    def is_composite(self) -> bool:
        return isinstance(self.material, LaminateDefinition)

    @property
    def is_isotropic(self) -> bool:
        return isinstance(self.material, IsotropicMaterial)

    def effective_strip_width(self) -> float:
        if self.strip_width_m is not None and self.strip_width_m > 0:
            return float(self.strip_width_m)
        return float(max(self.thickness, 1e-12))


@dataclass
class SectionDefinition:
    station_z: float
    subcomponents: List[SubcomponentGeometry]
    R_deformed: NDArray[np.float64] | None = None  # (3, 3)

    def __post_init__(self) -> None:
        if self.R_deformed is not None:
            self.R_deformed = np.asarray(self.R_deformed, dtype=np.float64)
            if self.R_deformed.shape != (3, 3):
                raise ValueError("R_deformed must be (3, 3).")
