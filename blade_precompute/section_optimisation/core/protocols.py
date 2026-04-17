"""Protocols and Tier-B drivers for prescribed-resultant beam workflows."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..engine.beam_k7 import PrescribedResultantBeamState, solve as prescribed_resultant_solve
from .types import ExtremeLoads, OptimBladeGeometry


@runtime_checkable
class BeamResultantDriverProtocol(Protocol):
    """Swappable Tier-B driver: stiffness stack + loads + geometry → resultants and nodal R."""

    def drive(
        self,
        K7_stack: NDArray[np.float64],
        extreme_loads: ExtremeLoads,
        blade_geometry: OptimBladeGeometry,
    ) -> PrescribedResultantBeamState: ...


class PrescribedResultantDriver:
    """Default implementation wrapping :func:`~design_optimisation.engine.beam_k7.solve`."""

    __slots__ = ()

    def drive(
        self,
        K7_stack: NDArray[np.float64],
        extreme_loads: ExtremeLoads,
        blade_geometry: OptimBladeGeometry,
    ) -> PrescribedResultantBeamState:
        return prescribed_resultant_solve(K7_stack, extreme_loads, blade_geometry)
