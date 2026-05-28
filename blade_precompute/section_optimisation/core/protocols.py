"""Protocols and Tier-B drivers for prescribed-resultant beam workflows."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .types import ExtremeLoads, OptimBladeGeometry


@runtime_checkable
class BeamResultantStateProtocol(Protocol):
    """Result object returned by a beam resultant driver.

    ``resultants`` must be in section/K7 order ``[N, My, Mz, T, Vy, Vz, B]``.
    """

    resultants: NDArray[np.float64]
    nodal_R: NDArray[np.float64]


@runtime_checkable
class BeamResultantDriverProtocol(Protocol):
    """Swappable Tier-B driver: stiffness stack + loads + geometry → resultants and nodal R."""

    def drive(
        self,
        K7_stack: NDArray[np.float64],
        extreme_loads: ExtremeLoads,
        blade_geometry: OptimBladeGeometry,
        *,
        K6_stack: NDArray[np.float64] | None = None,
        mass_per_length: NDArray[np.float64] | None = None,
    ) -> BeamResultantStateProtocol: ...
