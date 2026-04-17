"""Public façade for Tier-C fused recovery cache construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.core.types import SectionSolveResult

from ..engine.builder import build_recovery_cache, plane_stress_voigt_from_R
from ..engine.cache import RecoveryCache
from ..core.types import RecoveryCacheStorage


class RecoveryCacheBuilder:
    """``build`` wraps :func:`~recovery_cache.engine.builder.build_recovery_cache`."""

    @staticmethod
    def build(
        section_results: Sequence[SectionSolveResult],
        section0_subcomponents: Sequence[SubcomponentGeometry],
        z_stations: NDArray[np.float64],
        *,
        nodal_R_stack: NDArray[np.float64] | None = None,
        enable_tier3: bool = False,
    ) -> RecoveryCacheStorage:
        return build_recovery_cache(
            section_results=list(section_results),
            z_stations=z_stations,
            nodal_R=nodal_R_stack,
            section0_subcomponents=section0_subcomponents,
            enable_tier3=enable_tier3,
        )

    @staticmethod
    def plane_stress_voigt_from_rotation(R: NDArray[np.float64]) -> NDArray[np.float64]:
        return plane_stress_voigt_from_R(R)


__all__ = ["RecoveryCacheBuilder", "RecoveryCache", "RecoveryCacheStorage"]
