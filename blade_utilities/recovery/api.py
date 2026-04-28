"""Public façade for fused recovery cache construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.core.types import SectionSolveResult
from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry

from blade_utilities.recovery.core.cache_types import RecoveryCacheStorage
from blade_utilities.recovery.core.transforms import plane_stress_voigt_from_R
from blade_utilities.recovery.tensor_cache.builder import build_recovery_cache
from blade_utilities.recovery.tensor_cache.cache import RecoveryCache


class RecoveryCacheBuilder:
    """Thin wrapper around :func:`~blade_utilities.recovery.tensor_cache.builder.build_recovery_cache`."""

    @staticmethod
    def build(
        section_results: Sequence[SectionSolveResult],
        section0_subcomponents: Sequence[SubcomponentGeometry],
        z_stations: NDArray[np.float64],
        *,
        nodal_R_stack: NDArray[np.float64] | None = None,
    ) -> RecoveryCacheStorage:
        return build_recovery_cache(
            section_results=list(section_results),
            z_stations=z_stations,
            nodal_R=nodal_R_stack,
            section0_subcomponents=section0_subcomponents,
        )

    @staticmethod
    def plane_stress_voigt_from_rotation(R: NDArray[np.float64]) -> NDArray[np.float64]:
        return plane_stress_voigt_from_R(R)


__all__ = ["RecoveryCacheBuilder", "RecoveryCache", "RecoveryCacheStorage"]
