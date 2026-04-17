"""Public façade for fatigue analysis (stress history → damage)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blade_utilities.recovery import RecoveryCache

from ..core.loads import ResultantHistory
from ..core.workflows import (
    ExtremeWorkflowSpec,
    OperationalWorkflowSpec,
    validate_shared_calibration,
)
from ..engine.pipeline import FatiguePipeline
from ..engine.sn_curves import SNcurve
from ..core.types import FatigueResult


class FatigueAnalysis:
    """Thin wrapper over :class:`~blade_analysis.fatigue_damage.engine.pipeline.FatiguePipeline` with ``run``."""

    def __init__(self, pipeline: FatiguePipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def from_cache(
        cls,
        cache: RecoveryCache,
        sn_curves: dict[str, SNcurve],
        *,
        chunk_size: int = 256,
        stress_component: int = 0,
        n_range_bins: int = 128,
        apply_goodman: bool = False,
        enable_tier3_delam: bool = False,
        design_life_years: float = 25.0,
    ) -> FatigueAnalysis:
        return cls(
            FatiguePipeline(
                cache,
                sn_curves,
                chunk_size=chunk_size,
                stress_component=stress_component,
                n_range_bins=n_range_bins,
                apply_goodman=apply_goodman,
                enable_tier3_delam=enable_tier3_delam,
                design_life_years=design_life_years,
            )
        )

    def run(self, history: ResultantHistory, memory_limit_mb: float = 512.0) -> FatigueResult:
        return self._pipeline.run(history, memory_limit_mb=memory_limit_mb)

    @staticmethod
    def load_operational_loads_dat(
        path: str | Path,
        *,
        z_geometry: NDArray[np.float64] | None = None,
        z_match_tol: float = 1e-3,
    ) -> ResultantHistory:
        """
        Parse operational distributed-load ``.dat``, integrate each time slice, and return
        :class:`~blade_analysis.fatigue_damage.core.loads.ResultantHistory`.
        """
        from blade_precompute.design_optimisation.io.distributed_load_dat import resultant_history_from_operational_dat

        return resultant_history_from_operational_dat(
            path, z_geometry=z_geometry, z_match_tol=z_match_tol
        )

    def run_operational(
        self,
        operational: OperationalWorkflowSpec,
        *,
        extreme: ExtremeWorkflowSpec | None = None,
        cache: RecoveryCache | None = None,
        memory_limit_mb: float = 512.0,
    ) -> FatigueResult:
        """
        Explicit operational workflow entrypoint.

        When ``extreme`` is provided, validates shared calibration and spanwise
        station grids before running fatigue.
        """
        if extreme is not None:
            validate_shared_calibration(
                extreme,
                operational,
                cache_z_stations=(cache.z_stations if cache is not None else None),
            )
        return self._pipeline.run(operational.history, memory_limit_mb=memory_limit_mb)
