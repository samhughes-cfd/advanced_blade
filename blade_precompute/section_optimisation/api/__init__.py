"""Public façade for blade sizing: geometry load, station build, evaluation (Tier B)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import SectionDefinition

from ..engine.evaluator import DesignEvaluator
from ..engine.section_builder import SectionBuilder
from ..core.types import (
    DesignEvaluation,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimBladeGeometry,
)
from ..io.yaml_loader import load_blade_geometry


class BladeDesignProblem:
    """
    Cohesive sizing workflow: design problem + ``evaluate`` / ``build_sections``.

    Uses :class:`~section_optimisation.core.protocols.PrescribedResultantDriver` internally
    via :class:`~section_optimisation.engine.evaluator.DesignEvaluator`.
    """

    def __init__(self, problem: DesignProblem) -> None:
        self._problem = problem
        self._evaluator = DesignEvaluator(problem)

    @property
    def problem(self) -> DesignProblem:
        return self._problem

    @staticmethod
    def load_geometry(path: str | Path) -> OptimBladeGeometry:
        """Load :class:`~section_optimisation.core.types.OptimBladeGeometry` from YAML."""
        return load_blade_geometry(path)

    @staticmethod
    def load_extreme_loads_dat(
        path: str | Path,
        *,
        z_geometry: NDArray[np.float64] | None = None,
        z_match_tol: float = 1e-3,
    ) -> ExtremeLoads:
        """
        Load extreme distributed-load ``.dat``, integrate to internal resultants, and return
        :class:`~section_optimisation.core.types.ExtremeLoads`.

        When ``z_geometry`` is the problem blade ``z_stations``, load coordinates are validated.
        """
        from ..io.distributed_load_dat import (
            extreme_loads_from_distributed,
            load_extreme_distributed_loads_dat,
        )

        z, q_y, q_z, m_x = load_extreme_distributed_loads_dat(
            path, z_geometry=z_geometry, z_match_tol=z_match_tol
        )
        return extreme_loads_from_distributed(z, q_y, q_z, m_x)

    def build_sections(self, dv: DesignVector) -> list[SectionDefinition]:
        """Spanwise :class:`~section_model.engine.geometry.SectionDefinition` list for a design vector."""
        return SectionBuilder.build(dv, self._problem.blade_geometry)

    def evaluate(self, dv: DesignVector) -> DesignEvaluation:
        """Full section → beam (Tier B) → stress → failure evaluation."""
        return self._evaluator.evaluate(dv)
