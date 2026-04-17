"""Public façade for section YAML load + solve (consistent verb vocabulary)."""

from __future__ import annotations

from pathlib import Path

from ..engine.geometry import SectionDefinition
from ..engine.implicit_section_geometry import GeometryConstraintSpec, build_section_from_constraints
from ..engine.solver import MidsurfaceSectionSolver
from ..core.types import SectionSolveResult, SectionSolverProtocol
from ..io.yaml_loader import load_section_from_yaml


class SectionAnalysis:
    """One entry object for ``load`` → ``solve`` on midsurface sections."""

    def __init__(self, solver: SectionSolverProtocol | None = None) -> None:
        self._solver: SectionSolverProtocol = solver or MidsurfaceSectionSolver()

    @staticmethod
    def load(path: str | Path) -> SectionDefinition:
        """Parse YAML into :class:`~section_model.engine.geometry.SectionDefinition`."""
        return load_section_from_yaml(path)

    def solve(self, section: SectionDefinition) -> SectionSolveResult:
        """Run the configured section solver."""
        return self._solver.solve_one(section)

    @staticmethod
    def from_constraints(spec: GeometryConstraintSpec) -> SectionDefinition:
        """Build a constrained implicit-geometry section in S frame."""
        return build_section_from_constraints(spec).section

    def load_and_solve(self, path: str | Path) -> SectionSolveResult:
        """``load`` then ``solve``."""
        return self.solve(self.load(path))

    def solve_constraints(self, spec: GeometryConstraintSpec) -> SectionSolveResult:
        """Build constrained section then solve with configured solver."""
        return self.solve(self.from_constraints(spec))
