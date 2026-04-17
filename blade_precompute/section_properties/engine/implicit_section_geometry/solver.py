from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from blade_precompute.section_properties.core.types import SectionSolveResult, SectionSolverProtocol
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver

from .pipeline import build_section_from_constraints
from .types import GeometryConstraintSpec


@dataclass
class SDFSectionSolver(SectionSolverProtocol):
    """
    Protocol-compatible solver scaffold for implicit geometry sections.

    Phase 1/2 behavior: construct constrained midsurface geometry and delegate to
    MidsurfaceSectionSolver for K6/K7/warping recovery.
    """

    midsurface_solver: MidsurfaceSectionSolver = MidsurfaceSectionSolver()

    def solve_one(self, section_def: object) -> SectionSolveResult:
        if not isinstance(section_def, GeometryConstraintSpec):
            raise TypeError("SDFSectionSolver expects GeometryConstraintSpec input.")
        built = build_section_from_constraints(section_def)
        return self.midsurface_solver.solve_one(built.section)

    def solve(self, section_defs: Iterable[object]) -> list[SectionSolveResult]:
        return [self.solve_one(s) for s in section_defs]

