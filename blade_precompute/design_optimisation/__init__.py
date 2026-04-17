"""Stable import path ``blade_precompute.design_optimisation``.

Implementation lives in :mod:`blade_precompute.section_optimisation`.
"""

from blade_precompute.section_optimisation import (
    BeamResultantDriverProtocol,
    BladeDesignProblem,
    BladeOptimizer,
    DesignEvaluation,
    DesignEvaluator,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimBladeGeometry,
    OptimisationResult,
    PrescribedResultantBeamState,
    PrescribedResultantDriver,
    SectionBuilder,
    StationCache,
    beam_solve_k7,
    ks_aggregate,
    load_blade_geometry,
    mass_objective,
    solve_dirty_stations,
)

__all__ = [
    "OptimBladeGeometry",
    "DesignVector",
    "ExtremeLoads",
    "DesignEvaluation",
    "DesignProblem",
    "OptimisationResult",
    "StationCache",
    "SectionBuilder",
    "DesignEvaluator",
    "BladeOptimizer",
    "BladeDesignProblem",
    "BeamResultantDriverProtocol",
    "PrescribedResultantDriver",
    "mass_objective",
    "ks_aggregate",
    "solve_dirty_stations",
    "beam_solve_k7",
    "PrescribedResultantBeamState",
    "load_blade_geometry",
]
