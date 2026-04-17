"""Blade structural sizing optimisation (midsurface strip ``K7`` pipeline).

Run: ``python -m blade_precompute.design_optimisation``
"""

from .api import BladeDesignProblem
from .core.protocols import BeamResultantDriverProtocol, PrescribedResultantDriver
from .core.types import (
    DesignEvaluation,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimizationObjective,
    OptimBladeGeometry,
    OptimisationResult,
    StationCache,
)
from .engine.aggregation import ks_aggregate
from .engine.beam_k7 import PrescribedResultantBeamState, solve as beam_solve_k7
from .engine.evaluator import DesignEvaluator
from .engine.mass import mass_objective
from .engine.optimizer import BladeOptimizer
from .engine.parallel import solve_dirty_stations
from .engine.section_builder import SectionBuilder
from .io.yaml_loader import load_blade_geometry

__all__ = [
    "OptimBladeGeometry",
    "DesignVector",
    "ExtremeLoads",
    "DesignEvaluation",
    "DesignProblem",
    "OptimizationObjective",
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
