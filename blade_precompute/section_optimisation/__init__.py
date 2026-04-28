"""Blade structural sizing optimisation (midsurface strip ``K7`` pipeline).

Run: ``python -m blade_precompute.section_optimisation``

Single load-case assumption (K.1)
----------------------------------
This module optimises against a **single extreme-load case** (one DLC envelope
``.dat`` file).  Multi-LC envelope sweeps are out of scope.

The following assumptions are persisted in ``section_optimisation/summary.json``
and ``inputs.json`` under ``assumptions`` for provenance and post-run audit:

``assumptions.single_load_case = True``
    Only one ultimate-load file is used per optimisation run.

``assumptions.hydrodynamic_load_invariant_under_sls_tip = True``
    The SLS flapwise tip deflection constraint is evaluated against the same
    extreme-load case; it implicitly assumes the hydrodynamic (wave/current)
    loading envelope does not change with tip deflection under the aero-elastic
    operating state. This is the standard screening assumption for preliminary
    sizing (the operating loads and the extreme loads are both in the same .dat
    file and are not re-derived from the deflected shape).
"""

from .api import BladeDesignProblem
from .core.protocols import BeamResultantDriverProtocol
from .core.types import (
    DesignEvaluation,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimisationObjective,
    OptimBladeGeometry,
    OptimisationResult,
    StationCache,
    apply_dv_to_bg,
)
from .engine.aggregation import ks_aggregate
from .engine.beam_k7 import (
    PrescribedResultantBeamState,
    PrescribedResultantDriver,
    solve as beam_solve_k7,
)
from .engine.evaluator import DesignEvaluator
from .engine.mass import mass_objective
from .engine.optimizer import BladeOptimizer
from .engine.parallel import solve_dirty_stations
from .engine.section_builder import SectionBuilder
from .io.blade_geometry_loader import load_blade_geometry

__all__ = [
    "OptimBladeGeometry",
    "DesignVector",
    "ExtremeLoads",
    "DesignEvaluation",
    "DesignProblem",
    "OptimisationObjective",
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
    "apply_dv_to_bg",
]
