"""
global_beam_model
=================
Geometrically exact 3D Simo–Reissner beam with Vlasov warping (7 DOFs per node),
linear ``K7`` section law, and optional precurvature from :class:`BladeGeometry`.

**Level 2 (out of scope):** section shape distortion under large torsion (>~15°).

Strain six-vector follows :class:`blade_precompute.section_properties.core.types.SectionProps`.
Global resultants are ``[N, Vy, Vz, My, Mz, T, B]``; use
:func:`blade_precompute.global_beam_model.engine.constitutive.resultants_to_recovery6` for the compatible
recovery ordering.

Run: ``python -m blade_precompute.global_beam_model``. Options include ``--verbose``,
``--print-spanwise``, ``--plot``, ``--plot-out path.pdf`` (see
:mod:`blade_precompute.global_beam_model.interface.plot`).
"""

from __future__ import annotations

from .api import BeamAnalysis
from .core import tier_paths
from .core.types import (
    BeamElement,
    BeamLoads,
    BeamModel,
    BeamSolveResult,
    BeamSolverProtocol,
    BoundaryCondition,
    LoadCase,
    NodeState,
    SectionStation,
    SectionStiffnessArray,
    SolverOptions,
    SolveOptions,
    SolveResult,
    default_initial_state,
)
from .engine.blade_geometry import BladeGeometry, beam_model_from_blade_geometry
from .engine.constitutive import resultants_to_recovery6
from .engine.solver import solve_static

__all__ = [
    "tier_paths",
    "BeamAnalysis",
    "solve_static",
    "BladeGeometry",
    "beam_model_from_blade_geometry",
    "resultants_to_recovery6",
    "BeamModel",
    "BeamElement",
    "NodeState",
    "SectionStation",
    "SectionStiffnessArray",
    "BoundaryCondition",
    "BeamLoads",
    "LoadCase",
    "SolverOptions",
    "SolveOptions",
    "BeamSolveResult",
    "SolveResult",
    "BeamSolverProtocol",
    "default_initial_state",
]
