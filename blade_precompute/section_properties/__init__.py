"""
Midsurface cross-section model (Saint-Venant warping + Vlasov-style ``K7``,
CLPT / isotropic recovery).

Theory tag: **midsurface-v1** — see :mod:`blade_precompute.section_properties.engine.solver`.

Run: ``python -m blade_precompute.section_properties``.
"""

from __future__ import annotations

from .api import AnalysisConfig, SectionAnalysis
from .core.types import MaterialProps, SectionProps, SectionSolveResult, SectionSolverProtocol
from .engine.clpt_recovery import clpt_ply_stresses_section_frame, rotate_plies_to_material
from .engine.failure_criteria import hashin_fi, hashin_fi_plies, von_mises_plane_stress_fi
from .engine.geometry import MaterialAssignment, SectionDefinition, SubcomponentGeometry
from .engine.interlaminar_recovery import (
    build_interlaminar_operators,
    interlaminar_stress_recovery,
    recover_interlaminar,
)
from .engine.panel_buckling import (
    PanelBucklingResult,
    PanelBucklingSectionResult,
    SectionBucklingResult,
    assess_panel_buckling_section,
    composite_edge_panel_stresses_from_reference,
)
from .engine.strip_shear_equilibrium import (
    StripShearFlowSummary,
    compute_strip_shear_flow_summary,
    recover_interlaminar_strip_equilibrium,
)
from .engine.isotropic_recovery import isotropic_membrane_stresses, von_mises_plane_stress
from .engine.laminate import LaminateDefinition, tsai_wu_fi, tsai_wu_polynomial
from .engine.materials import IsotropicMaterial, OrthotropicPly, plane_stress_Q, plane_stress_Q_isotropic
from .engine.section_properties import print_section_summary
from .engine.solver import MidsurfaceSectionSolver
from .engine.implicit_section_geometry import GeometryConstraintSpec, SDFSectionSolver, build_section_from_constraints
from .io.external_results import ExternalSectionResultSolver, section_result_from_mapping
from .io.section_loader import load_section_from_spec


def solve_section(section: SectionDefinition) -> SectionSolveResult:
    """One-shot solve using :class:`MidsurfaceSectionSolver`."""
    return MidsurfaceSectionSolver().solve_one(section)


def section_props_from_solve(res: SectionSolveResult) -> SectionProps:
    """Map :class:`SectionSolveResult` into legacy :class:`SectionProps` for beam tools."""
    import numpy as np

    K6 = res.K6
    return SectionProps(
        K6=K6,
        M6=res.M6,
        elastic_center=res.elastic_center,
        mass_center=res.mass_center,
        shear_center=res.shear_center,
        EA=float(K6[0, 0]),
        EIy=float(K6[1, 1]),
        EIz=float(K6[2, 2]),
        EIyz=float(-K6[1, 2]),
        GJ=float(K6[3, 3]),
        mu=float(res.mass_per_length),
        K7=res.K7,
    )


__all__ = [
    "SectionAnalysis",
    "AnalysisConfig",
    "MaterialProps",
    "MaterialAssignment",
    "SubcomponentGeometry",
    "SectionDefinition",
    "LaminateDefinition",
    "IsotropicMaterial",
    "OrthotropicPly",
    "plane_stress_Q",
    "plane_stress_Q_isotropic",
    "tsai_wu_fi",
    "tsai_wu_polynomial",
    "hashin_fi",
    "hashin_fi_plies",
    "von_mises_plane_stress_fi",
    "MidsurfaceSectionSolver",
    "SDFSectionSolver",
    "GeometryConstraintSpec",
    "build_section_from_constraints",
    "SectionSolveResult",
    "SectionSolverProtocol",
    "SectionProps",
    "solve_section",
    "section_props_from_solve",
    "load_section_from_spec",
    "section_result_from_mapping",
    "ExternalSectionResultSolver",
    "print_section_summary",
    "clpt_ply_stresses_section_frame",
    "rotate_plies_to_material",
    "isotropic_membrane_stresses",
    "von_mises_plane_stress",
    "interlaminar_stress_recovery",
    "recover_interlaminar",
    "build_interlaminar_operators",
    "assess_panel_buckling_section",
    "composite_edge_panel_stresses_from_reference",
    "PanelBucklingResult",
    "PanelBucklingSectionResult",
    "SectionBucklingResult",
    "StripShearFlowSummary",
    "compute_strip_shear_flow_summary",
    "recover_interlaminar_strip_equilibrium",
]
