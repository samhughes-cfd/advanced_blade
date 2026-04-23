"""
GBT — Generalised Beam Theory for mixed-material blade sections.

Example package :mod:`section_beam_model` (cross-section / member-axis stability),
not spanwise :mod:`blade_precompute.global_beam_model`.
"""

from blade_precompute.global_beam_model.core.types import SectionStiffness

from .materials   import IsotropicMaterial, LaminateMaterial, Lamina
from .section     import CrossSection, WallDefinition, SectionNode, WallStrip
from .kinematics  import KirchhoffKinematics, MindlinKinematics
from .prebuckling import PreBucklingAnalysis, SectionLoads
from .modal import (
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    ModalResult,
    classical_export_indices,
    export_label_to_coarse_bucket,
    select_modes,
    truncation_report,
    validate_export_classification,
)
from .section_stiffness_export import (
    gbt_to_beam_stiffness,
    gbt_to_k7,
    section_stiffness_to_k6,
    section_stiffness_to_station,
)
from .boundary    import BoundaryConditions, EndCondition
from .member      import MemberBucklingAnalysis, MemberBucklingResult

__all__ = [
    "IsotropicMaterial", "LaminateMaterial", "Lamina",
    "CrossSection", "WallDefinition", "SectionNode", "WallStrip",
    "KirchhoffKinematics", "MindlinKinematics",
    "PreBucklingAnalysis", "SectionLoads",
    "CrossSectionModalAnalysis",
    "ModalResult",
    "DEFAULT_BEAM_EXPORT_MODE_LABELS",
    "select_modes",
    "truncation_report",
    "classical_export_indices",
    "export_label_to_coarse_bucket",
    "validate_export_classification",
    "SectionStiffness",
    "gbt_to_beam_stiffness",
    "gbt_to_k7",
    "section_stiffness_to_k6",
    "section_stiffness_to_station",
    "BoundaryConditions", "EndCondition",
    "MemberBucklingAnalysis", "MemberBucklingResult",
]
