"""
GBT — Generalised Beam Theory for mixed-material blade sections.

Part of :mod:`blade_precompute.section_beam_model` (cross-section / member-axis
stability), not spanwise :mod:`blade_precompute.global_beam_model`.
"""

from .materials   import IsotropicMaterial, LaminateMaterial, Lamina
from .section     import CrossSection, WallDefinition, SectionNode, WallStrip
from .kinematics  import KirchhoffKinematics, MindlinKinematics
from .prebuckling import PreBucklingAnalysis, SectionLoads
from .modal import (
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    ModalResult,
    classical_export_indices,
    select_modes,
    truncation_report,
)
from .section_stiffness_export import (
    SectionStiffness,
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
    "SectionStiffness",
    "gbt_to_beam_stiffness",
    "gbt_to_k7",
    "section_stiffness_to_k6",
    "section_stiffness_to_station",
    "BoundaryConditions", "EndCondition",
    "MemberBucklingAnalysis", "MemberBucklingResult",
]
