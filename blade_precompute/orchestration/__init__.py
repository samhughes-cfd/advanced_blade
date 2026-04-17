"""Precompute orchestration helpers (system layout, material map, export DTOs)."""

from .precompute_context import PrecomputeOrchestrationContext
from .component_materials import (
    COMPONENT_MATERIAL_KEYS,
    ComponentMaterialsMap,
    load_component_materials_json,
    ply_library_material_table,
    validate_component_indices,
)
from .midline_export_semantics import MIDLINE_CONTRACT_VERSION, midline_series_contract_doc
from .section_boundary_export_dto import (
    SectionBoundaryExportV1,
    section_boundary_stub_from_labels,
)
from .system_layout import (
    SYSTEM_TYPE_KEYS,
    SystemLayoutSpec,
    build_section_view,
    resolve_system_type,
)
from .sdf_centreline_compat import (
    SectionPhiCallable,
    assert_grid_phi_finite,
    describe_sdf_for_centreline,
)

__all__ = [
    "PrecomputeOrchestrationContext",
    "COMPONENT_MATERIAL_KEYS",
    "ComponentMaterialsMap",
    "MIDLINE_CONTRACT_VERSION",
    "SectionBoundaryExportV1",
    "SYSTEM_TYPE_KEYS",
    "build_section_view",
    "SectionPhiCallable",
    "SystemLayoutSpec",
    "assert_grid_phi_finite",
    "describe_sdf_for_centreline",
    "load_component_materials_json",
    "midline_series_contract_doc",
    "ply_library_material_table",
    "resolve_system_type",
    "section_boundary_stub_from_labels",
    "validate_component_indices",
]
