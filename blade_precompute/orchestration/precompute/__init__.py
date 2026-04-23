"""Blade precompute pipeline (stages, inputs, grid helpers)."""

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    BeamModelParams,
    GridConfig,
    LinspaceSpec,
    PrecomputeInputs,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionOptimisationOutputs,
    SectionOptimisationParams,
    SectionPropertiesOutputs,
    SectionPropertiesParams,
)
from blade_precompute.orchestration.precompute.grid import (
    interp_series,
    linspace_from_spec,
    resample_blade_geometry_to_z,
    resample_precompute_inputs,
    station_indices,
)
from blade_precompute.orchestration.precompute.inputs import (
    build_precompute_orchestration_context,
    load_inputs,
    resolve_component_materials_path,
)
from blade_precompute.orchestration.precompute.jsonutil import to_jsonable, write_json
from blade_precompute.orchestration.precompute.stage_facade import (
    BeamModelStage,
    SectionGeometryStage,
    SectionOptimisationStage,
    SectionPropertiesStage,
)

__all__ = [
    "BeamModelOutputs",
    "BeamModelParams",
    "BeamModelStage",
    "GridConfig",
    "LinspaceSpec",
    "PrecomputeInputs",
    "SectionGeometryOutputs",
    "SectionGeometryParams",
    "SectionGeometryStage",
    "SectionOptimisationOutputs",
    "SectionOptimisationParams",
    "SectionOptimisationStage",
    "SectionPropertiesOutputs",
    "SectionPropertiesParams",
    "SectionPropertiesStage",
    "build_precompute_orchestration_context",
    "interp_series",
    "linspace_from_spec",
    "load_inputs",
    "resample_blade_geometry_to_z",
    "resample_precompute_inputs",
    "resolve_component_materials_path",
    "station_indices",
    "to_jsonable",
    "write_json",
]
