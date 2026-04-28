"""Blade precompute pipeline (stages, inputs, grid helpers).

Stages are run sequentially from the main entrypoint for simplicity and reproducible ``summary.json`` ordering.
Section properties does not depend on section_geometry or section_shell_model outputs; the beam model depends
on section properties. Independent compute could be reordered, but do not use threads to parallelize matplotlib
(``section_geometry`` / ``section_shell_model``) with other stages: matplotlib is not thread-safe in one process.
Process-based fan-out is possible but needs careful pickling and I/O. Parallel midsurface solves use
``ProcessPoolExecutor`` in ``section_optimisation`` / ``section_properties`` and should be sized with
``design_n_workers`` and ``section_solve_n_workers`` to avoid over-subscription when both run in one job.
"""

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    BeamModelParams,
    GridConfig,
    grid_resolution_manifest,
    runtime_statistics_manifest,
    LinspaceSpec,
    PrecomputeInputs,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionOptimisationOutputs,
    SectionOptimisationParams,
    SectionPropertiesOutputs,
    SectionPropertiesParams,
    SectionShellModelOutputs,
    SectionShellModelParams,
)
from blade_precompute.orchestration.precompute.blade_geometry_from_spanwise import (
    build_optim_blade_geometry_from_spanwise,
)
from blade_precompute.orchestration.precompute.grid import (
    interp_series,
    job_span_z_m,
    linspace_from_spec,
    resample_blade_geometry_to_z,
    resample_precompute_inputs,
    station_indices,
    station_subdir_name,
    warn_geometry_shorter_than_job_span,
    warn_job_span_exceeds_geometry,
)
from blade_precompute.orchestration.precompute.inputs import (
    build_precompute_orchestration_context,
    load_inputs,
    resolve_component_materials_path,
)
from blade_precompute.orchestration.precompute.material_library import (
    apply_material_library_to_blade_geometry,
    load_material_library_dat,
    material_resolution_manifest,
    normalize_logical_subcomponent_material_map,
    subcomponent_box_materials_from_csv,
    validate_material_library_bindings,
)
from blade_precompute._utils.jsonutil import to_jsonable, write_json
from blade_precompute.orchestration.precompute.shell_spars import section_shell_spars_from_layout
from blade_precompute.orchestration.precompute.stage_facade import (
    BeamModelStage,
    SectionGeometryStage,
    SectionOptimisationStage,
    SectionPropertiesStage,
    SectionShellModelStage,
)

__all__ = [
    "BeamModelOutputs",
    "BeamModelParams",
    "BeamModelStage",
    "GridConfig",
    "grid_resolution_manifest",
    "runtime_statistics_manifest",
    "LinspaceSpec",
    "PrecomputeInputs",
    "build_optim_blade_geometry_from_spanwise",
    "subcomponent_box_materials_from_csv",
    "SectionGeometryOutputs",
    "SectionGeometryParams",
    "SectionGeometryStage",
    "SectionOptimisationOutputs",
    "SectionOptimisationParams",
    "SectionOptimisationStage",
    "SectionPropertiesOutputs",
    "SectionPropertiesParams",
    "SectionPropertiesStage",
    "SectionShellModelOutputs",
    "SectionShellModelParams",
    "SectionShellModelStage",
    "section_shell_spars_from_layout",
    "build_precompute_orchestration_context",
    "apply_material_library_to_blade_geometry",
    "load_material_library_dat",
    "material_resolution_manifest",
    "normalize_logical_subcomponent_material_map",
    "validate_material_library_bindings",
    "interp_series",
    "job_span_z_m",
    "linspace_from_spec",
    "load_inputs",
    "resample_blade_geometry_to_z",
    "resample_precompute_inputs",
    "resolve_component_materials_path",
    "station_indices",
    "station_subdir_name",
    "warn_geometry_shorter_than_job_span",
    "warn_job_span_exceeds_geometry",
    "to_jsonable",
    "write_json",
]
