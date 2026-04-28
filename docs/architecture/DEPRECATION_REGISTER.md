# blade_precompute Deprecation Register

This register tracks symbols and modules that are superseded, low-use, or planned for removal.
Current policy: deprecate with one-cycle compatibility where practical; remove immediately when inbound usage is zero.

## Status Legend

- `active`: currently supported.
- `deprecated`: retained for compatibility, replacement exists.
- `candidate`: likely removable, pending confirmation.

## Entries

| Item | Status | Rationale | Replacement | Target |
|---|---|---|---|---|
| `section_optimisation/core/failure.py` | removed | Deprecated re-export shim had zero inbound imports. | `blade_precompute.section_properties.engine.failure_criteria` | completed |
| `orchestration/precompute/stages.py::naca4` | removed | Unused helper in orchestration stage module (zero inbound calls). | use `spanwise_airfoil_label` path | completed |
| `section_geometry/io/export.py` re-export layer | removed | Alias surface retired; one internal importer migrated. | `section_geometry.interface.export` | completed |
| `section_geometry/viz/plot.py` re-export layer | removed | Alias surface retired; canonical plot import is `section_geometry.vis`. | `section_geometry.vis` | completed |
| `global_beam_model/interface/plot.py` path | deprecated | Migrated to canonical `vis.py`. | `blade_precompute.global_beam_model.vis` | one cycle |
| `section_optimisation/interface/plot.py` path | deprecated | Migrated to canonical `vis.py`. | `blade_precompute.section_optimisation.vis` | one cycle |
| `section_shell_model/lib/example_plots.py` path | deprecated | Migrated to package-level `vis.py`. | `blade_precompute.section_shell_model.vis` | one cycle |
| `section_shell_model` `global_bc_mode=\"legacy\"` | removed | Legacy alias had zero call sites; support path deleted. | `global_bc_mode=\"full_clamp\"` | completed |
| `OptimizationObjective` type alias name | removed | American spelling alias had zero call sites. | `OptimisationObjective` | completed |
| `orchestration/precompute/stage_inputs.py` | removed | Re-export wrapper had zero inbound imports. | `orchestration/precompute/containers.py` | completed |
| `orchestration/precompute/stage_outputs.py` | removed | Re-export wrapper had zero inbound imports. | `orchestration/precompute/containers.py` | completed |
| `orchestration/precompute/stage_params.py` | removed | Re-export wrapper had zero inbound imports. | `orchestration/precompute/containers.py` | completed |
| `orchestration/precompute/stage_manifests.py` | removed | Re-export wrapper had zero inbound imports. | `orchestration/precompute/containers.py` | completed |
| `orchestration/precompute/stages_section_*.py` wrappers | removed | Wrapper split was unwound; stage facade now imports `stages.py` directly. | `orchestration/precompute/stages.py` | completed |
| `orchestration/precompute/jsonutil.py` | removed | Utility relocated to shared package to remove compute→orchestration dependency. | `blade_precompute._utils.jsonutil` | completed |
| `section_shell_model` import from `orchestration.precompute.jsonutil` | removed | Violated package dependency rule (compute importing orchestration). | `blade_precompute._utils.jsonutil` | completed |

## Process

When marking an item deprecated:

1. Keep backward-compatible import/alias for one cycle when inbound usage is non-zero.
2. Emit a warning only on public entry points (avoid hot loops).
3. Add migration note in docstrings and this register.
4. Remove after one release cycle with no unresolved references.
