# section_geometry Efficiency Refactor Plan

## Objective

Implement computational-efficiency improvements for `blade_precompute/section_geometry` and its orchestration usage with a staged, low-regression approach.

The plan is sequenced so each wave can ship independently and provide measurable gains before moving to higher-risk architecture changes.

## Success Criteria

- Reduce total wall-clock time for `section_geometry_impl` on multi-station runs.
- Reduce duplicate `grid.eval(component)` calls per station.
- Preserve numerical behavior for section properties and exported geometry within established tolerances.
- Maintain existing public APIs (`MultiCellSection`, `BladeSectionGeometry`, plotting/export interfaces) unless explicitly versioned.

## Dependency Order

- Wave 1 is foundational and should merge first.
- Wave 2 depends on Wave 1 utilities and baselines.
- Wave 3 depends on Wave 2 caching abstractions.
- Wave 4 depends on Wave 1 baseline/perf instrumentation and can proceed in parallel with late Wave 2/3 if interfaces are stable.

## Wave 1 - Quick Wins (Low Risk, Days)

### W1.1 Vectorize `sdf_polygon`

- Target files:
  - `blade_precompute/section_geometry/geometry/primitives.py`
  - `blade_precompute/section_geometry/tests/test_airfoil.py`
  - `blade_precompute/section_geometry/tests/test_primitives.py`
- Change:
  - Replace edge-wise Python loop in `sdf_polygon` with edge-axis vectorized evaluation and reduction.
  - Keep exact function signature and sign convention.
- Expected speed-up class: High.
- Risk class: Medium (numeric corner cases around winding and edge degeneracy).
- Verification:
  - Existing geometry tests pass.
  - Add deterministic regression checks for known polygons and point clouds.

### W1.2 Fuse section property computation to one pass

- Target files:
  - `blade_precompute/section_geometry/geometry/grid.py`
  - `blade_precompute/section_geometry/interface/export.py`
- Change:
  - Introduce `section_properties_fused(phi)` to compute `area`, `cx`, `cy`, `Ixx`, `Iyy`, `Ixy`, `r_gyr_x`, `r_gyr_y` in a single mask pass.
  - Keep legacy methods as wrappers for backward compatibility.
- Expected speed-up class: Medium-High.
- Risk class: Low.
- Verification:
  - Compare fused vs legacy values on representative shapes with strict tolerances.

### W1.3 Replace matplotlib figure-based contour extraction

- Target files:
  - `blade_precompute/section_geometry/geometry/grid.py`
  - `blade_precompute/section_geometry/interface/export.py`
  - `blade_precompute/section_geometry/tests/test_csg.py` (or new contour extraction tests)
- Change:
  - Replace `plt.subplots()` + `ax.contour(...)` path in `zero_contour` / `level_set` with `contourpy.contour_generator(...)`.
  - Preserve list-of-polyline output schema.
- Expected speed-up class: Medium.
- Risk class: Low-Medium (dependency/format differences).
- Verification:
  - Compare contour segment counts and representative coordinates against baseline on stable inputs.

### W1.4 Memoize spanwise airfoil generation

- Target files:
  - `blade_precompute/section_geometry/geometry/naca_parametric.py`
  - `blade_precompute/orchestration/precompute/stages.py` (if wrapper keys need normalization)
- Change:
  - Add bounded `functools.lru_cache` for deterministic scalar-keyed airfoil vertex generation.
- Expected speed-up class: Low-Medium.
- Risk class: Low.
- Verification:
  - Ensure cache key normalization avoids float noise.
  - Validate repeated calls return consistent array values (copy outputs where needed).

### W1.5 Add array-conversion fast paths

- Target files:
  - `blade_precompute/section_geometry/geometry/primitives.py`
  - `blade_precompute/section_geometry/geometry/transforms.py`
  - `blade_precompute/section_geometry/sections/subcomponents.py`
- Change:
  - Add `_ensure_f64` helper and avoid redundant conversion work for already-correct ndarray inputs.
- Expected speed-up class: Low-Medium.
- Risk class: Low.
- Verification:
  - Unit tests + microbenchmarks around representative primitive calls.

## Wave 2 - Shared Evaluation Cache and Coordination (Medium Risk)

### W2.1 Add per-station `SectionEvalCache`

- Target files:
  - `blade_precompute/section_geometry/engine/eval_cache.py` (new)
  - `blade_precompute/section_geometry/engine/implicit_section_geometry/pipeline.py`
- Change:
  - Create a cache object storing precomputed `phi` by `(grid_identity, label)` and optionally by callable/object identity.
  - Provide a minimal API: `get_or_eval(label, sdf_callable, grid)`.
- Expected speed-up class: High.
- Risk class: Low-Medium.
- Verification:
  - Unit tests proving cache hit/miss behavior and shape consistency.

### W2.2 Thread cache through all consumers

- Target files:
  - `blade_precompute/section_geometry/interface/export.py`
  - `blade_precompute/section_geometry/interface/plot.py`
  - `blade_precompute/section_geometry/medial/extractor.py`
  - `blade_precompute/section_geometry/engine/implicit_section_geometry/pipeline.py`
- Change:
  - Add optional `eval_cache` arguments to properties, plotting, medial extraction, and JSON export paths.
  - On first consumer pass, populate cache; all later consumers reuse arrays.
- Expected speed-up class: High.
- Risk class: Medium (API threading and optional-arg behavior).
- Verification:
  - Backward compatibility tests where cache is omitted.
  - Instrument call counters to confirm reduced `grid.eval` calls.

### W2.3 Move twist to grid-query transform path

- Target files:
  - `blade_precompute/section_geometry/sections/multicell.py`
  - `blade_precompute/section_geometry/geometry/grid.py`
  - `blade_precompute/section_geometry/geometry/transforms.py`
- Change:
  - Avoid wrapping each component with `rotate_field(...)` when a fixed twist and fixed grid are used.
  - Precompute rotated query coordinates once per station/grid.
- Expected speed-up class: Medium.
- Risk class: Medium (frame-consistency correctness).
- Verification:
  - Compare component contours pre/post change for twisted sections.

### W2.4 Improve web-anchor intercept computation

- Target files:
  - `blade_precompute/section_geometry/sections/multicell.py`
  - `blade_precompute/section_geometry/tests/test_multicell.py`
- Change:
  - Replace or augment `_inner_y_at_x` with deterministic geometric interception logic (or cache-enabled faster scan fallback).
- Expected speed-up class: Low-Medium.
- Risk class: Medium.
- Verification:
  - Explicit anchor regression cases near LE/TE and high camber profiles.

## Wave 3 - Compiled CSG Expression Graph (Higher Risk, Highest Leverage)

### W3.1 Introduce CSG expression IR

- Target files:
  - `blade_precompute/section_geometry/engine/csg_ir.py` (new)
  - `blade_precompute/section_geometry/sections/subcomponents.py`
  - `blade_precompute/section_geometry/sections/multicell.py`
- Change:
  - Define immutable expression nodes for primitives and CSG ops.
  - Refactor subcomponent constructors to build expression trees instead of deeply nested closures.
- Expected speed-up class: High.
- Risk class: High.
- Verification:
  - Golden tests comparing expression-evaluated fields to existing closure fields.

### W3.2 Compile expressions with common-subexpression elimination

- Target files:
  - `blade_precompute/section_geometry/engine/csg_ir.py`
  - `blade_precompute/section_geometry/engine/eval_cache.py`
- Change:
  - Compile expression graph on a given grid, hash-cons subtree nodes, evaluate unique leaves once, fold operations with ndarray operations.
- Expected speed-up class: High.
- Risk class: High.
- Verification:
  - Benchmarks showing fewer leaf evaluations.
  - Numerical equivalence checks for section properties and contours.

### W3.3 Add compatibility shim for callable consumer APIs

- Target files:
  - `blade_precompute/section_geometry/sections/multicell.py`
  - `blade_precompute/section_geometry/sections/section.py`
  - `blade_precompute/section_geometry/__init__.py`
- Change:
  - Keep `__getitem__` returning callable-compatible objects while internals route through compiled cached arrays.
- Expected speed-up class: Medium-High (enables Wave 3 adoption without caller rewrites).
- Risk class: Medium-High.
- Verification:
  - Existing tests and orchestration pathways run unchanged.

## Wave 4 - Orchestration Parallelism and Adaptive Grid (High ROI, Operational Risk)

### W4.1 Process-parallel station execution

- Target files:
  - `blade_precompute/orchestration/precompute/stages.py`
  - `blade_precompute/orchestration/precompute/containers.py` (if worker config surfaces are added)
  - `tests/test_main_precompute_grid_controls.py` (or orchestration stage tests)
- Change:
  - Convert serial station loop in `section_geometry_impl` to process pool fan-out/fan-in.
  - Preserve deterministic ordering in summary payload.
- Expected speed-up class: High.
- Risk class: Medium (serialization/logging/plot backend lifecycle).
- Verification:
  - Multi-platform smoke test on representative station counts.
  - Confirm run logs and artifact paths remain stable.

### W4.2 Split property and plot grid resolutions

- Target files:
  - `blade_precompute/orchestration/precompute/stages.py`
  - `blade_precompute/orchestration/precompute/containers.py` (if manifest schema changes)
  - `tests/test_main_precompute_grid_controls.py`
- Change:
  - Introduce separate config for `properties_grid` and `plot_grid`.
  - Compute properties on coarser grid, reserve fine grid for rendering/export detail where required.
- Expected speed-up class: High.
- Risk class: Medium (accuracy/performance trade-off governance).
- Verification:
  - Define and enforce acceptable property error thresholds vs fine-grid baseline.

## Cross-Wave Validation and Guardrails

## Accuracy Guardrails

- Section property tolerance gates for area/centroid/moments against high-resolution reference.
- Contour topology checks (segment count and enclosure sanity) for representative components.
- Twisted-frame regression checks for contour alignment and orientation.

## Performance Guardrails

- Add perf harness (`tests/perf/test_section_geometry_perf.py`) for:
  - single station timings,
  - multi-station timings,
  - `grid.eval` call counts by label.
- Track baseline and post-wave deltas in runtime manifests where available.

## Rollout Strategy

- Ship one wave per PR where practical.
- Require correctness tests + perf deltas in each PR summary.
- Use feature flags for Wave 3 compiler path until equivalence confidence is established.

## Risk Register

- High numerical-risk areas:
  - polygon winding sign behavior after vectorization,
  - twist/frame transforms when refactoring evaluation coordinates,
  - expression graph equivalence to closure semantics.
- High operational-risk areas:
  - process-parallel orchestration interactions with plotting and run logging.

## Out of Scope

- Fundamental topology/API redesign of `MultiCellSection`.
- Renderer replacement for artifact visualization beyond contour extraction internals.
- GPU backend migration (CuPy/JAX) in this phase.
