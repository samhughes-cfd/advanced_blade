# SystemType{X}{Y}-{Z} — structural taxonomy

## Pattern

| Symbol | Meaning | Typical values |
|--------|---------|----------------|
| **X** | Number of shear webs | 0, 1, 2, … |
| **Y** | Spar-cap / laminate family | **A**, **B**, **C**, **D** (see below) |
| **Z** | Web orientation in the section | **CN** = chord-normal, **F** = flapwise *(omit when X = 0)* |

Display name: `SystemType{X}{Y}-{Z}` (e.g. `SystemType3D-CN` for three webs, continuous box caps, chord-normal).

## Y — spar-cap families (extended)

| Y | Description |
|---|-------------|
| **A** | No spar caps: skin, webs, cores (and optional inserts) only. |
| **B** | **Single** fixed upper/lower cap pair on **one** chordwise band. Anchor options (see §Axes): **pitching** (default fraction of chord from LE, typically **⅓**) or **max_thickness** (chord station from airfoil geometry). |
| **C** | **Multiple** discrete upper/lower cap patches — one band per web station — merged for CSG as two solids (`spar_cap_upper` / `spar_cap_lower`). |
| **D** | **Continuous box** spar: one upper and one lower laminate **spanning from the leftmost to the rightmost web** (requires **≥ 2** webs for a meaningful “box” between outer webs; **N = 1** is not registered in precompute). |

### Relation to older three-letter Y

Older docs used **Y = A/B/C** where **C** meant “box spar”. The extended scheme inserts **B** and **C** as discrete/fixed patterns and renames continuous box to **D** to avoid overloading **C**.

## Z — web orientation

- **CN**: Web mid-surface runs chord-normal in the chord (S) frame before section twist.
- **F**: Web mid-surface is vertical in the blade (B) frame (global flapwise direction). The chord-frame `ShearWeb` capsule uses **tilted S-frame endpoints** chosen so that, after the section’s global `+twist` about the origin, the axis is B-frame vertical and meets the **same** rotated inner-skin clip as the skin and caps (see `MultiCellSection` flapwise branch — not a `ShearWeb` counter-rotation).

Spar-cap clipping follows the same **chord-vertical** vs **parallel strip** logic as webs (`MultiCellSection`).

## Reference axes and terminology

These are **not** interchangeable.

| Axis / locus | Role |
|--------------|------|
| **Pitching axis** (process / load) | Chordwise line used for **stacking**, **twist reference**, **centre of pressure** / extreme load line in many workflows. Often taken at a **configurable fraction of chord from the LE** (default **⅓·c**). **Not** the elastic axis or shear centre. |
| **Max-thickness locus** | Chord station where **local thickness** (or t/c) is maximal for the section — used as an optional anchor for **Y = B**. The implementation uses `AirfoilSDF.thickness_distribution` (sampled max); a future refinement may use a three-point colinearity rule. |
| **Elastic / centroid axis** | From **stiffness** / **area** distribution — **not** used for cap placement in v1. |
| **Shear centre** | From shear/torsion theory — **not** used for cap placement in v1. |
| **Web-local reference** | For **Y = C**, caps are centred on each **web chord station** (same anchors as shear webs). A future refinement could use a laminate centroid per web. |

## Code mapping (`MultiCellSection`)

| `structural_family` | Behaviour |
|---------------------|-----------|
| `"D"` *(default)* | Continuous spar caps between first and last web; flapwise strip clipping when all webs are flapwise. |
| `"A"` | No `spar_cap_upper` / `spar_cap_lower`; cores exclude webs (and inserts) only. |
| `"B"` | One `SparCap` band per surface; anchor from `fixed_cap_anchor` (`pitching` \| `max_thickness`). |
| `"C"` | Union of per-web `SparCap` bands; `discrete_cap_chord_half_width` controls band width. |

### Kwargs (subset)

- `structural_family`: `"A"` \| `"B"` \| `"C"` \| `"D"`
- `fixed_cap_anchor`: `"pitching"` \| `"max_thickness"` (for **B**)
- `pitch_fraction_of_chord_from_le`: default `1/3`
- `fixed_cap_chord_half_width`: half-width of **B** band along chord; default `0.05 * chord` if omitted
- `discrete_cap_chord_half_width`: half-width per web for **C**; default `0.04 * chord` if omitted

## Precompute layout registry (`resolve_system_type`)

Blade precompute uses compact tokens as keys in `blade_precompute.orchestration.system_layout._LAYOUT_REGISTRY` (see `resolve_system_type`). The **middle letter of the key is taxonomy Y** and matches `structural_family` on the spec.

- **0-web:** only **`0A`** and **`0B`** are valid keys (**X = 0** ⇒ no **Z** suffix). **`0A`** = **Y = A** (no spar caps; skin-only structural intent). **`0B`** = **Y = B** (one upper/lower cap band per surface; anchor typically **max_thickness** or **pitching** — still “single band”, not web-junction discrete caps). Geometry mode: `airfoil_sdf_only` (not a `MultiCellSection` with webs).
- **No `0C` / `0D`:** **Y = C** places discrete cap bands at **each web chord station** (I-beam style with webs). **Y = D** is a continuous box **between outer webs**. Both require **X ≥ 1**, so **no C- or D-type family exists when X = 0**.
- **Multicell:** for **X = 1..5**, **Y = A, B, C, D**, **Z = CN, F** — **all combinations are registered except `1D-CN` and `1D-F`**. (Single-web continuous box is excluded; use **A** or **B** with one web instead.)
- That yields **38** multicell keys plus **0A** / **0B** (40 `SYSTEM_TYPE_KEYS` total). Higher web counts (4, 5) use shared defaults for web chord fractions; see `system_layout` for the exact dicts.

## Shell FE handoff

The MITC4 shell stage consumes **B-frame polylines** from
`build_shell_midline_strips` / `ShellMeshInputs`; it does **not** re-evaluate
the raw SDF (``section[label](x, y)``) for strip discretisation.

Key points:

- `build_shell_midline_strips` (in `section_geometry.interface.shell_midline_export`)
  calls `midline_polyline()` on each subcomponent in its chord (S) frame, then
  rotates into the B-frame via `rotate_chord_to_blade`.
- `build_shell_mesh_inputs` wraps the strips into a `ShellMeshInputs` payload
  and appends LE/TE context.  Once this returns, SDF evaluation should cease for
  that section station.
- `topology_v2.build_section_v2` emits panels in **skin → caps → webs** order;
  do not pair `panels[i]` directly with `midlines[i]` (they differ for any
  layout that has both caps and webs).
- **`airfoil_sdf_only` layouts (`0A`, `0B`)** do not expose
  `_components_unrotated` and therefore cannot be passed to
  `build_shell_mesh_inputs`.  Attempting to do so raises a descriptive
  `ValueError` distinguishing this case from a broken multicell object.

## Out of scope (v1)

- Cap placement driven by **shear-centre** or **elastic-axis** coordinates (requires composite data or external inputs).
- **X = 0** is **not** built as `MultiCellSection` (that path still assumes ≥ 1 web); **`0A`/`0B`** use **`airfoil_sdf_only`** / skin-outer section types — see orchestration `build_section_view` and shell PR3 meshing for strips + MITC4.
- Refined max-thickness definition beyond sampled `thickness_distribution`.
