# SystemType{X}{Y}-{Z} — structural taxonomy

## Pattern

| Symbol | Meaning | Typical values |
|--------|---------|----------------|
| **X** | Number of shear webs | 0, 1, 2, … |
| **Y** | Spar-cap / laminate family | **A**, **B**, **C**, **D** (see below) |
| **Z** | Web orientation in the section | **CN** = chord-normal, **F** = flapwise *(omit when X = 0)* |

Display name: `SystemType{X}{Y}-{Z}` (e.g. `SystemType3C-CN`).

## Y — spar-cap families (extended)

| Y | Description |
|---|-------------|
| **A** | No spar caps: skin, webs, cores (and optional inserts) only. |
| **B** | **Single** fixed upper/lower cap pair on **one** chordwise band. Anchor options (see §Axes): **pitching** (default fraction of chord from LE, typically **⅓**) or **max_thickness** (chord station from airfoil geometry). |
| **C** | **Multiple** discrete upper/lower cap patches — one band per web station — merged for CSG as two solids (`spar_cap_upper` / `spar_cap_lower`). |
| **D** | **Continuous box** spar: one upper and one lower laminate **spanning from the leftmost to the rightmost web** (requires ≥ 1 web in the implementation; degenerates when N = 1). |

### Relation to older three-letter Y

Older docs used **Y = A/B/C** where **C** meant “box spar”. The extended scheme inserts **B** and **C** as discrete/fixed patterns and renames continuous box to **D** to avoid overloading **C**.

## Z — web orientation

- **CN**: Web mid-surface runs chord-normal in the chord (S) frame before section twist.
- **F**: Web runs flapwise in the blade (B) frame; implemented via counter-rotation in `ShearWeb` plus global section twist.

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

## Out of scope (v1)

- Cap placement driven by **shear-centre** or **elastic-axis** coordinates (requires composite data or external inputs).
- **X = 0** skin-only section with no webs (`MultiCellSection` still requires ≥ 1 web).
- Refined max-thickness definition beyond sampled `thickness_distribution`.
