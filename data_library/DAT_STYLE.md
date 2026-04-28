# `data_library/` `.dat` style spec (v1)

All tabulated `.dat` files in this folder follow a single style so they are easy to
read by hand and easy to validate by machine. This document is the authoritative
reference; updates to any `.dat` file or its loader must keep both in sync.

## Skeleton

```
# <File title>
# Source     : <generator script or upstream dataset>
# Convention : SI units (m, kg, s, N, Pa, deg). Dimensionless = "-".
# Datum      : <e.g. z = 0 at first tabulated station>
#
# Schema (in column order):
#   <col_name>   [<unit>]   <one-line description>
#   ...
#
# units: <u1>, <u2>, ..., <uN>          <-- machine-parseable; comma-separated; one per column
   <col1>      <col2>      ...      <colN>          <-- whitespace-separated header row
   <row 0 values...>
   <row 1 values...>
```

## Rules

- Every line that starts with `#` is a comment. Blank `#` lines (`#`) are allowed
  for visual breaks. The plain blank line is also allowed.
- Exactly one machine-parseable `# units:` line is required immediately above the
  column-name header row. It is comma-separated; one entry per column.
- The header row is whitespace-separated column names. **Column count must equal
  unit count.** Loaders raise `ValueError` if they disagree.
- Multi-section files (e.g. `material_library.dat`) repeat the block per section,
  with a `[section_name]` marker preceding each block:

  ```
  [section_name]
  # units: <u1>, ..., <uN>
     <col1>  ...  <colN>
     <values...>
  ```

  Inside one section the same rules apply.

## Unit grammar (ASCII only)

- Base units: `m`, `kg`, `s`, `N`, `Pa`, `deg`, `rad`.
- Combination operators (no whitespace required between operands and operators):
  - `*` multiplication
  - `/` division
  - `^` power (e.g. `kg/m^3`)
- Examples: `N/m`, `N*m/m`, `kg/m^3`, `1/m`, `Pa`.
- Dimensionless quantities, identifiers, and free-text names use `-`.
- ASCII only so a simple `split(",")` plus `strip()` is sufficient. Unicode
  variants (e.g. `\u00b7` for `*`) are tolerated by the canonicaliser but not
  encouraged in source files.

## Canonicaliser

`data_library.plot_inputs._canonicalise_unit(s)` performs:

1. `s.strip().lower()`
2. replace Unicode `\u00b7` (`MIDDLE DOT`) and ` ` (whitespace) with `*`
3. replace `**` with `^`
4. strip trailing `^1`
5. collapse repeated `*` and `/` runs

so `"N*m/m" == "N\u00b7m/m" == "n m / m"` after canonicalisation.

## Files conforming to this spec

| File                                  | Sections                                           | Columns |
| ------------------------------------- | -------------------------------------------------- | ------- |
| `blade_spanwise_distribution.dat`     | (single)                                           | 13      |
| `extreme_load_distribution.dat`       | (single)                                           | 5       |
| `operational_load_timeseries.dat`     | (single, long format)                              | 6       |
| `material_library.dat`                | `[orthotropic_laminate_ply]`, `[isotropic]`        | 16, 6   |

## Loaders

- Generic columnar reader (back-compat, no unit info):
  `data_library.plot_inputs.read_columnar_dat(path) -> (names, data)`
- Generic columnar reader with units (preferred for new code):
  `data_library.plot_inputs.read_columnar_dat_with_units(path) -> (names, units, data)`
- Material library loader:
  `blade_precompute.orchestration.precompute.material_library.load_material_library_dat(path)`

The material-library loader and the precompute `load_inputs` both validate units
against a per-column expected map and raise `ValueError` listing every offending
column when the file's unit row disagrees.
