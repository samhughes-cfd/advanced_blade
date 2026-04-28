# blade_precompute Glossary

## Core Terms

- `dv`: design vector (`DesignVector`), structured spanwise design variables.
- `x`: flattened optimisation vector passed to SciPy (`numpy.ndarray`).
- `bg`: blade geometry object (`OptimBladeGeometry` or equivalent).
- `sec`: section properties output passed into beam stage (`SectionPropertiesOutputs`).

## Coordinates and Abscissae

- `z`: spanwise station coordinate along blade length (global blade axis for station indexing).
- `s`: curve abscissa used for interpolation tables or section contour parameterisation.
- Section plane axes are typically `y` and `z` in section-level definitions; beam local axis remains `x`.

## Stiffness Objects

- `K6`: 6x6 generalised section stiffness in the classical resultants/strains basis.
- `K7`: 7x7 section stiffness that extends `K6` with the warping DOF coupling.
- Rule of thumb: use `K6` for classical workflows; use `K7` where warping-aware coupling is required.

## Analysis Tiers

- Tier A: global beam FE solution (`global_beam_model`).
- Tier B: prescribed-resultant design/evaluation workflows (`section_optimisation`).
- Tier C: section recovery/enrichment payloads and caches (`blade_utilities.recovery`, shell enrichment).

## Spanwise Plot Taxonomy (Gauss-derived quantities)

- `Gauss-point evaluation`: direct quantity evaluation at element Gauss points (`z_stations_out`).
- `Shape-function nodal projection`: nodal pull-back from Gauss samples using cached element shape rows (`element_gauss_shape_matrix`) with patch averaging at shared nodes (`z_nodal_out`).
- `Shape-function interpolation`: the continuous plotted field between nodes obtained by piecewise-linear interpolation of nodal projections via the cached 2-node line shape functions (`N1`, `N2`).

## Web Orientation Vocabulary

- Canonical token: `chord_normal`.
- Alternate forms to map at boundaries only: `CN`, `chord_normalwise`.
- Secondary token: `flapwise`.

## Failure Naming Family

- `*_fi`: failure index scalar/tensor fields (Hashin, von Mises).
- `IFI`: interfacial failure index naming used in some interlaminar outputs.
- Project direction: prefer consistent `*_fi` public naming; keep `IFI` as legacy alias where required.
