"""Midline / contour export semantics (SDF-first; aligns with blade-structure contract *ideas*).

Blade-structure documents arc-length monotonicity and frame labelling in
``MIDLINE_CONTOUR_CONTRACT.md``. This project does **not** re-implement Shapely
junction snapping; when exporting polylines derived from implicit fields:

- **x, y**: section-plane coordinates in the same frame as the SDF grid
  (typically chord-normal section ``y,z`` used elsewhere in ``blade_precompute``).
- **s**: cumulative arc length along an open or closed polyline; must be strictly
  increasing after any branch cut / ``NaN`` jump markers are removed.
- **winding**: closed airfoil polylines follow a single consistent orientation
  (outer skin) so ``phi < 0`` interior matches ``sdf_polygon`` sign convention.

Downstream medial-axis CSV export already lives in
:class:`blade_precompute.section_geometry.interface.export`.
"""

MIDLINE_CONTRACT_VERSION = "sandbox-sdf-1"


def midline_series_contract_doc() -> str:
    return __doc__ or ""
