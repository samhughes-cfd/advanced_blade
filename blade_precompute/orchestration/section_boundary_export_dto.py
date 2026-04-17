"""Minimal JSON-serialisable DTO mirroring *high-level* ``SectionBoundaries`` fields.

Blade-structure ``section_boundaries`` carries void polygons, per-cell contours,
and deduplicated component boundaries. For SDF-first workflows, dense polygons are
**derived** from ``phi`` (e.g. marching squares). This DTO captures only stable
identifiers and metadata for precompute exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SectionBoundaryExportV1:
    """Versioned stub for cross-tooling with stress / Bredt-Batho consumers."""

    schema_version: int = 1
    cell_indices: list[int] = field(default_factory=list)
    component_labels: list[str] = field(default_factory=list)
    notes: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cell_indices": list(self.cell_indices),
            "component_labels": list(self.component_labels),
            "notes": self.notes,
        }


def section_boundary_stub_from_labels(labels: list[str], *, n_cells_hint: int | None = None) -> SectionBoundaryExportV1:
    cells = list(range(n_cells_hint)) if n_cells_hint is not None else []
    return SectionBoundaryExportV1(
        cell_indices=cells,
        component_labels=list(labels),
        notes="SDF-first export: geometry contours are not populated here; use phi sampling / marching squares when needed.",
    )
