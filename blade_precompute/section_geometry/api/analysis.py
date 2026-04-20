"""Public facade for section geometry workflows."""

from __future__ import annotations

from ..engine.implicit_section_geometry import (
    AirfoilSDF,
    MedialAxisExtractor,
    MultiCellSection,
    SDFGrid,
)
from ..io import SectionPropertiesReport, export_midlines_csv, export_section_json


class SectionGeometryAnalysis:
    """High-level facade mirroring section_properties.SectionAnalysis style."""

    def build_airfoil(self, code: str, chord: float = 1.0) -> AirfoilSDF:
        return AirfoilSDF.from_naca(code, chord=chord)

    def build_section(self, airfoil_sdf: AirfoilSDF, **kwargs) -> MultiCellSection:
        return MultiCellSection(airfoil_sdf=airfoil_sdf, **kwargs)

    def build_grid(self, airfoil_sdf: AirfoilSDF, **kwargs) -> SDFGrid:
        return SDFGrid.from_airfoil(airfoil_sdf, **kwargs)

    def section_properties(self, section_geometry, grid: SDFGrid) -> SectionPropertiesReport:
        return SectionPropertiesReport(section_geometry, grid)

    def extract_midlines(self, section_geometry, grid: SDFGrid, **kwargs):
        return MedialAxisExtractor(grid, **kwargs).extract_for_section(section_geometry)

    def export_midlines_csv(self, midline_dict, filepath: str) -> str:
        return export_midlines_csv(midline_dict, filepath)

    def export_section_json(self, section_geometry, grid: SDFGrid, midline_dict, filepath: str, **kwargs) -> str:
        return export_section_json(section_geometry, grid, midline_dict, filepath, **kwargs)
