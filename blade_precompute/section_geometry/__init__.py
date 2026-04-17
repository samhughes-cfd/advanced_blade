"""Public package surface for blade section geometry workflows."""

from .api import SectionGeometryAnalysis
from .core.types import BuildSpec, GridSpec, SDFCallable, SectionGeometryReport
from .engine.implicit_section_geometry import (
    AirfoilSDF,
    BladeSectionGeometry,
    MedialAxisExtractor,
    MultiCellSection,
    SDFGrid,
    extract_midline,
)
from .io import SectionPropertiesReport, export_midlines_csv, export_section_json
from .viz import plot_medial_axes, plot_sdf_field, plot_section

__version__ = "0.2.0"
__author__ = "Sam Hughes"

__all__ = [
    "SectionGeometryAnalysis",
    "SDFCallable",
    "GridSpec",
    "BuildSpec",
    "SectionGeometryReport",
    "AirfoilSDF",
    "SDFGrid",
    "MultiCellSection",
    "BladeSectionGeometry",
    "MedialAxisExtractor",
    "extract_midline",
    "SectionPropertiesReport",
    "export_midlines_csv",
    "export_section_json",
    "plot_section",
    "plot_sdf_field",
    "plot_medial_axes",
]
