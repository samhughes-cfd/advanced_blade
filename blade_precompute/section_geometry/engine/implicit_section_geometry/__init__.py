"""Implicit section geometry engine namespace."""

from .airfoil import AirfoilSDF
from .builder import BladeSectionGeometry, MultiCellSection
from .components import (
    ContinuousSparCap,
    LEInsert,
    OuterSkin,
    SandwichCore,
    ShearWeb,
    SparCap,
    TEInsert,
)
from .grid import SDFGrid
from .medial import MedialAxisExtractor, extract_midline
from .pipeline import ImplicitSectionBuildResult, build_section_pipeline
from .sdf import (
    blend,
    intersect,
    offset,
    sdf_box,
    sdf_capsule,
    sdf_circle,
    sdf_ellipse,
    sdf_half_plane,
    sdf_oriented_box,
    sdf_polygon,
    sdf_rounded_box,
    sdf_segment,
    shell,
    smooth_intersect,
    smooth_subtract,
    smooth_union,
    subtract,
    union,
)
from .transforms import SDFFrame, rotate_field, scale_field, translate_field
from ...geometry.section_axes import max_thickness_chord_x, pitch_axis_x_from_le
from ...structural import (
    FixedCapAnchor,
    StructuralFamily,
    parse_fixed_cap_anchor,
    parse_structural_family,
)

__all__ = [
    "AirfoilSDF",
    "SDFGrid",
    "OuterSkin",
    "SparCap",
    "ContinuousSparCap",
    "ShearWeb",
    "SandwichCore",
    "TEInsert",
    "LEInsert",
    "MultiCellSection",
    "BladeSectionGeometry",
    "MedialAxisExtractor",
    "extract_midline",
    "ImplicitSectionBuildResult",
    "build_section_pipeline",
    "sdf_circle",
    "sdf_box",
    "sdf_half_plane",
    "sdf_segment",
    "sdf_capsule",
    "sdf_rounded_box",
    "sdf_ellipse",
    "sdf_polygon",
    "sdf_oriented_box",
    "union",
    "smooth_union",
    "intersect",
    "smooth_intersect",
    "subtract",
    "smooth_subtract",
    "offset",
    "shell",
    "blend",
    "rotate_field",
    "translate_field",
    "scale_field",
    "SDFFrame",
    "StructuralFamily",
    "FixedCapAnchor",
    "parse_structural_family",
    "parse_fixed_cap_anchor",
    "pitch_axis_x_from_le",
    "max_thickness_chord_x",
]
