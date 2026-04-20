"""Backward-compatible geometry exports."""

from .primitives import (
    sdf_circle,
    sdf_box,
    sdf_half_plane,
    sdf_segment,
    sdf_capsule,
    sdf_rounded_box,
    sdf_ellipse,
    sdf_polygon,
    sdf_oriented_box,
)
from .csg import (
    union,
    smooth_union,
    intersect,
    smooth_intersect,
    subtract,
    smooth_subtract,
    offset,
    shell,
    blend,
)
from .airfoil import AirfoilSDF
from .section_axes import max_thickness_chord_x, pitch_axis_x_from_le
from .transforms import rotate_field, translate_field, scale_field, SDFFrame
from .grid import SDFGrid

__all__ = [
    "sdf_circle", "sdf_box", "sdf_half_plane", "sdf_segment",
    "sdf_capsule", "sdf_rounded_box", "sdf_ellipse", "sdf_polygon",
    "sdf_oriented_box",
    "union", "smooth_union", "intersect", "smooth_intersect",
    "subtract", "smooth_subtract", "offset", "shell", "blend",
    "AirfoilSDF", "SDFGrid",
    "pitch_axis_x_from_le", "max_thickness_chord_x",
    "rotate_field", "translate_field", "scale_field", "SDFFrame",
]
