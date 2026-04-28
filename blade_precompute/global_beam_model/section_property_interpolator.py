"""Backward-compatible re-export for section property interpolation."""

from __future__ import annotations

from .engine.section_property_interpolator import (
    SectionPropertyInterpolator,
    section_stiffness_array_from_sequence,
)

__all__ = ["SectionPropertyInterpolator", "section_stiffness_array_from_sequence"]
