"""Backward-compatible section builder exports."""

from .subcomponents import (
    OuterSkin,
    SparCap,
    ContinuousSparCap,
    ShearWeb,
    SandwichCore,
    TEInsert,
    LEInsert,
)
from .multicell import MultiCellSection
from .section import BladeSectionGeometry

__all__ = [
    "OuterSkin", "SparCap", "ContinuousSparCap",
    "ShearWeb", "SandwichCore", "TEInsert", "LEInsert",
    "MultiCellSection", "BladeSectionGeometry",
]
