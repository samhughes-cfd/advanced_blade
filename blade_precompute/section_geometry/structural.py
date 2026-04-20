"""Structural family labels for ``MultiCellSection`` (SystemType{X}{Y}-{Z}, Y dimension)."""

from __future__ import annotations

from typing import Literal

StructuralFamily = Literal["A", "B", "C", "D"]
FixedCapAnchor = Literal["pitching", "max_thickness"]


def parse_structural_family(value: str | StructuralFamily) -> str:
    s = str(value).strip().upper()
    if s not in ("A", "B", "C", "D"):
        raise ValueError(
            f"structural_family must be one of A|B|C|D, got {value!r}"
        )
    return s


def parse_fixed_cap_anchor(value: str | FixedCapAnchor) -> str:
    v = str(value).strip().lower().replace("-", "_")
    if v == "max thickness":
        v = "max_thickness"
    if v not in ("pitching", "max_thickness"):
        raise ValueError(
            f"fixed_cap_anchor must be 'pitching' or 'max_thickness', got {value!r}"
        )
    return v
