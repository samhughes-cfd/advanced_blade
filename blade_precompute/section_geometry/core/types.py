"""Core public types for section geometry workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, runtime_checkable


@runtime_checkable
class SDFCallable(Protocol):
    """Callable signed-distance field interface."""

    def __call__(self, x: Any, y: Any) -> Any:
        ...


@dataclass(frozen=True)
class GridSpec:
    """Structured grid definition used for SDF evaluation."""

    nx: int = 512
    ny: int = 200
    x_min: float = -0.05
    x_max: float = 1.05
    y_min: float = -0.25
    y_max: float = 0.25


@dataclass(frozen=True)
class BuildSpec:
    """Generic build specification for a multi-cell section."""

    web_x_positions: tuple[float, ...]
    web_thickness: float | tuple[float, ...] = 0.004
    web_alignment: str | tuple[str, ...] = "chord_normal"
    cap_height: float | tuple[float, float] = 0.012
    skin_thickness: float = 0.003
    twist_angle: float = 0.0
    # Optional — see docs/system_type_xyz_taxonomy.md
    structural_family: str | None = None
    fixed_cap_anchor: str | None = None
    pitch_fraction_of_chord_from_le: float | None = None
    fixed_cap_chord_half_width: float | None = None
    discrete_cap_chord_half_width: float | None = None


@dataclass(frozen=True)
class SectionGeometryReport:
    """Container for per-component section properties."""

    components: Mapping[str, Dict[str, Any]]
