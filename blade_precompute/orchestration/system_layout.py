"""SystemType-style layout keys → frozen layout specs (SDF-first; no Shapely)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

WebOrientation = Literal["chord_normalwise", "flapwise"]
WebPositioning = Literal["uniform", "structural", "fixed", "none"]


@dataclass(frozen=True)
class SystemLayoutSpec:
    """Sandbox mirror of ``SystemTypeN`` *intent* (see blade-structure ``system_type_n.py``).

    ``web_chord_fracs`` are fractions in ``[0, 1]`` measured along chord from the
    airfoil leading edge in section *x* (same convention as :class:`AirfoilSDF`).

    When ``n_webs == 0``, the precompute driver builds an **outer-skin-only** callable
    map (true single-region SDF). Spar-cap-only ``0B`` topology without webs is not
    yet represented in :class:`BladeSectionGeometry` / :class:`MultiCellSection`; use
    ``geometry_mode`` in exports to distinguish.
    """

    key: str
    n_cells: int
    n_webs: int
    has_spar_caps: bool
    web_orientation: WebOrientation
    web_positioning: WebPositioning
    web_chord_fracs: tuple[float, ...]
    spar_cap_positioning: Literal["max_thickness", "fixed", "box_spar", "none"]
    geometry_mode: Literal["multicell", "airfoil_sdf_only"]
    description: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "n_cells": self.n_cells,
            "n_webs": self.n_webs,
            "has_spar_caps": self.has_spar_caps,
            "web_orientation": self.web_orientation,
            "web_positioning": self.web_positioning,
            "web_chord_fracs": list(self.web_chord_fracs),
            "spar_cap_positioning": self.spar_cap_positioning,
            "geometry_mode": self.geometry_mode,
            "description": self.description,
        }


def _uniform_fracs(n_webs: int) -> tuple[float, ...]:
    if n_webs <= 0:
        return ()
    edges = [i / (n_webs + 1) for i in range(1, n_webs + 1)]
    return tuple(float(x) for x in edges)


_LAYOUT_REGISTRY: dict[str, SystemLayoutSpec] = {
    # Matches legacy :class:`BladeSectionGeometry` default shear web chord fractions.
    "legacy": SystemLayoutSpec(
        key="legacy",
        n_cells=3,
        n_webs=2,
        has_spar_caps=True,
        web_orientation="chord_normalwise",
        web_positioning="fixed",
        web_chord_fracs=(0.15, 0.50),
        spar_cap_positioning="max_thickness",
        geometry_mode="multicell",
        description="Historical default: two chord-normal webs at 15% and 50% chord.",
    ),
    # 1 cell, 0 webs — outer airfoil SDF only (matches “skin-only” medial use).
    "0A": SystemLayoutSpec(
        key="0A",
        n_cells=1,
        n_webs=0,
        has_spar_caps=False,
        web_orientation="chord_normalwise",
        web_positioning="none",
        web_chord_fracs=(),
        spar_cap_positioning="none",
        geometry_mode="airfoil_sdf_only",
        description="Type 0A analogue: one cell, no webs, no spar caps (SDF = outer skin).",
    ),
    "0B": SystemLayoutSpec(
        key="0B",
        n_cells=1,
        n_webs=0,
        has_spar_caps=True,
        web_orientation="chord_normalwise",
        web_positioning="none",
        web_chord_fracs=(),
        spar_cap_positioning="max_thickness",
        geometry_mode="airfoil_sdf_only",
        description="Type 0B intent: spar caps without webs — currently exported as airfoil SDF only until zero-web cap SDF is added.",
    ),
    "1B": SystemLayoutSpec(
        key="1B",
        n_cells=2,
        n_webs=1,
        has_spar_caps=True,
        web_orientation="chord_normalwise",
        web_positioning="fixed",
        web_chord_fracs=(0.50,),
        spar_cap_positioning="max_thickness",
        geometry_mode="multicell",
        description="Two cells, one chord-normal web, continuous spar caps.",
    ),
    "2B": SystemLayoutSpec(
        key="2B",
        n_cells=3,
        n_webs=2,
        has_spar_caps=True,
        web_orientation="chord_normalwise",
        web_positioning="uniform",
        web_chord_fracs=_uniform_fracs(2),
        spar_cap_positioning="max_thickness",
        geometry_mode="multicell",
        description="Three cells, two uniform webs.",
    ),
    "3B": SystemLayoutSpec(
        key="3B",
        n_cells=4,
        n_webs=3,
        has_spar_caps=True,
        web_orientation="chord_normalwise",
        web_positioning="uniform",
        web_chord_fracs=_uniform_fracs(3),
        spar_cap_positioning="max_thickness",
        geometry_mode="multicell",
        description="Four cells, three uniform webs.",
    ),
    "2B-F": SystemLayoutSpec(
        key="2B-F",
        n_cells=3,
        n_webs=2,
        has_spar_caps=True,
        web_orientation="flapwise",
        web_positioning="uniform",
        web_chord_fracs=_uniform_fracs(2),
        spar_cap_positioning="max_thickness",
        geometry_mode="multicell",
        description="Three cells, two flapwise-aligned webs (web_alignment=flapwise).",
    ),
}


SYSTEM_TYPE_KEYS: tuple[str, ...] = tuple(sorted(_LAYOUT_REGISTRY.keys()))


def resolve_system_type(key: str) -> SystemLayoutSpec:
    k = (key or "").strip()
    if not k:
        raise ValueError("system_type key must be a non-empty string.")
    if k not in _LAYOUT_REGISTRY:
        allowed = ", ".join(SYSTEM_TYPE_KEYS)
        raise KeyError(f"Unknown system_type {k!r}. Known keys: {allowed}")
    return _LAYOUT_REGISTRY[k]


def build_section_view(
    airfoil_sdf: Any,
    layout: SystemLayoutSpec,
    *,
    twist_angle_rad: float,
) -> Any:
    """Return a section object usable by :class:`SectionPropertiesReport` (labels + callables)."""
    from blade_precompute.section_geometry.sections.section import BladeSectionGeometry

    if layout.geometry_mode == "airfoil_sdf_only":
        return _OuterSkinOnlySection(airfoil_sdf, layout=layout)
    xs = tuple(float(f) * float(airfoil_sdf.chord) for f in layout.web_chord_fracs)
    if len(xs) != layout.n_webs:
        raise ValueError("web_chord_fracs length must equal n_webs for multicell mode.")
    align = "flapwise" if layout.web_orientation == "flapwise" else "chord_normal"
    return BladeSectionGeometry.from_n_web(
        airfoil_sdf,
        web_x_positions=list(xs),
        web_thickness=0.004,
        web_alignment=align,
        cap_height=0.012,
        skin_thickness=0.003,
        twist_angle=float(twist_angle_rad),
        core_enabled=True,
    )


class _OuterSkinOnlySection(Mapping[str, Any]):
    """Minimal mapping: only ``outer_skin`` SDF (for ``0A`` / ``0B`` airfoil-only mode)."""

    __slots__ = ("_af", "_layout")

    def __init__(self, airfoil_sdf: Any, *, layout: SystemLayoutSpec) -> None:
        self._af = airfoil_sdf
        self._layout = layout

    @property
    def labels(self) -> tuple[str, ...]:
        return ("outer_skin",)

    def __getitem__(self, label: str) -> Any:
        if label != "outer_skin":
            raise KeyError(label)
        return self._af

    def __iter__(self) -> Any:
        return iter(self.labels)

    def __len__(self) -> int:
        return 1

    @property
    def airfoil(self) -> Any:
        return self._af

    def eval_union(self, x: Any, y: Any) -> Any:
        return self._af(x, y)
