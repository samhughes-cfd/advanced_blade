"""SystemType-style layout keys → frozen layout specs (SDF-first; no Shapely).

Registry dict keys are the **compact** ``SystemType{X}{Y}-{Z}`` tokens from
``blade_precompute/section_geometry/docs/system_type_xyz_taxonomy.md`` (e.g.
``2D-CN``, ``2D-F``). Middle letter is taxonomy **Y** (``A``/``B``/``C``/``D``). Pass these strings to :func:`resolve_system_type`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from blade_precompute.section_geometry.sections.subcomponents import SandwichCore

WebOrientation = Literal["chord_normal", "flapwise"]
WebPositioning = Literal["uniform", "structural", "fixed", "none"]
StructuralFamily = Literal["A", "B", "C", "D"]

# Nominal skin thickness passed to multicell and airfoil-only section views (pre-clamp).
_DEFAULT_SECTION_SKIN_THICKNESS_M = 0.003


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
    structural_family: StructuralFamily = "D"
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
            "structural_family": self.structural_family,
            "description": self.description,
        }


def _uniform_fracs(n_webs: int) -> tuple[float, ...]:
    if n_webs <= 0:
        return ()
    edges = [i / (n_webs + 1) for i in range(1, n_webs + 1)]
    return tuple(float(x) for x in edges)


def _default_web_fracs(n_webs: int) -> tuple[float, ...]:
    # Keep legacy fixed stations for X=2; use centerline for X=1; uniform otherwise.
    if n_webs == 1:
        return (0.50,)
    if n_webs == 2:
        return (0.15, 0.50)
    return _uniform_fracs(n_webs)


def _default_web_positioning(n_webs: int) -> WebPositioning:
    return "fixed" if n_webs <= 2 else "uniform"


def _family_desc(y: str) -> str:
    return {
        "A": "no spar caps",
        "B": "single cap band (pitch / max-thickness)",
        "C": "discrete spar caps per web",
        "D": "continuous box spar caps",
    }[y]


_LAYOUT_REGISTRY: dict[str, SystemLayoutSpec] = {
    # 1 cell, 0 webs — outer airfoil SDF only (matches “skin-only” medial use).
    "0A": SystemLayoutSpec(
        key="0A",
        n_cells=1,
        n_webs=0,
        has_spar_caps=False,
        web_orientation="chord_normal",
        web_positioning="none",
        web_chord_fracs=(),
        spar_cap_positioning="none",
        geometry_mode="airfoil_sdf_only",
        structural_family="A",
        description="Type 0A analogue: one cell, no webs, no spar caps (SDF = outer skin).",
    ),
    "0B": SystemLayoutSpec(
        key="0B",
        n_cells=1,
        n_webs=0,
        has_spar_caps=True,
        web_orientation="chord_normal",
        web_positioning="none",
        web_chord_fracs=(),
        spar_cap_positioning="max_thickness",
        geometry_mode="airfoil_sdf_only",
        structural_family="B",
        description="Type 0B intent: spar caps without webs — currently exported as airfoil SDF only until zero-web cap SDF is added.",
    ),
}

for _x in range(1, 6):
    _fracs = _default_web_fracs(_x)
    _pos = _default_web_positioning(_x)
    for _y in ("A", "B", "C", "D"):
        # D = continuous box between first and last web; with one web it degenerates
        # into zero-span caps and is not a valid physical configuration.
        if _x == 1 and _y == "D":
            continue
        for _z, _orient in (("CN", "chord_normal"), ("F", "flapwise")):
            _key = f"{_x}{_y}-{_z}"
            _has_caps = _y != "A"
            _cap_pos: Literal["max_thickness", "fixed", "box_spar", "none"] = (
                "none" if _y == "A" else "max_thickness"
            )
            _LAYOUT_REGISTRY[_key] = SystemLayoutSpec(
                key=_key,
                n_cells=_x + 1,
                n_webs=_x,
                has_spar_caps=_has_caps,
                web_orientation=_orient,
                web_positioning=_pos,
                web_chord_fracs=_fracs,
                spar_cap_positioning=_cap_pos,
                geometry_mode="multicell",
                structural_family=_y,  # type: ignore[arg-type]
                description=(
                    f"SystemType{_x}{_y}-{_z}: {_x + 1} cells, {_x} "
                    f"{'chord-normal' if _z == 'CN' else 'flapwise'} webs, {_family_desc(_y)}."
                ),
            )


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
    """Return a section object usable by :class:`SectionPropertiesReport` (labels + callables).

    Nominal skin/web thicknesses are clamped to realistic minima inside
    :class:`MultiCellSection` (see ``laminate_thickness_limits``).
    """
    from blade_precompute.section_geometry.sections.section import BladeSectionGeometry

    if layout.geometry_mode == "airfoil_sdf_only":
        return _OuterSkinOnlySection(
            airfoil_sdf,
            layout=layout,
            skin_thickness=_DEFAULT_SECTION_SKIN_THICKNESS_M,
            twist_angle_rad=float(twist_angle_rad),
        )
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
        skin_thickness=_DEFAULT_SECTION_SKIN_THICKNESS_M,
        twist_angle=float(twist_angle_rad),
        core_enabled=True,
        structural_family=layout.structural_family,
    )


def _b_frame_sdf(sdf_s: Any, twist_rad: float) -> Any:
    """Wrap a chord-frame (S-frame) SDF so it evaluates correctly at B-frame coordinates.

    The B→S inverse rotation is ``R(-twist)``: ``x_S = cos·x_B + sin·y_B``,
    ``y_S = -sin·x_B + cos·y_B``.  When ``twist_rad`` is near zero the original
    callable is returned unchanged.
    """
    import math
    if abs(twist_rad) < 1e-12:
        return sdf_s
    c = math.cos(float(twist_rad))
    s = math.sin(float(twist_rad))

    def _eval(x: Any, y: Any) -> Any:
        import numpy as _np
        xf = _np.asarray(x, dtype=float)
        yf = _np.asarray(y, dtype=float)
        return sdf_s(c * xf + s * yf, -s * xf + c * yf)

    return _eval


class _OuterSkinOnlySection(Mapping[str, Any]):
    """Airfoil-only layout: ``outer_skin``, optional spar caps, and ``core_0``.

    **Coordinate frames**

    ``_components_unrotated`` / ``_spar_cap_components_unrotated`` hold
    chord-frame (S-frame) objects consumed by
    :func:`~blade_precompute.section_geometry.interface.shell_midline_export.build_shell_midline_strips`
    for midline export (those routines apply the B-frame rotation themselves).

    ``__getitem__`` returns B-frame-aware SDF callables (via :func:`_b_frame_sdf`)
    so the SDF visualisation in the repro matches the rotated airfoil grid.

    **0B spar caps**

    Upper and lower :class:`~blade_precompute.section_geometry.sections.subcomponents.SparCap`
    objects are built at the max-thickness chord station (±10 % chord band).
    The cap SDF and midline are available immediately; they appear as
    ``"spar_cap_upper"`` / ``"spar_cap_lower"`` in :attr:`labels`.
    """

    __slots__ = (
        "_af", "_layout", "_skin_t", "_twist_rad", "_core_0",
        "_skin_s", "_cap_upper_s", "_cap_lower_s",
        "_skin_b", "_cap_upper_b", "_cap_lower_b",
    )

    _CAP_HEIGHT_M: float = 0.012
    _CAP_HALF_BAND_FRAC: float = 0.10  # ± fraction of chord centred at max-thickness

    def __init__(
        self,
        airfoil_sdf: Any,
        *,
        layout: SystemLayoutSpec,
        skin_thickness: float = _DEFAULT_SECTION_SKIN_THICKNESS_M,
        twist_angle_rad: float = 0.0,
    ) -> None:
        import numpy as _np
        from blade_precompute.section_geometry.sections.subcomponents import OuterSkin, SparCap

        self._af = airfoil_sdf
        self._layout = layout
        self._skin_t = float(skin_thickness)
        self._twist_rad = float(twist_angle_rad)
        self._core_0 = SandwichCore(airfoil_sdf, self._skin_t, exclusion_sdfs=None)

        # S-frame (chord frame) skin — used for midline export
        self._skin_s = OuterSkin(airfoil_sdf, self._skin_t)
        # B-frame skin SDF for visualisation
        self._skin_b = _b_frame_sdf(self._skin_s, self._twist_rad)

        # Caps (0B only)
        self._cap_upper_s: Any = None
        self._cap_lower_s: Any = None
        self._cap_upper_b: Any = None
        self._cap_lower_b: Any = None
        if layout.has_spar_caps:
            chord = float(airfoil_sdf.chord)
            xc, t_dist = airfoil_sdf.thickness_distribution(n_points=200)
            x_mt = float(xc[_np.argmax(t_dist)])
            half = self._CAP_HALF_BAND_FRAC * chord
            x0 = max(float(xc[0]), x_mt - half)
            x1 = min(float(xc[-1]), x_mt + half)
            cap_h = self._CAP_HEIGHT_M
            self._cap_upper_s = SparCap(airfoil_sdf, self._skin_t, x0, x1, cap_h, "upper")
            self._cap_lower_s = SparCap(airfoil_sdf, self._skin_t, x0, x1, cap_h, "lower")
            self._cap_upper_b = _b_frame_sdf(self._cap_upper_s, self._twist_rad)
            self._cap_lower_b = _b_frame_sdf(self._cap_lower_s, self._twist_rad)

    @property
    def labels(self) -> tuple[str, ...]:
        base = ("outer_skin", "core_0")
        if self._layout.has_spar_caps:
            return ("outer_skin", "spar_cap_upper", "spar_cap_lower", "core_0")
        return base

    def __getitem__(self, label: str) -> Any:
        if label == "outer_skin":
            return self._skin_b
        if label == "spar_cap_upper" and self._cap_upper_b is not None:
            return self._cap_upper_b
        if label == "spar_cap_lower" and self._cap_lower_b is not None:
            return self._cap_lower_b
        if label == "core_0":
            return self._core_0
        raise KeyError(label)

    def __iter__(self) -> Any:
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def airfoil(self) -> Any:
        return self._af

    @property
    def _components_unrotated(self) -> dict[str, Any]:
        """Chord-frame skin; consumed by ``build_shell_midline_strips``."""
        return {"outer_skin": self._skin_s}

    @property
    def _spar_cap_components_unrotated(self) -> dict[str, Any]:
        """Chord-frame cap objects for ``0B``; empty for ``0A``."""
        if self._cap_upper_s is not None and self._cap_lower_s is not None:
            return {"spar_cap_upper": self._cap_upper_s, "spar_cap_lower": self._cap_lower_s}
        return {}

    def eval_union(self, x: Any, y: Any) -> Any:
        return self._af(x, y)
