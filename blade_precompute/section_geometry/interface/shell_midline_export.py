"""
shell_midline_export
====================
Export ``section_geometry`` subcomponent midlines as FE-agnostic
:class:`ShellMidlineStrip` records in the blade (B) frame.

**Dependency rule:** this module may import from ``blade_precompute.contract``
and from ``blade_precompute.section_geometry``, but must **never** import from
``blade_precompute.section_shell_model``.

**Strip order:** ``build_shell_midline_strips`` yields skins and webs in
``_components_unrotated`` dict-insertion order, then spar caps in
``_spar_cap_components_unrotated`` dict-insertion order.  Any consumer that
cares about a canonical ordering (e.g. ``skin → caps → webs`` as required by
``topology_v2.build_section_v2``) must sort or reorder independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.contract.shell_midline_strip import ShellMidlineStrip


def rotate_chord_to_blade(
    points_S: NDArray[np.float64], twist_rad: float
) -> NDArray[np.float64]:
    """Rotate ``(N, 2)`` chord-frame points by ``+twist_rad`` into the B-frame.

    Mirrors the rotation that :class:`MultiCellSection` applies to every
    component's SDF when ``twist_angle != 0``: ``x' = R(twist) @ x`` about
    the origin (``cx=0``, ``cy=0``).

    Parameters
    ----------
    points_S
        ``(N, 2)`` array of points in the chord (S) frame.
    twist_rad
        Section twist angle in radians.

    Returns
    -------
    NDArray[np.float64]
        ``(N, 2)`` array of points rotated into the blade (B) frame.
    """
    if abs(twist_rad) < 1e-15:
        return np.asarray(points_S, dtype=float).copy()
    c = float(np.cos(twist_rad))
    s = float(np.sin(twist_rad))
    p = np.asarray(points_S, dtype=float)
    return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]])


def build_shell_midline_strips(
    section: Any,
    *,
    twist_rad: float,
    n_web_samples: int = 20,
    n_cap_samples: int = 80,
) -> tuple[ShellMidlineStrip, ...]:
    """Extract per-subcomponent midline strips from a ``MultiCellSection``.

    Iterates ``_components_unrotated`` (skin + webs) then
    ``_spar_cap_components_unrotated`` (caps), calls ``midline_polyline()``
    on each subcomponent in its chord frame, and rotates each polyline into
    the B-frame via :func:`rotate_chord_to_blade`.

    Parameters
    ----------
    section
        :class:`blade_precompute.section_geometry.sections.multicell.MultiCellSection`
        instance.  Must expose ``_components_unrotated`` and optionally
        ``_spar_cap_components_unrotated``.
    twist_rad
        Section twist in radians applied to every chord-frame midline.
    n_web_samples, n_cap_samples
        Number of sample points for web / cap polylines.  Skin midline
        resolution is set by the underlying airfoil tessellation.

    Returns
    -------
    tuple[ShellMidlineStrip, ...]
        Strips in dict-insertion order: skins + webs first (from
        ``_components_unrotated``), then caps (from
        ``_spar_cap_components_unrotated``).
    """
    from blade_precompute.section_geometry.sections.subcomponents import (
        ContinuousSparCap,
        OuterSkin,
        ShearWeb,
        SparCap,
    )

    components_S = getattr(section, "_components_unrotated", None)
    if components_S is None:
        raise ValueError(
            "section must expose '_components_unrotated'. "
            "This export requires a MultiCellSection-compatible object."
        )

    strips: list[ShellMidlineStrip] = []

    for label, comp in components_S.items():
        if isinstance(comp, OuterSkin):
            mid_S = comp.midline_polyline()
            strips.append(
                ShellMidlineStrip(
                    label=str(label),
                    kind="skin",
                    midline_b=rotate_chord_to_blade(mid_S, twist_rad),
                    thickness_m=float(comp.thickness),
                    closed=True,
                )
            )
        elif isinstance(comp, ShearWeb):
            mid_S = comp.midline_polyline(n=n_web_samples)
            strips.append(
                ShellMidlineStrip(
                    label=str(label),
                    kind="web",
                    midline_b=rotate_chord_to_blade(mid_S, twist_rad),
                    thickness_m=float(comp.thickness),
                    closed=False,
                    alignment=str(comp.alignment),
                )
            )

    cap_components = getattr(section, "_spar_cap_components_unrotated", {}) or {}
    for label, comp in cap_components.items():
        if not isinstance(comp, (ContinuousSparCap, SparCap)):
            continue
        mid_S = comp.midline_polyline(n=n_cap_samples)
        strips.append(
            ShellMidlineStrip(
                label=str(label),
                kind="cap",
                midline_b=rotate_chord_to_blade(mid_S, twist_rad),
                thickness_m=float(comp.cap_height),
                closed=False,
                surface=str(comp.surface),
            )
        )

    return tuple(strips)


__all__ = ["build_shell_midline_strips", "rotate_chord_to_blade"]
