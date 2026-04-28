"""
ShellMidlineStrip
=================
FE-agnostic DTO carrying one subcomponent's mid-surface locus and thickness
for handoff from ``section_geometry`` to a shell mesh builder.

**Boundary contract:**

* This record is a *geometric locus*, not an SDF evaluation cache.  The
  ``midline_b`` array is the materialised polyline of the subcomponent's
  mid-surface in the rotated **B-frame** — produced by calling
  ``midline_polyline()`` on the chord-frame subcomponent object and rotating
  into the B-frame via ``rotate_chord_to_blade``.

* It carries no MITC4-, ``Panel``-, laminate-, or ``n_elements``-level
  information; those details are added downstream in ``section_shell_model``.

* Consumers (e.g. ``build_shell_mesh_inputs``) must not call any SDF
  evaluation (``section[label](x, y)``) after this record exists — the polyline
  is the sole geometry source beyond this point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ShellMidlineStrip:
    """Mid-surface strip for one SDF subcomponent; FE-kernel-agnostic.

    Parameters
    ----------
    label
        Component key from ``MultiCellSection._components_unrotated`` or
        ``_spar_cap_components_unrotated``
        (e.g. ``"outer_skin"``, ``"web_0"``, ``"spar_cap_upper"``).
    kind
        Shell role: ``"skin"``, ``"web"``, or ``"cap"``.
    midline_b
        ``(N, 2)`` polyline in the rotated **B-frame**.

        * ``"skin"``  : closed loop, TE → upper → LE → lower → TE.
        * ``"web"``   : open, top → bot.
        * ``"cap"``   : open, ascending in chord-frame x.
    thickness_m
        Full laminate through-thickness in metres.
    closed
        ``True`` for the outer skin loop, ``False`` for webs and caps.
    surface
        For caps only: ``"upper"`` or ``"lower"``.  ``None`` otherwise.
    alignment
        For webs only: ``"chord_normal"`` or ``"flapwise"``.  ``None``
        otherwise.
    """

    label: str
    kind: str
    midline_b: NDArray[np.float64]
    thickness_m: float
    closed: bool = False
    surface: str | None = None
    alignment: str | None = None

    def arc_length_m(self) -> float:
        """Total polyline arc length in metres (B-frame)."""
        if self.midline_b.shape[0] < 2:
            return 0.0
        d = np.diff(self.midline_b, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


__all__ = ["ShellMidlineStrip"]
