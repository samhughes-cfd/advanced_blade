"""Realistic lower bounds (m) on explicit SDF laminate thickness.

These floors are applied when building :class:`MultiCellSection` and when mapping
optimisation design variables to section strips, so unphysically thin inputs are
not silently used.
"""

from __future__ import annotations

# Conservative blade-laminate floors: shell, shear web, and spar-cap thickness
# are kept at or above typical GFRP/CFRP manufacturing minimums for structural plies.
MIN_REALISTIC_SKIN_LAMINATE_THICKNESS_M = 0.0025
MIN_REALISTIC_WEB_LAMINATE_THICKNESS_M = 0.0040
MIN_REALISTIC_SPAR_LAMINATE_THICKNESS_M = 0.0050


def clamp_skin_thickness_m(t: float) -> float:
    return max(float(t), MIN_REALISTIC_SKIN_LAMINATE_THICKNESS_M)


def clamp_web_thickness_m(t: float) -> float:
    return max(float(t), MIN_REALISTIC_WEB_LAMINATE_THICKNESS_M)


def clamp_spar_laminate_thickness_m(t: float) -> float:
    return max(float(t), MIN_REALISTIC_SPAR_LAMINATE_THICKNESS_M)
