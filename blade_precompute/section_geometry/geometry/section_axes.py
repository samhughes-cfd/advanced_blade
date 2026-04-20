"""
Chord-frame reference stations for structural family B (single-band caps).

Pitching-axis and max-thickness loci are geometric only — not elastic/shear axes.
"""

from __future__ import annotations

import numpy as np

from .airfoil import AirfoilSDF


def pitch_axis_x_from_le(airfoil: AirfoilSDF, fraction: float) -> float:
    """Chordwise x at ``fraction * chord`` measured from the leading edge.

    Uses ``airfoil.leading_edge[0]`` as LE x when available; otherwise falls back
    to ``min(vertices[:,0])``.
    """
    try:
        le_x = float(airfoil.leading_edge[0])
    except Exception:
        le_x = float(np.min(airfoil.vertices[:, 0]))
    return le_x + float(fraction) * float(airfoil.chord)


def max_thickness_chord_x(airfoil: AirfoilSDF, n_points: int = 200) -> float:
    """Chord station where ``thickness_distribution`` is maximal (sampled)."""
    xc, t = airfoil.thickness_distribution(n_points=n_points)
    i = int(np.argmax(t))
    return float(xc[i])
