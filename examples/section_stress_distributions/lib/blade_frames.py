"""
Blade body frame B vs section-fixed frame S (chord-attached).

Convention (documented — keep consistent with STRESS_MODEL.md):
  - **S** (section, chord-fixed): ``y`` chordwise LE→TE, ``z`` thickness-wise (same as
    ``multi_cell_blade_section`` plots).
  - **B** (blade body at station x): ``edge`` = edgewise axis, ``flap`` = flapwise axis;
    both lie in the section plane, orthogonal, right-handed with spanwise ``+x`` (root→tip).

**Geometric twist** ``theta_geom`` [rad] is the angle from **B** to **S**: the chord (S)
is rotated **CCW** in the (y,z) plane when looking **along +x** from tip toward root
(right-hand rule about +x). Components of a fixed spatial vector satisfy

  ``[Vy, Vz]_S = R(theta_geom) @ [V_edge, V_flap]_B``

with ``R`` the usual 2×2 rotation. **N** (axial) and **T** (torque about x) are unchanged
under this in-plane change of basis for the cross-section resultants.

Sign check: ``theta_geom = 0`` ⇒ ``Vy = V_edge``, ``Vz = V_flap``.
"""

from __future__ import annotations

import numpy as np


def rotation_B_to_S(theta_geom_rad: float) -> np.ndarray:
    """2×2 rotation ``R`` with ``v_S = R @ v_B`` for column vectors ``[comp0, comp1]``."""
    t = float(theta_geom_rad)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)


def shear_B_to_S(V_edge: float, V_flap: float, theta_geom_rad: float) -> tuple[float, float]:
    """Map shear components from B (edge, flap) to S (Vy, Vz)."""
    R = rotation_B_to_S(theta_geom_rad)
    v = R @ np.array([V_edge, V_flap], dtype=float)
    return float(v[0]), float(v[1])


def moment_B_to_S(M_edge: float, M_flap: float, theta_geom_rad: float) -> tuple[float, float]:
    """Map bending moments from B (about edge / about flap axes) to S (My, Mz)."""
    R = rotation_B_to_S(theta_geom_rad)
    m = R @ np.array([M_edge, M_flap], dtype=float)
    return float(m[0]), float(m[1])


def resultants_B_to_S(
    N: float,
    V_edge: float,
    V_flap: float,
    M_edge: float,
    M_flap: float,
    T: float,
    theta_geom_rad: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Full section resultant vector in S for use with ``run_section``.

    Returns
    -------
    N, Vy, Vz, My, Mz, T
    """
    Vy, Vz = shear_B_to_S(V_edge, V_flap, theta_geom_rad)
    My, Mz = moment_B_to_S(M_edge, M_flap, theta_geom_rad)
    return float(N), Vy, Vz, My, Mz, float(T)
