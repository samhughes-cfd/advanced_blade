"""
utils.py — Section builders and serialisation utilities.

Functions
---------
naca4_section(code, chord, n_walls, material, n_strips_per_wall)
    Build a CrossSection from a NACA 4-digit aerofoil profile.

box_section(width, height, t_material, n_strips)
    Build a thin-walled rectangular box section.

i_section(height, flange_width, web_thickness, flange_thickness,
          web_material, flange_material, n_strips)
    Build a symmetric I/H section.

c_section(height, flange_width, material, n_strips)
    Build a lipped-C section.

section_to_dict(section)
    Serialise a CrossSection geometry to a JSON-compatible dict.

section_from_dict(d, material_map)
    Reconstruct a CrossSection from a serialised dict.
"""
from __future__ import annotations
import json
import numpy as np
from numpy.typing import NDArray

from .section import CrossSection, WallDefinition


# ---------------------------------------------------------------------------
# NACA 4-digit aerofoil section builder
# ---------------------------------------------------------------------------

def _naca4_coords(code: str, n_points: int = 100) -> tuple[NDArray, NDArray]:
    """
    Return upper and lower surface (x, y) coordinates for a NACA 4-digit profile.

    Parameters
    ----------
    code     : str   4-digit NACA code, e.g. '0012', '2412'.
    n_points : int   Number of points per surface (default 100).

    Returns
    -------
    upper, lower : each (n_points, 2) array of (x, y) in normalised chord [0,1].
    """
    code = str(code).zfill(4)
    m  = int(code[0]) / 100.0    # max camber
    p  = int(code[1]) / 10.0     # position of max camber
    t  = int(code[2:]) / 100.0   # max thickness

    x = np.linspace(0, 1, n_points)

    # Thickness distribution
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)

    # Camber line
    yc = np.where(
        x < p,
        (m / p**2) * (2*p*x - x**2),
        (m / (1-p)**2) * ((1 - 2*p) + 2*p*x - x**2)
    ) if p > 0 else np.zeros_like(x)

    # Camber gradient
    dyc_dx = np.where(
        x < p,
        (2*m / p**2) * (p - x),
        (2*m / (1-p)**2) * (p - x)
    ) if p > 0 else np.zeros_like(x)

    theta = np.arctan(dyc_dx)

    upper = np.column_stack([x  - yt * np.sin(theta),
                              yc + yt * np.cos(theta)])
    lower = np.column_stack([x  + yt * np.sin(theta),
                              yc - yt * np.cos(theta)])
    return upper, lower


def naca4_section(
    code: str,
    chord: float,
    material,
    n_walls: int = 20,
    n_strips_per_wall: int = 2,
) -> CrossSection:
    """
    Build a CrossSection approximating a NACA 4-digit aerofoil profile.

    The profile is discretised into n_walls straight wall segments following
    the upper surface from leading edge to trailing edge, and n_walls segments
    along the lower surface back to the leading edge.

    Parameters
    ----------
    code               : str    NACA 4-digit code (e.g. '0015').
    chord              : float  Chord length [m].
    material           : Material  Applied uniformly to all walls.
    n_walls            : int    Wall segments per surface (default 20).
    n_strips_per_wall  : int    Finite strips per wall (default 2).

    Returns
    -------
    CrossSection
    """
    upper, lower = _naca4_coords(code, n_walls + 1)

    # Scale by chord and reorder: upper LE→TE, lower TE→LE (closed loop)
    upper_pts = upper * chord
    lower_pts = lower * chord
    # Perimeter: upper LE→TE then lower TE→LE
    pts = np.vstack([upper_pts, lower_pts[-2::-1]])

    walls = []
    for i in range(len(pts) - 1):
        p0 = pts[i];  p1 = pts[i + 1]
        label = 'upper' if i < n_walls else 'lower'
        walls.append(WallDefinition(
            node_start = [p0[0], p0[1]],
            node_end   = [p1[0], p1[1]],
            material   = material,
            n_strips   = n_strips_per_wall,
            name       = f'{label}_{i}',
        ))

    return CrossSection(walls)


# ---------------------------------------------------------------------------
# Parametric thin-walled section builders
# ---------------------------------------------------------------------------

def box_section(
    width: float,
    height: float,
    material,
    n_strips: int = 4,
) -> CrossSection:
    """
    Thin-walled rectangular box section.

    Corners: bottom-left (0,0), bottom-right (w,0), top-right (w,h), top-left (0,h).
    """
    w, h = width, height
    walls = [
        WallDefinition([0, 0],   [w, 0],   material, n_strips, 'bottom'),
        WallDefinition([w, 0],   [w, h],   material, n_strips, 'right'),
        WallDefinition([w, h],   [0, h],   material, n_strips, 'top'),
        WallDefinition([0, h],   [0, 0],   material, n_strips, 'left'),
    ]
    return CrossSection(walls)


def i_section(
    height: float,
    flange_width: float,
    web_material,
    flange_material=None,
    n_strips_web: int = 6,
    n_strips_flange: int = 3,
) -> CrossSection:
    """
    Symmetric I/H section.  Web runs from (0,0) to (0,height).
    Flanges are symmetric about the web centreline.

    Parameters
    ----------
    height         : float  Web height [m].
    flange_width   : float  Total flange width [m] (half extends each side).
    web_material   : Material
    flange_material: Material or None  (uses web_material if None)
    """
    if flange_material is None:
        flange_material = web_material
    h = height; b = flange_width / 2.0
    walls = [
        WallDefinition([0, 0],  [0, h],   web_material,    n_strips_web,    'web'),
        WallDefinition([-b, 0], [b, 0],   flange_material, n_strips_flange, 'bot_flange'),
        WallDefinition([-b, h], [b, h],   flange_material, n_strips_flange, 'top_flange'),
    ]
    return CrossSection(walls)


def c_section(
    height: float,
    flange_width: float,
    material,
    n_strips_web: int = 6,
    n_strips_flange: int = 3,
) -> CrossSection:
    """
    Thin-walled C (channel) section opening to the right.
    Web: (0,0) to (0,height).  Flanges extend in +y direction.
    """
    h = height; b = flange_width
    walls = [
        WallDefinition([0, h],  [b, h],   material, n_strips_flange, 'top_flange'),
        WallDefinition([0, 0],  [0, h],   material, n_strips_web,    'web'),
        WallDefinition([0, 0],  [b, 0],   material, n_strips_flange, 'bot_flange'),
    ]
    return CrossSection(walls)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def section_to_dict(section: CrossSection) -> dict:
    """
    Serialise a CrossSection's geometry (node coordinates and wall connectivity)
    to a JSON-compatible dictionary.  Material objects are referenced by index.

    Returns
    -------
    dict with keys 'nodes', 'walls', 'strips'.
    """
    return {
        "n_nodes":  section.n_nodes,
        "n_strips": section.n_strips,
        "nodes": [
            {"id": n.node_id, "y": n.y, "z": n.z, "wall_ids": n.wall_ids}
            for n in section._nodes
        ],
        "walls": [
            {
                "index":      i,
                "name":       w.name,
                "node_start": w.node_start.tolist(),
                "node_end":   w.node_end.tolist(),
                "n_strips":   w.n_strips,
            }
            for i, w in enumerate(section.walls)
        ],
        "strips": [
            {
                "strip_id": s.strip_id,
                "wall_id":  s.wall_id,
                "node_i":   s.node_i,
                "node_j":   s.node_j,
                "length":   s.length,
            }
            for s in section._strips
        ],
    }


def section_to_json(section: CrossSection, filepath: str):
    """Write section geometry to a JSON file."""
    d = section_to_dict(section)
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"Section geometry saved to {filepath}")


def section_from_dict(d: dict, material_map: dict) -> CrossSection:
    """
    Reconstruct a CrossSection from a serialised dict.

    Parameters
    ----------
    d            : dict  Output of section_to_dict().
    material_map : dict  Maps wall index (int) or name (str) to a Material object.
                         e.g. {0: IsotropicMaterial(...), 'top_flange': LaminateMaterial(...)}
    """
    walls = []
    for wdict in d['walls']:
        idx  = wdict['index']
        name = wdict['name']
        mat  = material_map.get(idx) or material_map.get(name)
        if mat is None:
            raise ValueError(
                f"No material found for wall index={idx} name='{name}'. "
                "Provide material_map entry for the wall index or name."
            )
        walls.append(WallDefinition(
            node_start = wdict['node_start'],
            node_end   = wdict['node_end'],
            material   = mat,
            n_strips   = wdict['n_strips'],
            name       = name,
        ))
    return CrossSection(walls)


# ---------------------------------------------------------------------------
# Grid convergence index (GCI) helper
# ---------------------------------------------------------------------------

def grid_convergence_index(
    f_fine:   float,
    f_medium: float,
    f_coarse: float,
    r:        float = 2.0,
    p:        float | None = None,
) -> dict:
    """
    Estimate the Grid Convergence Index (GCI) for a scalar quantity f
    computed on three successively refined meshes (refinement ratio r).

    Parameters
    ----------
    f_fine, f_medium, f_coarse : float  Values on fine, medium, coarse meshes.
    r     : float  Refinement ratio (default 2 — mesh doubled each time).
    p     : float or None  Known order of convergence. If None, estimated.

    Returns
    -------
    dict with keys: 'p' (order), 'GCI_fine', 'GCI_medium', 'asymptotic_ratio'.
    """
    eps_21 = f_medium - f_fine
    eps_32 = f_coarse - f_medium

    if abs(eps_21) < 1e-30:
        return dict(p=np.inf, GCI_fine=0.0, GCI_medium=0.0, asymptotic_ratio=1.0)

    if p is None:
        ratio = eps_32 / eps_21
        if abs(ratio) > 0 and ratio > 0:
            p = abs(np.log(abs(ratio)) / np.log(r))
        else:
            p = 2.0   # default second-order

    Fs = 1.25   # factor of safety
    GCI_fine   = Fs * abs(eps_21) / (r**p - 1.0) / abs(f_fine)
    GCI_medium = Fs * abs(eps_32) / (r**p - 1.0) / abs(f_medium)
    asymptotic = GCI_medium / (r**p * GCI_fine) if GCI_fine > 0 else np.nan

    return dict(p=p, GCI_fine=GCI_fine, GCI_medium=GCI_medium,
                asymptotic_ratio=asymptotic)
