"""
geometry.csg
============
Constructive Solid Geometry (CSG) operations on SDF fields.

Boolean operations use both sharp (exact) and smooth (C-infinity) variants.
Smooth operations use the polynomial blending kernel from Inigo Quilez,
which preserves the SDF property approximately near the blend zone.

All operations accept either:
  - Pre-evaluated ndarray fields (phi_a, phi_b already evaluated on a grid), OR
  - Callables  f(x, y) → ndarray  (lazy evaluation)

The functional form is preferred for composition without materialising
intermediate grids.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal dispatch
# ---------------------------------------------------------------------------

def _eval(f, x, y):
    """Evaluate f on (x, y): if callable, call it; else return as-is."""
    if callable(f):
        return f(x, y)
    return np.asarray(f, dtype=float)


# ---------------------------------------------------------------------------
# Sharp Boolean operations
# ---------------------------------------------------------------------------

def union(f_a, f_b):
    """Boolean union: φ = min(φ_a, φ_b).

    Returns a callable (x, y) → ndarray.
    """
    def _op(x, y):
        return np.minimum(_eval(f_a, x, y), _eval(f_b, x, y))
    return _op


def intersect(f_a, f_b):
    """Boolean intersection: φ = max(φ_a, φ_b)."""
    def _op(x, y):
        return np.maximum(_eval(f_a, x, y), _eval(f_b, x, y))
    return _op


def subtract(f_base, f_cutter):
    """Boolean subtraction: remove f_cutter from f_base.

    φ = max(φ_base, −φ_cutter)
    """
    def _op(x, y):
        return np.maximum(_eval(f_base, x, y), -_eval(f_cutter, x, y))
    return _op


# ---------------------------------------------------------------------------
# Smooth Boolean operations  (Quilez polynomial blend)
# ---------------------------------------------------------------------------

def smooth_union(f_a, f_b, k=0.1):
    """Smooth union with blending radius k.

    Uses the polynomial smooth-min (smin) kernel:
        h = clamp(0.5 + 0.5*(b-a)/k, 0, 1)
        smin = mix(b, a, h) - k*h*(1-h)

    Parameters
    ----------
    k : float
        Blend radius in the same units as the SDF. Larger → smoother blend.
    """
    def _op(x, y):
        a = _eval(f_a, x, y)
        b = _eval(f_b, x, y)
        h = np.clip(0.5 + 0.5 * (b - a) / (k + 1e-30), 0.0, 1.0)
        return a * h + b * (1.0 - h) - k * h * (1.0 - h)
    return _op


def smooth_intersect(f_a, f_b, k=0.1):
    """Smooth intersection (smooth-max)."""
    def _op(x, y):
        a = _eval(f_a, x, y)
        b = _eval(f_b, x, y)
        h = np.clip(0.5 - 0.5 * (b - a) / (k + 1e-30), 0.0, 1.0)
        return a * h + b * (1.0 - h) + k * h * (1.0 - h)
    return _op


def smooth_subtract(f_base, f_cutter, k=0.1):
    """Smooth subtraction."""
    def _neg_cutter(x, y):
        return -_eval(f_cutter, x, y)
    return smooth_intersect(f_base, _neg_cutter, k=k)


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

def offset(f, amount):
    """Offset (dilate if amount > 0, erode if amount < 0).

    Exact for true SDFs; approximate after repeated CSG operations.

    Parameters
    ----------
    amount : float
        Positive → expand boundary outward; negative → shrink.
    """
    def _op(x, y):
        return _eval(f, x, y) - amount
    return _op


def shell(f, thickness):
    """Extract a shell of given thickness around a surface.

    φ_shell = |φ_f| − thickness/2

    Parameters
    ----------
    thickness : float
        Full shell thickness (half on each side of the zero-level-set).
    """
    half = thickness / 2.0
    def _op(x, y):
        return np.abs(_eval(f, x, y)) - half
    return _op


def blend(f_a, f_b, t):
    """Linear blend between two SDF fields.

    φ = (1-t)*φ_a + t*φ_b

    Note: the result is generally NOT a true SDF but useful for morphing.

    Parameters
    ----------
    t : float in [0, 1]
        Blend parameter (0 → f_a, 1 → f_b).
    """
    def _op(x, y):
        return (1.0 - t) * _eval(f_a, x, y) + t * _eval(f_b, x, y)
    return _op


# ---------------------------------------------------------------------------
# Compound helpers
# ---------------------------------------------------------------------------

def union_all(fields):
    """Union of an arbitrary list of SDF callables."""
    if not fields:
        raise ValueError("fields must be non-empty.")
    result = fields[0]
    for f in fields[1:]:
        result = union(result, f)
    return result


def intersect_all(fields):
    """Intersection of an arbitrary list of SDF callables."""
    if not fields:
        raise ValueError("fields must be non-empty.")
    result = fields[0]
    for f in fields[1:]:
        result = intersect(result, f)
    return result
