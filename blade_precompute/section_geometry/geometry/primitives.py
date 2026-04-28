"""
geometry.primitives
===================
Exact signed distance functions for 2-D geometric primitives.

All functions accept scalar or NumPy array inputs for (x, y) and return
an array of the same broadcast shape.

Sign convention:
    phi < 0  →  interior
    phi = 0  →  boundary
    phi > 0  →  exterior
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_arrays(*args):
    return tuple(_ensure_f64(a) for a in args)


def _ensure_f64(a):
    """Return float64 ndarray with a fast-path for already-correct arrays."""
    if isinstance(a, np.ndarray) and a.dtype == np.float64:
        return a
    return np.asarray(a, dtype=float)


def _clamp(val, lo, hi):
    return np.clip(val, lo, hi)


def _length(vx, vy):
    return np.sqrt(vx**2 + vy**2)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def sdf_circle(x, y, cx=0.0, cy=0.0, r=1.0):
    """Exact SDF for a circle.

    Parameters
    ----------
    x, y : array-like
        Evaluation coordinates.
    cx, cy : float
        Centre coordinates.
    r : float
        Radius (> 0).

    Returns
    -------
    phi : ndarray
        Signed distance field.
    """
    x, y = _as_arrays(x, y)
    return _length(x - cx, y - cy) - r


def sdf_box(x, y, cx=0.0, cy=0.0, half_w=1.0, half_h=0.5):
    """Exact SDF for an axis-aligned rectangle.

    Parameters
    ----------
    cx, cy : float
        Centre of the box.
    half_w, half_h : float
        Half-extents in x and y.
    """
    x, y = _as_arrays(x, y)
    dx = np.abs(x - cx) - half_w
    dy = np.abs(y - cy) - half_h
    outside = _length(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside  = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


def sdf_half_plane(x, y, nx=0.0, ny=1.0, d=0.0):
    """Signed distance to a half-plane  n·p + d ≤ 0.

    The normal (nx, ny) points *outward* (into the positive / exterior region).

    Parameters
    ----------
    nx, ny : float
        Outward unit normal (will be normalised internally).
    d : float
        Signed offset: the boundary is nx*x + ny*y + d = 0.
    """
    x, y = _as_arrays(x, y)
    n_mag = np.sqrt(nx**2 + ny**2)
    if n_mag < 1e-14:
        raise ValueError("Normal vector must be non-zero.")
    return (nx * x + ny * y + d) / n_mag


def sdf_segment(x, y, ax, ay, bx, by):
    """Exact SDF to a line segment AB.

    Returns the unsigned distance to the nearest point on the segment.
    (Useful as a building block; no interior defined for a 1-D object.)
    """
    x, y = _as_arrays(x, y)
    abx, aby = bx - ax, by - ay
    apx, apy = x - ax, y - ay
    t = _clamp((apx * abx + apy * aby) / (abx**2 + aby**2 + 1e-30), 0.0, 1.0)
    return _length(apx - t * abx, apy - t * aby)


def sdf_capsule(x, y, ax, ay, bx, by, r):
    """Exact SDF for a capsule (Minkowski sum of segment and disk of radius r)."""
    return sdf_segment(x, y, ax, ay, bx, by) - r


def sdf_rounded_box(x, y, cx=0.0, cy=0.0, half_w=1.0, half_h=0.5, r=0.1):
    """Exact SDF for a rectangle with uniformly rounded corners.

    Parameters
    ----------
    r : float
        Corner radius (clamped to min(half_w, half_h)).
    """
    r = min(r, min(half_w, half_h))
    x, y = _as_arrays(x, y)
    dx = np.abs(x - cx) - half_w + r
    dy = np.abs(y - cy) - half_h + r
    outside = _length(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside  = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside - r


def sdf_ellipse(x, y, cx=0.0, cy=0.0, rx=1.0, ry=0.5):
    """Approximate SDF for an axis-aligned ellipse.

    Uses the Inigo Quilez iterative approximation (4 Newton iterations),
    which gives sub-pixel accuracy for typical aspect ratios.

    Parameters
    ----------
    rx, ry : float
        Semi-axes in x and y (> 0).
    """
    x, y = _as_arrays(x, y)
    # Map to first quadrant
    px = np.abs(x - cx)
    py = np.abs(y - cy)

    # Initial guess
    tx = np.full_like(px, 0.707)
    ty = np.full_like(py, 0.707)

    for _ in range(4):
        ex = rx * tx
        ey = ry * ty
        qx = px - ex
        qy = py - ey
        r_ = _length(qx, qy)
        r_ = np.where(r_ < 1e-14, 1e-14, r_)
        # Gradient of Lagrangian
        lx = (ex - px) / (rx * rx)
        ly = (ey - py) / (ry * ry)
        l_ = _length(lx, ly)
        l_ = np.where(l_ < 1e-14, 1e-14, l_)
        tx = _clamp(lx / l_, 0.0, 1.0)
        ty = _clamp(ly / l_, 0.0, 1.0)
        t_mag = _length(tx, ty)
        t_mag = np.where(t_mag < 1e-14, 1e-14, t_mag)
        tx /= t_mag
        ty /= t_mag

    nearest_x = rx * tx
    nearest_y = ry * ty
    dist = _length(px - nearest_x, py - nearest_y)
    # Sign: inside ellipse when (x/rx)^2 + (y/ry)^2 < 1
    inside_mask = (px / rx)**2 + (py / ry)**2 <= 1.0
    return np.where(inside_mask, -dist, dist)


def sdf_oriented_box(x, y, cx=0.0, cy=0.0, half_w=1.0, half_h=0.5, angle=0.0):
    """Exact SDF for a rotated axis-aligned rectangle.

    Equivalent to sdf_box evaluated in a frame rotated by -angle about (cx, cy).
    The box principal axes are rotated by `angle` CCW from the x-axis.

    Parameters
    ----------
    cx, cy : float
        Centre of the box.
    half_w, half_h : float
        Half-extents along the box's local x and y axes.
    angle : float
        CCW rotation of the box's local axes from the global x-axis (radians).
    """
    x, y = _as_arrays(x, y)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    # Pull query point into box local frame
    xr = x - cx
    yr = y - cy
    xp = cos_a * xr - sin_a * yr
    yp = sin_a * xr + cos_a * yr
    dx = np.abs(xp) - half_w
    dy = np.abs(yp) - half_h
    outside = _length(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside  = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


def sdf_polygon(x, y, vertices):
    """Exact SDF for a closed polygon (winding-number sign).

    Parameters
    ----------
    vertices : array-like, shape (N, 2)
        Polygon vertices in order (need not be closed; last edge wraps).

    Returns
    -------
    phi : ndarray
    """
    x, y = np.broadcast_arrays(_ensure_f64(x), _ensure_f64(y))
    verts = _ensure_f64(vertices)
    n = int(len(verts))
    if n < 3:
        raise ValueError("Polygon must contain at least 3 vertices.")

    # Flatten grid/sample coordinates once and process in chunks to avoid
    # materialising a full (n_edges, n_points) tensor on very large grids.
    xf = x.ravel()
    yf = y.ravel()
    m = int(xf.size)

    ax = verts[:, 0]
    ay = verts[:, 1]
    bx = verts[(np.arange(n) + 1) % n, 0]
    by = verts[(np.arange(n) + 1) % n, 1]
    abx = bx - ax
    aby = by - ay
    ab_denom = (abx**2 + aby**2 + 1e-30)[:, None]

    d_out = np.empty(m, dtype=np.float64)
    winding_out = np.empty(m, dtype=np.int32)

    chunk_size = 1 << 16
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        px = xf[start:stop][None, :]
        py = yf[start:stop][None, :]

        apx = px - ax[:, None]
        apy = py - ay[:, None]
        t = _clamp((apx * abx[:, None] + apy * aby[:, None]) / ab_denom, 0.0, 1.0)
        dx = apx - t * abx[:, None]
        dy = apy - t * aby[:, None]
        d_out[start:stop] = np.min(dx**2 + dy**2, axis=0)

        c1 = py >= ay[:, None]
        c2 = py < by[:, None]
        c3 = abx[:, None] * (py - ay[:, None]) - aby[:, None] * (px - ax[:, None])
        wn_up = np.sum(c1 & c2 & (c3 > 0.0), axis=0, dtype=np.int32)
        wn_dn = np.sum((~c1) & (~c2) & (c3 < 0.0), axis=0, dtype=np.int32)
        winding_out[start:stop] = wn_up - wn_dn

    sign = np.where(winding_out == 0, 1.0, -1.0)
    return (sign * np.sqrt(d_out)).reshape(x.shape)
