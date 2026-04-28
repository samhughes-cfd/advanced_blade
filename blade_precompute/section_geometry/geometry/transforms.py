"""
geometry.transforms
===================
Rigid-body and affine transforms for SDF callables.

All transforms operate in the *pull-back* sense: instead of moving the
geometry, we transform the query coordinates into the geometry's local frame.
This preserves the SDF property exactly for rotations and translations
(which are isometries).

Key functions
-------------
rotate_field(f, angle, cx, cy)
    Rotate the SDF field by `angle` radians CCW about (cx, cy).
    Equivalent to evaluating f in a frame rotated CW by `angle`.

translate_field(f, dx, dy)
    Translate the SDF field by (dx, dy).

SDFFrame
    Composable transform stack: translate → rotate → scale (applied in order).
    Call sdf_frame.apply(f) to wrap an SDF callable with the full transform.

Usage
-----
    from geometry.transforms import rotate_field, SDFFrame

    # Web aligned to flapwise axis at 15° twist:
    web_local = lambda x, y: sdf_capsule(x, y, 0.3, 0.08, 0.3, -0.05, 0.004)
    web_global = rotate_field(web_local, twist_angle, cx=0.3, cy=0.0)

    # Composable frame:
    frame = SDFFrame().rotate(np.radians(15), cx=0.3, cy=0.0).translate(0.05, 0.0)
    web_global = frame.apply(web_local)
"""

import numpy as np


def _ensure_f64(a):
    """Return float64 ndarray with a fast-path for ndarray inputs."""
    if isinstance(a, np.ndarray) and a.dtype == np.float64:
        return a
    return np.asarray(a, dtype=float)


# ---------------------------------------------------------------------------
# Elementary transforms
# ---------------------------------------------------------------------------

def rotate_field(f, angle, cx=0.0, cy=0.0):
    """Wrap an SDF callable with a CCW rotation of `angle` radians about (cx, cy).

    The geometry appears rotated CCW; equivalently the query point is rotated
    CW (pull-back) before being passed to f.

    Parameters
    ----------
    f : callable (x, y) → ndarray
    angle : float
        Rotation angle in radians, CCW positive.
    cx, cy : float
        Centre of rotation.

    Returns
    -------
    callable (x, y) → ndarray
    """
    cos_a = np.cos(-angle)   # pull-back is CW = negative angle
    sin_a = np.sin(-angle)

    def _rotated(x, y):
        x = _ensure_f64(x)
        y = _ensure_f64(y)
        # Shift to rotation centre
        xr = x - cx
        yr = y - cy
        # Apply rotation
        xp = cos_a * xr - sin_a * yr + cx
        yp = sin_a * xr + cos_a * yr + cy
        return f(xp, yp)

    return _rotated


def translate_field(f, dx, dy):
    """Wrap an SDF callable with a translation by (dx, dy).

    Parameters
    ----------
    f : callable (x, y) → ndarray
    dx, dy : float

    Returns
    -------
    callable (x, y) → ndarray
    """
    def _translated(x, y):
        x = _ensure_f64(x)
        y = _ensure_f64(y)
        return f(x - dx, y - dy)
    return _translated


def scale_field(f, sx, sy=None, cx=0.0, cy=0.0):
    """Wrap an SDF callable with anisotropic scaling about (cx, cy).

    Note: anisotropic scaling (sx ≠ sy) distorts the SDF — the result is
    no longer a true distance field.  Use only for uniform scaling (sx == sy)
    if exact distances are required.

    Parameters
    ----------
    f : callable (x, y) → ndarray
    sx : float
        Scale factor in x (and y if sy is None).
    sy : float or None
        Scale factor in y. Defaults to sx (uniform).
    cx, cy : float
        Centre of scaling.

    Returns
    -------
    callable (x, y) → ndarray
    """
    if sy is None:
        sy = sx
    sx = float(sx)
    sy = float(sy)
    if (not np.isfinite(sx)) or (not np.isfinite(sy)):
        raise ValueError(f"sx and sy must be finite; got sx={sx!r}, sy={sy!r}.")
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError(
            f"sx and sy must be strictly positive to define a valid scaling transform; got sx={sx!r}, sy={sy!r}."
        )

    def _scaled(x, y):
        x = _ensure_f64(x)
        y = _ensure_f64(y)
        xp = (x - cx) / sx + cx
        yp = (y - cy) / sy + cy
        # Correct distance by minimum scale factor for approximate SDF
        return f(xp, yp) * min(sx, sy)

    return _scaled


def mirror_field_x(f):
    """Mirror the SDF field about the x-axis (y → -y)."""
    def _mirrored(x, y):
        return f(_ensure_f64(x), -_ensure_f64(y))
    return _mirrored


def mirror_field_y(f):
    """Mirror the SDF field about the y-axis (x → -x)."""
    def _mirrored(x, y):
        return f(-_ensure_f64(x), _ensure_f64(y))
    return _mirrored


# ---------------------------------------------------------------------------
# Composable transform stack
# ---------------------------------------------------------------------------

class SDFFrame:
    """A composable, ordered sequence of 2-D rigid-body transforms.

    Transforms are stored and applied in the order they are added.
    Call ``apply(f)`` to wrap an SDF callable with the full chain.

    The pull-back convention means transforms are applied to query
    coordinates in *reverse* order (last-added first).

    Example
    -------
        frame = SDFFrame().translate(0.1, 0.0).rotate(np.radians(15), cx=0.3)
        web_global = frame.apply(web_local_sdf)
    """

    def __init__(self):
        self._ops = []   # list of (type, kwargs)

    # ------------------------------------------------------------------
    # Builder methods (return self for chaining)
    # ------------------------------------------------------------------

    def rotate(self, angle, cx=0.0, cy=0.0):
        """Add a CCW rotation of `angle` radians about (cx, cy)."""
        self._ops.append(("rotate", {"angle": angle, "cx": cx, "cy": cy}))
        return self

    def translate(self, dx, dy):
        """Add a translation by (dx, dy)."""
        self._ops.append(("translate", {"dx": dx, "dy": dy}))
        return self

    def scale(self, sx, sy=None, cx=0.0, cy=0.0):
        """Add a (uniform or anisotropic) scaling."""
        self._ops.append(("scale", {"sx": sx, "sy": sy, "cx": cx, "cy": cy}))
        return self

    def mirror_x(self):
        """Add a mirror about the x-axis."""
        self._ops.append(("mirror_x", {}))
        return self

    def mirror_y(self):
        """Add a mirror about the y-axis."""
        self._ops.append(("mirror_y", {}))
        return self

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, f):
        """Wrap callable f with this transform chain.

        Transforms are applied to f in reverse order (pull-back convention):
        the last transform added acts on the query coordinates first.

        Parameters
        ----------
        f : callable (x, y) → ndarray

        Returns
        -------
        callable (x, y) → ndarray
        """
        # Pull-back convention: to apply transforms T1, T2, … in order
        # (geometry moves T1 then T2), we wrap f as:
        #   wrapped(p) = f(T1^-1(T2^-1(p)))
        # Equivalently, we build the wrapping chain in reverse order
        # so the last-added transform is the outermost (applied first to query).
        # However, SDFFrame stores ops in application order, so we wrap from
        # last to first: each new wrapper pre-transforms before calling inner.
        result = f
        for op_type, kwargs in self._ops:   # forward order: each wraps around result
            if op_type == "rotate":
                result = rotate_field(result, **kwargs)
            elif op_type == "translate":
                result = translate_field(result, **kwargs)
            elif op_type == "scale":
                result = scale_field(result, **kwargs)
            elif op_type == "mirror_x":
                result = mirror_field_x(result)
            elif op_type == "mirror_y":
                result = mirror_field_y(result)
        return result

    def inverse(self):
        """Return a new SDFFrame representing the inverse transform chain."""
        inv = SDFFrame()
        for op_type, kwargs in self._ops:
            if op_type == "rotate":
                inv._ops.insert(0, ("rotate", {
                    "angle": -kwargs["angle"],
                    "cx": kwargs["cx"],
                    "cy": kwargs["cy"],
                }))
            elif op_type == "translate":
                inv._ops.insert(0, ("translate", {
                    "dx": -kwargs["dx"],
                    "dy": -kwargs["dy"],
                }))
            elif op_type == "scale":
                sx = kwargs["sx"]
                sy = kwargs.get("sy") or sx
                inv._ops.insert(0, ("scale", {
                    "sx": 1.0 / sx,
                    "sy": 1.0 / sy,
                    "cx": kwargs.get("cx", 0.0),
                    "cy": kwargs.get("cy", 0.0),
                }))
            elif op_type in ("mirror_x", "mirror_y"):
                inv._ops.insert(0, (op_type, {}))  # self-inverse
        return inv

    def compose(self, other):
        """Return a new SDFFrame that applies self then other."""
        new_frame = SDFFrame()
        new_frame._ops = self._ops + other._ops
        return new_frame

    def __repr__(self):
        ops_str = ", ".join(
            f"{t}({', '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in kw.items())})"
            for t, kw in self._ops
        )
        return f"SDFFrame([{ops_str}])"


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rotate_points(x, y, angle, cx=0.0, cy=0.0):
    """Rotate (x, y) point arrays CCW by `angle` radians about (cx, cy).

    Returns
    -------
    xr, yr : ndarray
    """
    x = _ensure_f64(x)
    y = _ensure_f64(y)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    xr = cos_a * (x - cx) - sin_a * (y - cy) + cx
    yr = sin_a * (x - cx) + cos_a * (y - cy) + cy
    return xr, yr


def flapwise_aligned_web_angle(twist_angle):
    """Return the in-plane rotation needed to align a web with the flapwise axis.

    A chord-normal web has its axis vertical (along y) when twist = 0.
    The flapwise axis in the global frame is vertical regardless of twist.
    To keep the web flapwise-aligned in a section rotated by `twist_angle`,
    the web must be counter-rotated by `-twist_angle`.

    Parameters
    ----------
    twist_angle : float
        Section twist in radians (CCW positive, LE up).

    Returns
    -------
    float
        Web rotation angle to apply (radians).
    """
    return -twist_angle
