"""
sections.subcomponents
======================
Parameterised SDF callables for individual blade section subcomponents.

Design philosophy
-----------------
Each class wraps a CSG expression built from geometry primitives.  The
resulting object is callable:  phi = component(x, y).

Corrected spar cap construction
--------------------------------
The spar cap outer face is flush against the inner skin surface (the airfoil
boundary offset inward by skin_thickness).  The inner face is a further
inward offset by cap_height — so both faces share identical curvature:

    phi_inner_skin  = offset(phi_airfoil, -skin_thickness / 2.0)

    phi_cap_upper   = intersect(
        cap_laminate_slab(phi_inner_skin, cap_height),  # cavity side: -cap_h < phi_inner < 0
        half_plane(y > 0),                             # suction side only
        chordwise_clip,
    )

``shell(offset(inner_skin, cap_h/2), cap_h)`` was wrong: it places the band
where ``0 < inner_skin < cap_h`` (outboard of the inner mold). The slab
``max(inner_skin, -inner_skin - cap_h)`` keeps the laminate between the inner
mold (``inner_skin=0``) and ``inner_skin=-cap_h`` into the cavity.

Web alignment
-------------
Webs support two alignment modes:
  "chord_normal"  : web axis is perpendicular to the chord line (default).
                    ``(x_top, y_top, x_bot, y_bot)`` are taken in the chord
                    (S) frame; the web axis is vertical in the S-frame and
                    becomes tilted in the B-frame after the section's global
                    +twist rotation.
  "flapwise"      : web axis is locked to the global flapwise (y-B) direction.
                    The CALLER supplies ``(x_top, y_top, x_bot, y_bot)`` in
                    the chord (S) frame as the **inverse-rotation** of the
                    desired B-frame vertical axis (i.e. a tilted S-frame
                    segment that becomes vertical at ``x_b = x_b_web`` in the
                    B-frame after the section's global +twist rotation).
                    This guarantees the web shares the same inner-skin clip
                    as the rest of the section in the B-frame (no SDF gap to
                    the skin/cap). See ``MultiCellSection`` for how the
                    inverse-rotation inputs are computed.

Coordinate conventions
-----------------------
  +x  →  trailing edge
  +y  →  suction (upper) surface
  All dimensions in the same units as the airfoil vertices.
"""

import numpy as np
from ..geometry.csg import intersect, subtract, shell, union, offset
from ..geometry.primitives import sdf_capsule, sdf_circle
from ..geometry.transforms import rotate_field
from ..engine.csg_ir import (
    Capsule,
    CallableField,
    Circle,
    HalfPlane,
    Intersect,
    Offset,
    Polygon,
    Rotate,
    Subtract,
)


def _ensure_f64(a):
    """Return float64 ndarray with a fast-path for ndarray inputs."""
    if isinstance(a, np.ndarray) and a.dtype == np.float64:
        return a
    return np.asarray(a, dtype=float)


def _airfoil_expr(airfoil_sdf):
    """Return polygon IR for AirfoilSDF-like inputs, else None."""
    verts = getattr(airfoil_sdf, "vertices", None)
    if verts is None:
        return None
    arr = np.asarray(verts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        return None
    return Polygon(tuple((float(x), float(y)) for x, y in arr))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chordwise_clip(x_start, x_end):
    """Return an SDF callable that is negative between x_start and x_end."""
    def _clip(x, y):
        x = _ensure_f64(x)
        left  = x - x_start   # < 0 left of x_start
        right = x_end - x     # < 0 right of x_end
        # Interior of clip box = intersect of two half-planes
        return np.maximum(-left, -right)  # standard intersect (max) of two half-planes
    return _clip


def _parallel_web_strip_clip(nx, ny, c_a, c_b):
    """Strip between two parallel lines n·p=c_a and n·p=c_b (same convention as chord clip).

    Use for spar caps / cores when shear webs are **flapwise**-aligned: boundaries
    must be parallel to the web direction, not chord-vertical (constant x).
    """
    lo = float(min(c_a, c_b))
    hi = float(max(c_a, c_b))
    nx = float(nx)
    ny = float(ny)

    def _clip(x, y):
        xa = _ensure_f64(x)
        ya = _ensure_f64(y)
        nd = nx * xa + ny * ya
        return np.maximum(lo - nd, nd - hi)

    return _clip


def _parallel_web_half_lt(nx, ny, c_bound):
    """Interior: n·p < c_bound (e.g. fore of first flapwise web)."""
    nx = float(nx)
    ny = float(ny)
    c = float(c_bound)

    def _clip(x, y):
        xa = _ensure_f64(x)
        ya = _ensure_f64(y)
        nd = nx * xa + ny * ya
        return nd - c

    return _clip


def _parallel_web_half_gt(nx, ny, c_bound):
    """Interior: n·p > c_bound (e.g. aft of last flapwise web)."""
    nx = float(nx)
    ny = float(ny)
    c = float(c_bound)

    def _clip(x, y):
        xa = _ensure_f64(x)
        ya = _ensure_f64(y)
        nd = nx * xa + ny * ya
        return c - nd

    return _clip


def flapwise_web_normal(twist_angle_rad):
    """Unit normal **n** (in chord frame) for lines parallel to flapwise web run.

    A flapwise web is a capsule aligned with global +y after section twist ω;
    in the chord frame its direction is **u**=(sin ω, cos ω).  Boundaries between
    cells are perpendicular to **u**, hence **n** = (-cos ω, sin ω).
    """
    w = float(twist_angle_rad)
    return -np.cos(w), np.sin(w)


def web_station_projection(nx, ny, x_web, y_mid):
    """Scalar n·p at a web chord station (mid-thickness point on the web axis)."""
    return float(nx) * float(x_web) + float(ny) * float(y_mid)


def _upper_half_plane():
    """SDF of the upper half-plane (y >= 0): phi = -y inside."""
    def _hp(x, y):
        return -_ensure_f64(y)
    return _hp


def _lower_half_plane():
    """SDF of the lower half-plane (y <= 0): phi = y inside."""
    def _hp(x, y):
        return _ensure_f64(y)
    return _hp


def _airfoil_offset_arc(
    airfoil_sdf,
    *,
    skin_thickness,
    cap_height,
    x_start,
    x_end,
    surface,
    n=80,
    strip_clip=None,
):
    """Sample the cap mid-surface in the chord frame.

    The cap SDF is a slab between ``inner_skin = 0`` (inner mold line, at
    ``offset(airfoil, -skin_thickness/2)``) and ``inner_skin = -cap_height``
    (inward), so its **midline** lies at ``inner_skin = -cap_height/2``,
    i.e. the airfoil contour offset inward by ``skin_thickness/2 +
    cap_height/2``.

    Endpoint definition
    -------------------
    * If ``strip_clip is None`` (default — chord-normal webs): restricted to
      chord-frame ``[x_start, x_end]`` on the requested surface; sampled at
      ``n`` chord-uniform points. This matches the cap SDF's chord clip
      ``_chordwise_clip(x_start, x_end)``.
    * If ``strip_clip`` is provided (flapwise webs): the cap SDF uses
      ``_parallel_web_strip_clip(nx, ny, c_a, c_b)`` (a strip between two
      parallel lines ``n·p = c_{a,b}``) instead of the chord-vertical
      ``_chordwise_clip``. To keep the midline consistent with the SDF, we
      sample the offset arc densely across a chord range that brackets
      ``[x_start, x_end]``, then prune to the segment where
      ``strip_clip(x, y) <= 0`` (inside the strip) and linearly interpolate
      at the two boundary crossings so the endpoints lie exactly on the
      strip boundary. After the section's global twist this places the
      midline endpoints at the same B-frame x as the bracketing flapwise
      webs (since ``n·p = -x_B`` for ``n = (-cos ω, sin ω)``).
    """
    if surface == "upper":
        seg = np.asarray(airfoil_sdf.upper_surface(), dtype=float)
        sign = -1.0
    elif surface == "lower":
        seg = np.asarray(airfoil_sdf.lower_surface(), dtype=float)
        sign = +1.0
    else:
        raise ValueError(f"surface must be 'upper' or 'lower', got {surface!r}")
    order = np.argsort(seg[:, 0])
    seg = seg[order]
    inward = float(skin_thickness) / 2.0 + float(cap_height) / 2.0

    if strip_clip is None:
        n_use = int(max(2, n))
        xs = np.linspace(float(x_start), float(x_end), n_use)
        ys = np.interp(xs, seg[:, 0], seg[:, 1])
        return np.column_stack([xs, ys + sign * inward])

    # ``strip_clip`` is a parallel-line strip in the chord frame whose
    # boundaries are typically *not* aligned with chord-frame x. Sample the
    # offset arc densely over the full chord (the strip can extend beyond
    # ``[x_start, x_end]`` once mapped to chord-frame x), then prune to the
    # interior of the strip and linearly interpolate at the boundary
    # crossings.
    n_use = int(max(8, 4 * n))
    x_min = float(seg[0, 0])
    x_max = float(seg[-1, 0])
    xs = np.linspace(x_min, x_max, n_use)
    ys = np.interp(xs, seg[:, 0], seg[:, 1]) + sign * inward
    arc = np.column_stack([xs, ys])
    phi = np.asarray(strip_clip(arc[:, 0], arc[:, 1]), dtype=float)
    inside = phi <= 0.0
    if not np.any(inside):
        return arc[:0]
    idx = np.where(inside)[0]
    i_lo = int(idx[0])
    i_hi = int(idx[-1])
    arc_clip = arc[i_lo : i_hi + 1].copy()
    if i_lo > 0:
        a = float(phi[i_lo - 1])
        b = float(phi[i_lo])
        if a != b:
            t = a / (a - b)
            arc_clip[0] = arc[i_lo - 1] + t * (arc[i_lo] - arc[i_lo - 1])
    if i_hi < n_use - 1:
        a = float(phi[i_hi])
        b = float(phi[i_hi + 1])
        if a != b:
            t = a / (a - b)
            arc_clip[-1] = arc[i_hi] + t * (arc[i_hi + 1] - arc[i_hi])
    return arc_clip


def _cap_laminate_slab(inner_skin, cap_height):
    """SDF of spar-cap solid between inner mold and cavity-ward inner face.

    ``inner_skin = offset(airfoil, -t/2)`` is negative in the cavity.  The cap
    occupies ``-cap_height < inner_skin < 0``.  Using ``shell`` around a shifted
    axis placed half the band on the wrong (positive ``inner_skin``) side.

    Returns φ = max(inner_skin, -inner_skin - cap_height); inside when both < 0.
    """
    cap_h = float(cap_height)

    def _op(x, y):
        is_ = _ensure_f64(inner_skin(x, y))
        return np.maximum(is_, -is_ - cap_h)

    return _op


def _cap_laminate_slab_expr(inner_skin_expr, cap_height):
    """IR-compatible cap laminate slab wrapper."""
    cap_h = float(cap_height)
    return CallableField(
        key=f"cap_slab_{id(inner_skin_expr)}_{cap_h:.12g}",
        fn=lambda x, y: np.maximum(
            _ensure_f64(inner_skin_expr.fn(x, y)),
            -_ensure_f64(inner_skin_expr.fn(x, y)) - cap_h,
        ),
    )


# ---------------------------------------------------------------------------
# Outer Skin
# ---------------------------------------------------------------------------

class OuterSkin:
    """Shell of uniform thickness around the airfoil boundary.

    The shell is centred on the outer airfoil zero-level-set, extending
    ±thickness/2 on each side.

    Parameters
    ----------
    airfoil_sdf : callable (x, y) → ndarray
    thickness : float
        Full laminate skin thickness.

    Attributes
    ----------
    inner_sdf : callable
        SDF of the inner skin surface (airfoil eroded by thickness/2).
    outer_sdf : callable
        SDF of the outer skin surface (airfoil dilated by thickness/2).
    """

    label = "outer_skin"

    def __init__(self, airfoil_sdf, thickness):
        self._af        = airfoil_sdf
        self.thickness  = float(thickness)
        self._sdf       = shell(airfoil_sdf, thickness)
        # inner_sdf: boundary moved inward (interior shrinks) → offset by -t/2
        # outer_sdf: boundary moved outward (interior grows)  → offset by +t/2
        self.inner_sdf  = offset(airfoil_sdf, -thickness / 2.0)
        self.outer_sdf  = offset(airfoil_sdf,  thickness / 2.0)

    def __call__(self, x, y):
        return self._sdf(x, y)

    def midline_polyline(self):
        """Return the skin midline polyline in the chord frame.

        ``OuterSkin._sdf = shell(airfoil, thickness)`` is centred on the
        airfoil zero-level-set, so the midline of the laminate is exactly
        the airfoil contour. Returns the airfoil's vertices verbatim
        (closed polyline, ordered TE → upper → LE → lower → TE).
        """
        verts = getattr(self._af, "vertices", None)
        if verts is None:
            raise AttributeError(
                "OuterSkin.midline_polyline requires the underlying airfoil "
                "SDF to expose a 'vertices' attribute (AirfoilSDF-compatible)."
            )
        return np.asarray(verts, dtype=float).copy()

    def __repr__(self):
        return f"OuterSkin(thickness={self.thickness:.4g})"


# ---------------------------------------------------------------------------
# Spar Cap  (corrected offset-surface construction)
# ---------------------------------------------------------------------------

class SparCap:
    """Constant-thickness spar cap flush against the inner skin surface.

    Both the outer and inner faces follow the airfoil curvature exactly:

        outer face : phi_inner_skin  = 0   (flush against inner skin boundary)
        inner face : phi_inner_skin  = -cap_height  (inward offset, same curvature)

    The cap is bounded chordwise by [x_start, x_end] and restricted to the
    upper or lower surface by a half-plane clip.

    Parameters
    ----------
    airfoil_sdf : callable
        Outer airfoil boundary SDF.
    skin_thickness : float
        Skin laminate full thickness. Inner mold line matches ``OuterSkin``:
        offset mid-surface by ``-skin_thickness/2`` (see ``OuterSkin.inner_sdf``).
    x_start, x_end : float
        Chordwise extent of the cap.
    cap_height : float
        Laminate cap through-thickness dimension (radial depth from inner skin).
    surface : {'upper', 'lower'}
    """

    label = "spar_cap"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_start, x_end, cap_height, surface="upper",
                 strip_clip=None):
        if surface not in ("upper", "lower"):
            raise ValueError("surface must be 'upper' or 'lower'.")

        self.surface        = surface
        self.x_start        = float(x_start)
        self.x_end          = float(x_end)
        self.cap_height     = float(cap_height)
        self.skin_thickness = float(skin_thickness)
        self._af            = airfoil_sdf
        self._strip_clip    = strip_clip

        # Inner skin reference surface (flush with OuterSkin.inner_sdf).
        # OuterSkin uses shell(af, t) with mid-surface at phi=0 → inner face at phi = -t/2.
        inner_skin = offset(airfoil_sdf, -skin_thickness / 2.0)

        cap_lam = _cap_laminate_slab(inner_skin, cap_height)

        # Half-plane clip to correct surface
        if surface == "upper":
            side_clip = _upper_half_plane()
        else:
            side_clip = _lower_half_plane()

        if strip_clip is not None:
            chord_clip = strip_clip
        else:
            chord_clip = _chordwise_clip(x_start, x_end)

        self._sdf = intersect(intersect(cap_lam, side_clip), chord_clip)
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            inner_skin_field = CallableField(
                key=f"inner_skin_{id(self)}",
                fn=offset(airfoil_sdf, -skin_thickness / 2.0),
            )
            cap_lam_expr = _cap_laminate_slab_expr(inner_skin_field, cap_height)
            side_expr = HalfPlane(nx=0.0, ny=-1.0, d=0.0) if surface == "upper" else HalfPlane(nx=0.0, ny=1.0, d=0.0)
            if strip_clip is not None:
                clip_expr = CallableField(key=f"spar_strip_{id(strip_clip)}", fn=strip_clip)
            else:
                clip_expr = Intersect(
                    HalfPlane(nx=-1.0, ny=0.0, d=float(x_start)),
                    HalfPlane(nx=1.0, ny=0.0, d=-float(x_end)),
                )
            self._expr = Intersect(Intersect(cap_lam_expr, side_expr), clip_expr)

    def __call__(self, x, y):
        return self._sdf(x, y)

    def midline_polyline(self, n=80):
        """Return the cap mid-surface polyline in the chord frame.

        The cap is a slab between ``inner_skin = 0`` and
        ``inner_skin = -cap_height``, so its midline is the airfoil
        contour offset inward by ``skin_thickness/2 + cap_height/2``,
        sliced on the requested surface. Endpoints follow the same chord
        clip the SDF uses (chord-vertical or parallel-strip — see
        :func:`_airfoil_offset_arc`).
        """
        return _airfoil_offset_arc(
            self._af,
            skin_thickness=self.skin_thickness,
            cap_height=self.cap_height,
            x_start=self.x_start,
            x_end=self.x_end,
            surface=self.surface,
            n=n,
            strip_clip=self._strip_clip,
        )

    def __repr__(self):
        return (f"SparCap(surface={self.surface!r}, "
                f"x=[{self.x_start:.3g},{self.x_end:.3g}], "
                f"cap_height={self.cap_height:.3g})")


# ---------------------------------------------------------------------------
# Continuous Spar Cap  (N-web generalisation)
# ---------------------------------------------------------------------------

class ContinuousSparCap:
    """A single curved-shell spar cap spanning the full inter-web extent.

    Identical offset-surface construction to SparCap, but the chordwise
    extent is set to span from the first to the last web position so that
    a single continuous cap laminate runs across all cells.

    Parameters
    ----------
    airfoil_sdf : callable
    skin_thickness : float
        Same convention as ``SparCap`` / ``OuterSkin.inner_sdf`` (inner mold at
        ``offset(airfoil, -skin_thickness/2)``).
    x_start, x_end : float
        Chordwise extent (typically x_webs[0] to x_webs[-1]).
    cap_height : float
        Laminate thickness (radial depth from inner skin surface).
    surface : {'upper', 'lower'}
    twist_angle : float, optional
        Section twist in radians.  If non-zero the cap SDF is rotated by
        this angle about the chordwise midpoint of ``[x_start, x_end]``.
        ``MultiCellSection`` passes ``0`` here and applies one global twist to
        all components.
    """

    label = "continuous_spar_cap"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_start, x_end, cap_height,
                 surface="upper", twist_angle=0.0,
                 strip_clip=None):
        if surface not in ("upper", "lower"):
            raise ValueError("surface must be 'upper' or 'lower'.")

        self.surface        = surface
        self.x_start        = float(x_start)
        self.x_end          = float(x_end)
        self.cap_height     = float(cap_height)
        self.skin_thickness = float(skin_thickness)
        self.twist_angle    = float(twist_angle)
        self._af            = airfoil_sdf
        self._strip_clip    = strip_clip

        inner_skin = offset(airfoil_sdf, -skin_thickness / 2.0)
        cap_lam    = _cap_laminate_slab(inner_skin, cap_height)

        side_clip  = _upper_half_plane() if surface == "upper" else _lower_half_plane()
        if strip_clip is not None:
            chord_clip = strip_clip
        else:
            chord_clip = _chordwise_clip(x_start, x_end)

        raw_sdf = intersect(intersect(cap_lam, side_clip), chord_clip)
        if abs(twist_angle) > 1e-10:
            cx = 0.5 * (self.x_start + self.x_end)
            self._sdf = rotate_field(raw_sdf, twist_angle, cx=cx, cy=0.0)
        else:
            self._sdf = raw_sdf
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            inner_skin_field = CallableField(
                key=f"cont_inner_skin_{id(self)}",
                fn=offset(airfoil_sdf, -skin_thickness / 2.0),
            )
            cap_lam_expr = _cap_laminate_slab_expr(inner_skin_field, cap_height)
            side_expr = HalfPlane(nx=0.0, ny=-1.0, d=0.0) if surface == "upper" else HalfPlane(nx=0.0, ny=1.0, d=0.0)
            if strip_clip is not None:
                clip_expr = CallableField(key=f"cont_strip_{id(strip_clip)}", fn=strip_clip)
            else:
                clip_expr = Intersect(
                    HalfPlane(nx=-1.0, ny=0.0, d=float(x_start)),
                    HalfPlane(nx=1.0, ny=0.0, d=-float(x_end)),
                )
            expr_raw = Intersect(Intersect(cap_lam_expr, side_expr), clip_expr)
            if abs(twist_angle) > 1e-10:
                cx = 0.5 * (self.x_start + self.x_end)
                self._expr = Rotate(expr_raw, angle=float(twist_angle), cx=float(cx), cy=0.0)
            else:
                self._expr = expr_raw

    def __call__(self, x, y):
        return self._sdf(x, y)

    def midline_polyline(self, n=80):
        """Return the cap mid-surface polyline in the chord frame.

        Same offset construction as :meth:`SparCap.midline_polyline`,
        including the chord-vertical / parallel-strip endpoint rule (see
        :func:`_airfoil_offset_arc`). If ``twist_angle != 0`` the polyline
        is then rotated about the chordwise midpoint of
        ``[x_start, x_end]`` to mirror the SDF's internal rotation.
        ``MultiCellSection`` typically passes ``twist_angle=0`` and
        applies one global rotation to all components.
        """
        arc = _airfoil_offset_arc(
            self._af,
            skin_thickness=self.skin_thickness,
            cap_height=self.cap_height,
            x_start=self.x_start,
            x_end=self.x_end,
            surface=self.surface,
            n=n,
            strip_clip=self._strip_clip,
        )
        if abs(self.twist_angle) > 1e-10:
            cx = 0.5 * (self.x_start + self.x_end)
            c = float(np.cos(self.twist_angle))
            s = float(np.sin(self.twist_angle))
            rel = arc - np.array([cx, 0.0])
            arc = np.column_stack(
                [c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]]
            ) + np.array([cx, 0.0])
        return arc

    def __repr__(self):
        return (f"ContinuousSparCap(surface={self.surface!r}, "
                f"x=[{self.x_start:.3g},{self.x_end:.3g}], "
                f"cap_height={self.cap_height:.3g}, "
                f"twist={np.degrees(self.twist_angle):.2f}°)")


# ---------------------------------------------------------------------------
# Shear Web
# ---------------------------------------------------------------------------

class ShearWeb:
    """Constant-thickness shear web as a shell around a capsule axis.

    The web is a capsule (segment + radius = thickness/2) clipped inside
    the inner airfoil surface.  The capsule already gives a constant-thickness
    laminate whose cross-section is always perpendicular to the web axis.

    Alignment modes
    ---------------
    "chord_normal" : web axis is vertical in the chord frame (default).
    "flapwise"     : web axis is aligned with the global flapwise (y) direction.
                     Achieved by counter-rotating the web SDF by -twist_angle.

    Parameters
    ----------
    airfoil_sdf : callable
    skin_thickness : float
    x_top, y_top : float
        Upper anchor point of the web axis in the section frame.
    x_bot, y_bot : float
        Lower anchor point of the web axis.
    thickness : float
        Web laminate thickness (full, i.e. diameter of the capsule = thickness).
    alignment : {'chord_normal', 'flapwise'}
    twist_angle : float
        Section twist in radians. Used only when alignment == 'flapwise'.
    """

    label = "shear_web"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_top, y_top, x_bot, y_bot, thickness,
                 alignment="chord_normal", twist_angle=0.0):
        if alignment not in ("chord_normal", "flapwise"):
            raise ValueError("alignment must be 'chord_normal' or 'flapwise'.")

        self.x_top      = float(x_top)
        self.y_top      = float(y_top)
        self.x_bot      = float(x_bot)
        self.y_bot      = float(y_bot)
        self.thickness  = float(thickness)
        self.alignment  = alignment
        self.twist_angle = float(twist_angle)

        r = thickness / 2.0
        _xt, _yt, _xb, _yb, _r = x_top, y_top, x_bot, y_bot, r

        def _cap_sdf(x, y):
            return sdf_capsule(x, y, _xt, _yt, _xb, _yb, _r)

        inner_af = offset(airfoil_sdf, -skin_thickness / 2.0)
        raw_sdf  = intersect(_cap_sdf, inner_af)
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            expr_raw = Intersect(
                Capsule(
                    ax=float(x_top),
                    ay=float(y_top),
                    bx=float(x_bot),
                    by=float(y_bot),
                    r=float(r),
                ),
                Offset(airfoil_expr, amount=-skin_thickness / 2.0),
            )
            self._expr = expr_raw

        # No counter-rotation here: for ``alignment="flapwise"`` the caller
        # has supplied ``(x_top, y_top, x_bot, y_bot)`` in the S-frame as the
        # inverse-rotation of the desired B-frame vertical axis, so the
        # section's global +twist rotation alone produces the intended
        # B-frame orientation while the inner-skin clip naturally rotates
        # with it (no translation/rotation mismatch with the skin).
        self._sdf = raw_sdf

    def __call__(self, x, y):
        return self._sdf(x, y)

    def midline_polyline(self, n=20):
        """Return the web midline polyline in the chord (S) frame, top → bot.

        The web SDF is a capsule of radius ``thickness/2`` built around the
        line ``(x_top, y_top) → (x_bot, y_bot)`` clipped by the chord-frame
        inner skin. For ``alignment='chord_normal'`` this S-frame axis is
        vertical at ``x = x_top = x_bot``; for ``alignment='flapwise'`` it
        is the **inverse-rotation of the intended B-frame vertical axis**
        (a tilted S-frame segment), supplied by ``MultiCellSection``. In
        both cases the same chord-frame line is the analytical midline and
        becomes the correct B-frame midline after the section's global
        +twist rotation.
        """
        n_use = max(2, int(n))
        top = np.array([self.x_top, self.y_top], dtype=float)
        bot = np.array([self.x_bot, self.y_bot], dtype=float)
        return np.linspace(top, bot, n_use)

    def __repr__(self):
        return (f"ShearWeb(x=[{self.x_top:.3g},{self.x_bot:.3g}], "
                f"alignment={self.alignment!r}, "
                f"twist={np.degrees(self.twist_angle):.2f}°, "
                f"thickness={self.thickness:.3g})")


# ---------------------------------------------------------------------------
# Sandwich Core
# ---------------------------------------------------------------------------

class SandwichCore:
    """Foam/balsa core infill: inside inner skin, outside all laminates.

    Parameters
    ----------
    airfoil_sdf : callable
    skin_thickness : float
    exclusion_sdfs : list of callable, optional
        Regions to subtract (caps, webs). Core = inner_airfoil − union(exclusions).
    x_start, x_end : float, optional
        Chordwise clip. None → full chord.
    y_min, y_max : float, optional
        Spanwise (y) clip for splitting upper/lower core zones.
    """

    label = "core"

    def __init__(self, airfoil_sdf, skin_thickness,
                 exclusion_sdfs=None,
                 x_start=None, x_end=None,
                 y_min=None, y_max=None,
                 strip_clip=None):
        inner_af = offset(airfoil_sdf, -skin_thickness / 2.0)
        sdf = inner_af

        if exclusion_sdfs:
            for ex in exclusion_sdfs:
                sdf = subtract(sdf, ex)

        if strip_clip is not None:
            sdf = intersect(sdf, strip_clip)
        else:
            # Chordwise clips: SDF convention is negative INSIDE.
            # x >= x_start: interior where x - x_start > 0 → negate for intersect
            if x_start is not None:
                sdf = intersect(sdf, lambda x, y, _xs=x_start: _xs - x)  # phi < 0 when x > xs
            if x_end is not None:
                sdf = intersect(sdf, lambda x, y, _xe=x_end: x - _xe)    # phi < 0 when x < xe
        if y_min is not None:
            sdf = intersect(sdf, lambda x, y, _ym=y_min: _ym - y)
        if y_max is not None:
            sdf = intersect(sdf, lambda x, y, _yx=y_max: y - _yx)

        self._sdf    = sdf
        self.x_start = x_start
        self.x_end   = x_end
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            expr = Offset(airfoil_expr, amount=-skin_thickness / 2.0)
            if exclusion_sdfs:
                for j, ex in enumerate(exclusion_sdfs):
                    ex_expr = getattr(ex, "_expr", None)
                    if ex_expr is None:
                        ex_expr = CallableField(key=f"core_ex_{j}_{id(ex)}", fn=ex)
                    expr = Subtract(expr, ex_expr)
            if strip_clip is not None:
                expr = Intersect(expr, CallableField(key=f"core_strip_{id(strip_clip)}", fn=strip_clip))
            else:
                if x_start is not None:
                    expr = Intersect(expr, HalfPlane(nx=-1.0, ny=0.0, d=float(x_start)))
                if x_end is not None:
                    expr = Intersect(expr, HalfPlane(nx=1.0, ny=0.0, d=-float(x_end)))
            if y_min is not None:
                expr = Intersect(expr, HalfPlane(nx=0.0, ny=-1.0, d=float(y_min)))
            if y_max is not None:
                expr = Intersect(expr, HalfPlane(nx=0.0, ny=1.0, d=-float(y_max)))
            self._expr = expr

    def __call__(self, x, y):
        return self._sdf(x, y)

    def __repr__(self):
        return f"SandwichCore(x_start={self.x_start}, x_end={self.x_end})"


# ---------------------------------------------------------------------------
# Trailing-Edge Insert
# ---------------------------------------------------------------------------

class TEInsert:
    """Trailing-edge reinforcement zone.

    Defined as the region inside the inner skin, aft of `x_start`, clipped
    by a convergence half-angle to form a natural wedge shape that follows
    the TE geometry.

    Parameters
    ----------
    airfoil_sdf : callable
    skin_thickness : float
    x_start : float
        Chordwise start of the TE insert (e.g. 0.75 * chord).
    exclusion_sdfs : list of callable, optional
        Any webs or caps to subtract (usually none in the TE zone).
    """

    label = "te_insert"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_start=0.75, exclusion_sdfs=None):
        self.x_start        = float(x_start)
        self.skin_thickness = float(skin_thickness)

        inner_af = offset(airfoil_sdf, -skin_thickness / 2.0)

        # Clip to TE region: x >= x_start AND inside inner skin
        # phi < 0 inside: for x >= xs, phi = xs - x
        sdf = intersect(inner_af,
                        lambda x, y, _xs=x_start: _xs - x)

        if exclusion_sdfs:
            for ex in exclusion_sdfs:
                sdf = subtract(sdf, ex)

        self._sdf = sdf
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            expr = Intersect(
                Offset(airfoil_expr, amount=-skin_thickness / 2.0),
                HalfPlane(nx=-1.0, ny=0.0, d=float(x_start)),
            )
            if exclusion_sdfs:
                for j, ex in enumerate(exclusion_sdfs):
                    ex_expr = getattr(ex, "_expr", None)
                    if ex_expr is None:
                        ex_expr = CallableField(key=f"ex_{j}_{id(ex)}", fn=ex)
                    expr = Subtract(expr, ex_expr)
            self._expr = expr

    def __call__(self, x, y):
        return self._sdf(x, y)

    def __repr__(self):
        return f"TEInsert(x_start={self.x_start:.3g})"


# ---------------------------------------------------------------------------
# Leading-Edge Insert
# ---------------------------------------------------------------------------

class LEInsert:
    """Leading-edge nose insert (solid glass or carbon nose block).

    Defined as a circular-cap region inside the inner skin, fore of `x_end`.
    A circular SDF centred at the LE with radius `radius` gives a smooth,
    physically meaningful LE nose shape.

    Parameters
    ----------
    airfoil_sdf : callable
    skin_thickness : float
    x_end : float
        Chordwise cutoff of the insert (e.g. 0.10 * chord).
    le_x, le_y : float
        Leading-edge coordinates (centre of the nose circle).
    radius : float
        Nose insert radius. Defaults to x_end (simple circular clip).
    """

    label = "le_insert"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_end=0.10, le_x=0.0, le_y=0.0, radius=None):
        self.x_end          = float(x_end)
        self.skin_thickness = float(skin_thickness)

        if radius is None:
            radius = x_end

        inner_af = offset(airfoil_sdf, -skin_thickness / 2.0)

        # Clip: inside inner skin AND fore of x_end AND inside nose circle
        nose_circle = lambda x, y, _lx=le_x, _ly=le_y, _r=radius: \
            sdf_circle(x, y, _lx, _ly, _r)

        # phi < 0 inside: for x <= xe, phi = x - xe
        sdf = intersect(
            intersect(inner_af, lambda x, y, _xe=x_end: x - _xe),
            nose_circle,
        )

        self._sdf = sdf
        self._expr = None
        airfoil_expr = _airfoil_expr(airfoil_sdf)
        if airfoil_expr is not None:
            self._expr = Intersect(
                Intersect(
                    Offset(airfoil_expr, amount=-skin_thickness / 2.0),
                    HalfPlane(nx=1.0, ny=0.0, d=-float(x_end)),
                ),
                Circle(cx=float(le_x), cy=float(le_y), r=float(radius)),
            )

    def __call__(self, x, y):
        return self._sdf(x, y)

    def __repr__(self):
        return f"LEInsert(x_end={self.x_end:.3g})"
