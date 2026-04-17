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

    phi_inner_skin  = offset(phi_airfoil, -skin_thickness)

    phi_cap_upper   = intersect(
        shell(phi_inner_skin, cap_height),   # band: outer = skin, inner = skin - cap_height
        half_plane(y > 0),                   # suction side only
        chordwise_clip,
    )

This guarantees constant laminate thickness everywhere along the cap and
removes any flat-face artefact from the old box-clipping approach.

Web alignment
-------------
Webs support two alignment modes:
  "chord_normal"  : web axis is perpendicular to the chord line (default).
  "flapwise"      : web axis is locked to the global flapwise (y-global) direction,
                    achieved by counter-rotating the web SDF by -twist_angle.

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chordwise_clip(x_start, x_end):
    """Return an SDF callable that is negative between x_start and x_end."""
    def _clip(x, y):
        x = np.asarray(x, dtype=float)
        left  = x - x_start   # < 0 left of x_start
        right = x_end - x     # < 0 right of x_end
        # Interior of clip box = intersect of two half-planes
        return np.maximum(-left, -right)  # standard intersect (max) of two half-planes
    return _clip


def _upper_half_plane():
    """SDF of the upper half-plane (y >= 0): phi = -y inside."""
    def _hp(x, y):
        return -np.asarray(y, dtype=float)
    return _hp


def _lower_half_plane():
    """SDF of the lower half-plane (y <= 0): phi = y inside."""
    def _hp(x, y):
        return np.asarray(y, dtype=float)
    return _hp


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
        Skin laminate thickness. Defines the inner skin reference surface.
    x_start, x_end : float
        Chordwise extent of the cap.
    cap_height : float
        Laminate cap through-thickness dimension (radial depth from inner skin).
    surface : {'upper', 'lower'}
    """

    label = "spar_cap"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_start, x_end, cap_height, surface="upper"):
        if surface not in ("upper", "lower"):
            raise ValueError("surface must be 'upper' or 'lower'.")

        self.surface        = surface
        self.x_start        = float(x_start)
        self.x_end          = float(x_end)
        self.cap_height     = float(cap_height)
        self.skin_thickness = float(skin_thickness)

        # Inner skin reference surface (the surface the cap outer face is flush against).
        # offset(f, -t): phi - (-t) = phi + t → zero-level set shifts inward by t.
        inner_skin = offset(airfoil_sdf, -skin_thickness)

        # Shell of thickness cap_height centred on the inner skin surface:
        #   outer face: inner_skin phi = 0    (at the skin boundary)
        #   inner face: inner_skin phi = -cap_height
        # shell(f, t) = |f| - t/2, so the band spans [-t/2, +t/2] around f=0
        # We want the band [0, -cap_height] → shift so band centre is at -cap_height/2
        # i.e. use offset(inner_skin, cap_height/2) as the shell axis
        cap_axis = offset(inner_skin, cap_height / 2.0)
        cap_shell = shell(cap_axis, cap_height)

        # Half-plane clip to correct surface
        if surface == "upper":
            side_clip = _upper_half_plane()
        else:
            side_clip = _lower_half_plane()

        # Chordwise clip
        chord_clip = _chordwise_clip(x_start, x_end)

        self._sdf = intersect(intersect(cap_shell, side_clip), chord_clip)

    def __call__(self, x, y):
        return self._sdf(x, y)

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
    x_start, x_end : float
        Chordwise extent (typically x_webs[0] to x_webs[-1]).
    cap_height : float
        Laminate thickness (radial depth from inner skin surface).
    surface : {'upper', 'lower'}
    twist_angle : float, optional
        Section twist in radians.  If non-zero the cap SDF is rotated by
        this angle so the cap stays aligned with the structural frame.
    """

    label = "continuous_spar_cap"

    def __init__(self, airfoil_sdf, skin_thickness,
                 x_start, x_end, cap_height,
                 surface="upper", twist_angle=0.0):
        if surface not in ("upper", "lower"):
            raise ValueError("surface must be 'upper' or 'lower'.")

        self.surface        = surface
        self.x_start        = float(x_start)
        self.x_end          = float(x_end)
        self.cap_height     = float(cap_height)
        self.skin_thickness = float(skin_thickness)
        self.twist_angle    = float(twist_angle)

        inner_skin = offset(airfoil_sdf, -skin_thickness)
        cap_axis   = offset(inner_skin, cap_height / 2.0)
        cap_shell  = shell(cap_axis, cap_height)

        side_clip  = _upper_half_plane() if surface == "upper" else _lower_half_plane()
        chord_clip = _chordwise_clip(x_start, x_end)

        raw_sdf = intersect(intersect(cap_shell, side_clip), chord_clip)

        # Apply twist rotation if needed
        if abs(twist_angle) > 1e-10:
            self._sdf = rotate_field(raw_sdf, twist_angle)
        else:
            self._sdf = raw_sdf

    def __call__(self, x, y):
        return self._sdf(x, y)

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

        inner_af = offset(airfoil_sdf, -skin_thickness)
        raw_sdf  = intersect(_cap_sdf, inner_af)

        if alignment == "flapwise" and abs(twist_angle) > 1e-10:
            # Counter-rotate by -twist_angle to stay flapwise-aligned
            cx = 0.5 * (x_top + x_bot)
            cy = 0.5 * (y_top + y_bot)
            self._sdf = rotate_field(raw_sdf, -twist_angle, cx=cx, cy=cy)
        else:
            self._sdf = raw_sdf

    def __call__(self, x, y):
        return self._sdf(x, y)

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
                 y_min=None, y_max=None):
        inner_af = offset(airfoil_sdf, -skin_thickness)
        sdf = inner_af

        if exclusion_sdfs:
            for ex in exclusion_sdfs:
                sdf = subtract(sdf, ex)

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

        inner_af = offset(airfoil_sdf, -skin_thickness)

        # Clip to TE region: x >= x_start AND inside inner skin
        # phi < 0 inside: for x >= xs, phi = xs - x
        sdf = intersect(inner_af,
                        lambda x, y, _xs=x_start: _xs - x)

        if exclusion_sdfs:
            for ex in exclusion_sdfs:
                sdf = subtract(sdf, ex)

        self._sdf = sdf

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

        inner_af = offset(airfoil_sdf, -skin_thickness)

        # Clip: inside inner skin AND fore of x_end AND inside nose circle
        nose_circle = lambda x, y, _lx=le_x, _ly=le_y, _r=radius: \
            sdf_circle(x, y, _lx, _ly, _r)

        # phi < 0 inside: for x <= xe, phi = x - xe
        sdf = intersect(
            intersect(inner_af, lambda x, y, _xe=x_end: x - _xe),
            nose_circle,
        )

        self._sdf = sdf

    def __call__(self, x, y):
        return self._sdf(x, y)

    def __repr__(self):
        return f"LEInsert(x_end={self.x_end:.3g})"
