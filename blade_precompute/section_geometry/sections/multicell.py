"""
sections.multicell
==================
MultiCellSection — generalised N-web blade section with N+1 sandwich-core bays.

Topology
--------
Given N web x-positions [x_0, x_1, …, x_{N-1}]:

    LE  |  x_0  |  x_1  | … |  x_{N-1}  |  TE
        [ bay_0 ][ bay_1 ] … [ bay_N ]

    - N ShearWeb objects (one per x-position)
    - 1 ContinuousSparCap upper + 1 lower (chord clip x_0 … x_{N-1}, then CSG
      subtract the web laminates so caps meet web walls flush)
    - N+1 SandwichCore objects: LE–first web, inter-web strips, last web–TE
    - Optional TEInsert  (aft of a chord station; separate SDF from foam cores)
    - Optional LEInsert  (nose region; separate SDF from foam cores)

    End-bay ``SandwichCore`` regions do not subtract ``le_insert`` / ``te_insert``;
    subtract those in a later revision if area integrals must be mutually exclusive.

Component labels
----------------
    "web_0" … "web_{N-1}"
    "spar_cap_upper", "spar_cap_lower"
    "core_0" … "core_N"   (N+1 labels when cores are built)
    "te_insert"   (if enabled)
    "le_insert"   (if enabled)
    "outer_skin"

Web alignment
-------------
Each web independently supports:
    "chord_normal" : web axis perpendicular to chord line.
    "flapwise"     : web axis locked to global flapwise direction.
                     Web SDF counter-rotated by -twist_angle.

Web anchor points
-----------------
By default web anchors are computed automatically by querying the upper and
lower inner skin y-coordinates at each web x-position (ray-cast).  The user
may also supply explicit (y_top, y_bot) per web via web_y_coords.

Usage
-----
    from geometry.airfoil import AirfoilSDF
    from sections.multicell import MultiCellSection

    af  = AirfoilSDF.from_naca("2412", chord=1.0)
    bsg = MultiCellSection(
        airfoil_sdf       = af,
        web_x_positions   = [0.20, 0.35, 0.50],
        web_thickness     = 0.004,
        web_alignment     = "chord_normal",
        cap_height        = (0.012, 0.010),     # (upper, lower)
        skin_thickness    = 0.003,
        twist_angle       = 0.0,
        te_insert_x       = 0.75,
        le_insert_x       = 0.10,
        core_enabled      = True,
    )
    phi_web0 = bsg["web_0"](X, Y)
"""

import numpy as np
from .subcomponents import (
    OuterSkin,
    ContinuousSparCap,
    ShearWeb,
    SandwichCore,
    SparCap,
    TEInsert,
    LEInsert,
    _parallel_web_half_gt,
    _parallel_web_half_lt,
    _parallel_web_strip_clip,
    flapwise_web_normal,
    web_station_projection,
)
from ..geometry.csg import offset, subtract, intersect, union, union_all
from ..geometry.section_axes import max_thickness_chord_x, pitch_axis_x_from_le
from ..geometry.transforms import rotate_field
from ..laminate_thickness_limits import (
    clamp_skin_thickness_m,
    clamp_spar_laminate_thickness_m,
    clamp_web_thickness_m,
)
from ..structural import parse_fixed_cap_anchor, parse_structural_family


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inner_y_at_x(
    airfoil_sdf,
    skin_thickness,
    x_query,
    y_search=None,
    n_samples=240,
    *,
    phi_inner=None,
    bisection_iters=10,
):
    """Estimate the upper and lower inner-skin y-coordinates at a given x.

    Uses a dense 1-D scan along y at the queried x, finding the sign change
    of phi_inner_skin(x_query, y), with phi_inner_skin = offset(airfoil, -t/2)
    to match ``OuterSkin.inner_sdf`` (mid-surface shell model).

    Parameters
    ----------
    y_search : tuple (y_min, y_max) or None
        y search range.  Defaults to ±0.5 (suitable for unit-chord sections).

    Returns
    -------
    y_top, y_bot : float
        Upper and lower inner skin y intercepts.
    """
    if y_search is None:
        y_search = (-0.5, 0.5)

    y_vals = np.linspace(y_search[0], y_search[1], int(n_samples))
    x_vals = np.full_like(y_vals, x_query)

    if phi_inner is None:
        phi_inner = offset(airfoil_sdf, -skin_thickness / 2.0)
    phi_vals  = phi_inner(x_vals, y_vals)

    # Find sign changes (inner skin crossings)
    crossings = np.where(np.diff(np.sign(phi_vals)))[0]

    if len(crossings) < 2:
        raise ValueError(
            "Could not determine inner skin intercepts at "
            f"x={x_query:.6g} within y_search={tuple(y_search)}. "
            "Provide explicit web_y_coords or widen y_search."
        )

    # Interpolate to find exact crossing y
    def _interp_crossing(idx):
        y0, y1 = float(y_vals[idx]), float(y_vals[idx + 1])
        p0, p1 = float(phi_vals[idx]), float(phi_vals[idx + 1])
        # Refine bracket by bisection when there is a robust sign change.
        if p0 == 0.0:
            return y0
        if p1 == 0.0:
            return y1
        if p0 * p1 < 0.0:
            for _ in range(int(bisection_iters)):
                ym = 0.5 * (y0 + y1)
                pm = float(phi_inner(np.array([x_query]), np.array([ym]))[0])
                if pm == 0.0:
                    return ym
                if p0 * pm < 0.0:
                    y1, p1 = ym, pm
                else:
                    y0, p0 = ym, pm
        return y0 - p0 * (y1 - y0) / (p1 - p0 + 1e-30)

    y_crossings = sorted([_interp_crossing(i) for i in crossings])
    y_bot = y_crossings[0]
    y_top = y_crossings[-1]
    return y_top, y_bot


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiCellSection:
    """N-web generalised blade section (N+1 structural bays, N+1 foam cores when enabled).

    Parameters
    ----------
    airfoil_sdf : callable (x, y) → ndarray
        Outer airfoil boundary SDF (e.g. AirfoilSDF instance).
    web_x_positions : list of float
        Chordwise x-coordinates of each web (N values, must be sorted).
    web_thickness : float or list of float
        Web laminate thickness. Scalar → uniform across all webs.
        Values below ``MIN_REALISTIC_WEB_LAMINATE_THICKNESS_M`` are raised
        (see :mod:`blade_precompute.section_geometry.laminate_thickness_limits`).
    web_alignment : str or list of str
        "chord_normal" or "flapwise", per web or uniform.
    cap_height : float or tuple (float, float)
        Spar cap laminate height (depth from inner skin).
        Scalar → same for upper and lower.
        Tuple → (upper_height, lower_height).
        Values below ``MIN_REALISTIC_SPAR_LAMINATE_THICKNESS_M`` are raised
        (see :mod:`blade_precompute.section_geometry.laminate_thickness_limits`).
    skin_thickness : float
        Outer skin laminate thickness. Values below
        ``MIN_REALISTIC_SKIN_LAMINATE_THICKNESS_M`` are raised
        (see :mod:`blade_precompute.section_geometry.laminate_thickness_limits`).
    twist_angle : float
        Section twist angle in radians (CCW positive, LE rotating upward).
        Applied to flapwise-aligned webs and optionally to the spar caps.
    web_y_coords : list of (y_top, y_bot) or None
        Explicit web anchor y-coordinates per web.  If None, auto-computed
        by ray-casting the inner skin surface.
    te_insert_x : float or None
        x-start of the trailing-edge insert.  None → no TE insert.
    le_insert_x : float or None
        x-end of the leading-edge insert.  None → no LE insert.
    le_radius : float or None
        Nose-circle radius for the LE insert.  Defaults to le_insert_x.
    core_enabled : bool
        Whether to build sandwich core regions.
    y_search : tuple (y_min, y_max) or None
        y-range for web anchor auto-detection.
    structural_family : str
        ``A`` no caps, ``B`` single band, ``C`` discrete per web, ``D`` box (default).
    fixed_cap_anchor : str
        For ``B``: ``pitching`` or ``max_thickness``.
    pitch_fraction_of_chord_from_le : float
        For ``B`` + pitching anchor (default 1/3).
    fixed_cap_chord_half_width : float or None
        For ``B``; default ``0.05 * chord``.
    discrete_cap_chord_half_width : float or None
        For ``C``; default ``0.04 * chord``.
    """

    def __init__(
        self,
        airfoil_sdf,
        web_x_positions,
        web_thickness    = 0.004,
        web_alignment    = "chord_normal",
        cap_height       = 0.012,
        skin_thickness   = 0.003,
        twist_angle      = 0.0,
        web_y_coords     = None,
        te_insert_x      = None,
        le_insert_x      = None,
        le_radius        = None,
        core_enabled     = True,
        y_search         = None,
        structural_family              = "D",
        fixed_cap_anchor               = "pitching",
        pitch_fraction_of_chord_from_le = 1.0 / 3.0,
        fixed_cap_chord_half_width     = None,
        discrete_cap_chord_half_width  = None,
    ):
        # ------------------------------------------------------------------
        # Normalise inputs
        # ------------------------------------------------------------------
        xs = sorted(float(x) for x in web_x_positions)
        if len(xs) < 1:
            raise ValueError("At least one web x-position is required.")
        N = len(xs)

        # Web thickness per web
        if np.isscalar(web_thickness):
            web_thicknesses = [float(web_thickness)] * N
        else:
            web_thicknesses = [float(t) for t in web_thickness]
            if len(web_thicknesses) != N:
                raise ValueError(f"web_thickness must be scalar or length {N}.")

        web_thicknesses = [clamp_web_thickness_m(t) for t in web_thicknesses]
        skin_thickness = clamp_skin_thickness_m(skin_thickness)

        # Web alignment per web
        if isinstance(web_alignment, str):
            web_alignments = [web_alignment] * N
        else:
            web_alignments = list(web_alignment)
            if len(web_alignments) != N:
                raise ValueError(f"web_alignment must be scalar or length {N}.")

        # Cap heights (upper, lower)
        if np.isscalar(cap_height):
            cap_h_upper = cap_h_lower = float(cap_height)
        else:
            cap_h_upper, cap_h_lower = float(cap_height[0]), float(cap_height[1])
        cap_h_upper = clamp_spar_laminate_thickness_m(cap_h_upper)
        cap_h_lower = clamp_spar_laminate_thickness_m(cap_h_lower)

        self._af          = airfoil_sdf
        self._skin_t      = float(skin_thickness)
        self._twist       = float(twist_angle)
        self._structural_family = parse_structural_family(structural_family)
        if self._structural_family in ("B", "C"):
            from ..geometry.airfoil import AirfoilSDF as _AirfoilSDF

            if not isinstance(airfoil_sdf, _AirfoilSDF):
                raise TypeError(
                    "structural_family 'B' or 'C' requires airfoil_sdf to be "
                    "AirfoilSDF (pitching / max-thickness anchors)."
                )

        self._components  = {}   # label → callable

        _chord = float(getattr(airfoil_sdf, "chord", 1.0))
        _f_anchor = (
            parse_fixed_cap_anchor(fixed_cap_anchor)
            if self._structural_family == "B"
            else None
        )
        _pitch_fr = float(pitch_fraction_of_chord_from_le)
        _fhw = (
            float(fixed_cap_chord_half_width)
            if fixed_cap_chord_half_width is not None
            else 0.05 * _chord
        )
        _dhw = (
            float(discrete_cap_chord_half_width)
            if discrete_cap_chord_half_width is not None
            else 0.04 * _chord
        )

        # ------------------------------------------------------------------
        # Outer skin
        # ------------------------------------------------------------------
        skin = OuterSkin(airfoil_sdf, skin_thickness)
        self._components["outer_skin"] = skin
        # Mold surfaces for JSON export (same twist as components; not the shell solid).
        skin_outer_sdf = skin.outer_sdf
        skin_inner_sdf = skin.inner_sdf

        # ------------------------------------------------------------------
        # Web anchor points
        # ------------------------------------------------------------------
        phi_inner_ref = offset(airfoil_sdf, -skin_thickness / 2.0)
        if web_y_coords is None:
            anchors = []
            for x in xs:
                y_top, y_bot = _inner_y_at_x(
                    airfoil_sdf,
                    skin_thickness,
                    x,
                    y_search=y_search,
                    phi_inner=phi_inner_ref,
                )
                anchors.append((y_top, y_bot))
        else:
            anchors = [(float(yt), float(yb)) for yt, yb in web_y_coords]
            if len(anchors) != N:
                raise ValueError(f"web_y_coords must have {N} entries.")

        all_flapwise = all(a == "flapwise" for a in web_alignments)
        if all_flapwise:
            _nx, _ny = flapwise_web_normal(float(twist_angle))
            c_station = [
                web_station_projection(_nx, _ny, xs[i], 0.5 * (anchors[i][0] + anchors[i][1]))
                for i in range(N)
            ]
        else:
            _nx = _ny = None
            c_station = None

        # ------------------------------------------------------------------
        # Flapwise web S-frame inputs from B-frame intent
        # ------------------------------------------------------------------
        # For ``alignment="flapwise"``, the web SDF must be vertical in the
        # global B-frame at ``x_b = x_b_web`` and clipped by the **B-frame**
        # inner skin (the same surface that bounds the cap and outer skin
        # after the section's global +twist rotation). The previous
        # implementation built the capsule + clip in the chord (S) frame and
        # counter-rotated the web SDF so that the S→B chain ended up as a
        # pure translation — but a translation does not equal a rotation, so
        # the translated S-frame ``inner_af`` clip landed off the actual
        # B-frame inner skin and the web fell short of the skin/cap (gaps
        # of 5–25 mm at large twist; see H20).
        #
        # Fix: feed ``ShearWeb`` S-frame inputs that are the **inverse
        # rotation** of the desired B-frame vertical axis. The S-frame
        # capsule axis is therefore tilted, and the chord-frame
        # ``inner_af`` clip used inside ``ShearWeb`` is the same surface
        # that — after the section's global +twist rotation — becomes the
        # B-frame inner skin. The web then meets the skin/cap exactly.
        cos_t = float(np.cos(float(twist_angle)))
        sin_t = float(np.sin(float(twist_angle)))
        flapwise_present = any(a == "flapwise" for a in web_alignments)
        needs_b_frame_lookup = flapwise_present and abs(float(twist_angle)) > 1e-10
        phi_inner_b = None
        y_search_b = y_search
        if needs_b_frame_lookup:
            phi_inner_b = rotate_field(phi_inner_ref, float(twist_angle))
            verts = getattr(airfoil_sdf, "vertices", None)
            if verts is not None:
                arr = np.asarray(verts, dtype=float)
                y_b_vals = sin_t * arr[:, 0] + cos_t * arr[:, 1]
                y_search_b = (
                    float(y_b_vals.min()) - 0.1,
                    float(y_b_vals.max()) + 0.1,
                )
            elif y_search_b is None:
                y_search_b = (-1.5, 1.5)

        # ------------------------------------------------------------------
        # Shear webs
        # ------------------------------------------------------------------
        web_sdfs = []
        for i, (x, t, align, (y_top, y_bot)) in enumerate(
            zip(xs, web_thicknesses, web_alignments, anchors)
        ):
            if align == "flapwise" and needs_b_frame_lookup:
                # B-frame x of this flapwise web (preserved exactly from the
                # legacy translation-equivalent placement): the chord-frame
                # web at ``(x, mid_y)`` lands at ``x_b = cos·x − sin·mid_y``
                # after the global +twist rotation. We use the chord-frame
                # auto-anchor mid (or user-provided y_top/y_bot mid) only to
                # compute ``x_b_web``; the web's actual extent is then
                # dictated by the B-frame inner skin at ``x_b_web``.
                mid_y_init = 0.5 * (y_top + y_bot)
                x_b_web = cos_t * x - sin_t * mid_y_init
                y_b_top, y_b_bot = _inner_y_at_x(
                    None,
                    None,
                    x_b_web,
                    y_search=y_search_b,
                    phi_inner=phi_inner_b,
                )
                # Inverse-rotate (x_b_web, y_b_top) and (x_b_web, y_b_bot)
                # back to the S-frame: R(-twist)·(x_b, y_b).
                x_top_S = cos_t * x_b_web + sin_t * y_b_top
                y_top_S = -sin_t * x_b_web + cos_t * y_b_top
                x_bot_S = cos_t * x_b_web + sin_t * y_b_bot
                y_bot_S = -sin_t * x_b_web + cos_t * y_b_bot
            else:
                x_top_S = x
                x_bot_S = x
                y_top_S = y_top
                y_bot_S = y_bot

            web = ShearWeb(
                airfoil_sdf    = airfoil_sdf,
                skin_thickness = skin_thickness,
                x_top          = x_top_S,
                y_top          = y_top_S,
                x_bot          = x_bot_S,
                y_bot          = y_bot_S,
                thickness      = t,
                alignment      = align,
                twist_angle    = float(twist_angle),
            )
            label = f"web_{i}"
            self._components[label] = web
            web_sdfs.append(web)

        # ------------------------------------------------------------------
        # Spar caps (structural family A/B/C/D)
        # ------------------------------------------------------------------
        sf = self._structural_family
        cap_upper = None
        cap_lower = None
        # Track the un-subtracted cap subcomponents so external consumers
        # (e.g. shell-stage adapters) can recover their analytical
        # x_start/x_end/cap_height/surface attributes after the web-volume
        # subtraction step below replaces the entries in ``self._components``
        # with opaque CSG callables.
        spar_cap_components_unrotated: dict[str, Any] = {}

        if sf == "D":
            x_cap_start = xs[0]
            x_cap_end = xs[-1]
            cap_strip = None
            if all_flapwise and c_station is not None:
                d_stat = abs(float(c_station[0]) - float(c_station[-1]))
                chord_scale = float(getattr(airfoil_sdf, "chord", 1.0))
                # Single flapwise web (N == 1): first/last ``c_station`` coincide,
                # so the parallel-line strip has zero width, ``strip_clip`` never
                # goes negative on sampled cap arcs, and ``midline_polyline`` can
                # return empty (breaks shell mesh). Fall back to chordwise clip.
                if d_stat > 1e-9 * max(chord_scale, 1e-9):
                    cap_strip = _parallel_web_strip_clip(
                        _nx, _ny, c_station[0], c_station[-1]
                    )
            cap_upper = ContinuousSparCap(
                airfoil_sdf=airfoil_sdf,
                skin_thickness=skin_thickness,
                x_start=x_cap_start,
                x_end=x_cap_end,
                cap_height=cap_h_upper,
                surface="upper",
                twist_angle=0.0,
                strip_clip=cap_strip,
            )
            cap_lower = ContinuousSparCap(
                airfoil_sdf=airfoil_sdf,
                skin_thickness=skin_thickness,
                x_start=x_cap_start,
                x_end=x_cap_end,
                cap_height=cap_h_lower,
                surface="lower",
                twist_angle=0.0,
                strip_clip=cap_strip,
            )
            spar_cap_components_unrotated["spar_cap_upper"] = cap_upper
            spar_cap_components_unrotated["spar_cap_lower"] = cap_lower
            if web_sdfs:
                _wu = union_all(web_sdfs)
                cap_upper = subtract(cap_upper, _wu)
                cap_lower = subtract(cap_lower, _wu)
        elif sf == "B":
            assert _f_anchor is not None
            if _f_anchor == "pitching":
                x_c = pitch_axis_x_from_le(airfoil_sdf, _pitch_fr)
            else:
                x_c = max_thickness_chord_x(airfoil_sdf)
            x_lo = x_c - _fhw
            x_hi = x_c + _fhw
            strip_b = None
            if all_flapwise and c_station is not None:
                y_tc, y_bc = _inner_y_at_x(
                    airfoil_sdf,
                    skin_thickness,
                    x_c,
                    y_search=y_search,
                    phi_inner=phi_inner_ref,
                )
                cy_c = 0.5 * (y_tc + y_bc)
                c_lo = web_station_projection(_nx, _ny, x_lo, cy_c)
                c_hi = web_station_projection(_nx, _ny, x_hi, cy_c)
                strip_b = _parallel_web_strip_clip(_nx, _ny, c_lo, c_hi)
            cap_upper = SparCap(
                airfoil_sdf,
                skin_thickness,
                x_lo,
                x_hi,
                cap_h_upper,
                "upper",
                strip_clip=strip_b,
            )
            cap_lower = SparCap(
                airfoil_sdf,
                skin_thickness,
                x_lo,
                x_hi,
                cap_h_lower,
                "lower",
                strip_clip=strip_b,
            )
            spar_cap_components_unrotated["spar_cap_upper"] = cap_upper
            spar_cap_components_unrotated["spar_cap_lower"] = cap_lower
            if web_sdfs:
                _wu = union_all(web_sdfs)
                cap_upper = subtract(cap_upper, _wu)
                cap_lower = subtract(cap_lower, _wu)
        elif sf == "C":
            caps_u = []
            caps_l = []
            for i in range(N):
                x_lo = xs[i] - _dhw
                x_hi = xs[i] + _dhw
                strip_c = None
                if all_flapwise and c_station is not None:
                    cy_i = 0.5 * (anchors[i][0] + anchors[i][1])
                    c_lo = web_station_projection(_nx, _ny, x_lo, cy_i)
                    c_hi = web_station_projection(_nx, _ny, x_hi, cy_i)
                    strip_c = _parallel_web_strip_clip(_nx, _ny, c_lo, c_hi)
                caps_u.append(
                    SparCap(
                        airfoil_sdf,
                        skin_thickness,
                        x_lo,
                        x_hi,
                        cap_h_upper,
                        "upper",
                        strip_clip=strip_c,
                    )
                )
                caps_l.append(
                    SparCap(
                        airfoil_sdf,
                        skin_thickness,
                        x_lo,
                        x_hi,
                        cap_h_lower,
                        "lower",
                        strip_clip=strip_c,
                    )
                )
            for i, (cu, cl) in enumerate(zip(caps_u, caps_l)):
                spar_cap_components_unrotated[f"spar_cap_upper_{i}"] = cu
                spar_cap_components_unrotated[f"spar_cap_lower_{i}"] = cl
            cap_upper = union_all(caps_u)
            cap_lower = union_all(caps_l)
            if web_sdfs:
                _wu = union_all(web_sdfs)
                cap_upper = subtract(cap_upper, _wu)
                cap_lower = subtract(cap_lower, _wu)
        # sf == "A": no spar caps

        if sf != "A":
            self._components["spar_cap_upper"] = cap_upper
            self._components["spar_cap_lower"] = cap_lower

        # Expose the un-subtracted spar-cap subcomponents for analytical
        # consumers (shell-stage adapter). Keys mirror the labels in
        # ``self._components`` for sf D, with per-web suffixes for sf C.
        self._spar_cap_components_unrotated = dict(spar_cap_components_unrotated)

        laminates_for_core = (
            list(web_sdfs) if sf == "A" else [cap_upper, cap_lower] + web_sdfs
        )

        # ------------------------------------------------------------------
        # Sandwich cores: N+1 bays (LE–web_0, inter-web, web_{N-1}–TE)
        # ------------------------------------------------------------------
        if core_enabled and N >= 1:
            laminates = laminates_for_core
            n_bays = N + 1
            for k in range(n_bays):
                if all_flapwise and c_station is not None:
                    if k == 0:
                        sclip = _parallel_web_half_lt(_nx, _ny, c_station[0])
                    elif k == n_bays - 1:
                        sclip = _parallel_web_half_gt(_nx, _ny, c_station[-1])
                    else:
                        sclip = _parallel_web_strip_clip(
                            _nx, _ny, c_station[k - 1], c_station[k]
                        )
                    core = SandwichCore(
                        airfoil_sdf    = airfoil_sdf,
                        skin_thickness = skin_thickness,
                        exclusion_sdfs = laminates,
                        strip_clip     = sclip,
                    )
                else:
                    if k == 0:
                        x_s, x_e = None, xs[0]
                    elif k == n_bays - 1:
                        x_s, x_e = xs[-1], None
                    else:
                        x_s, x_e = xs[k - 1], xs[k]
                    core = SandwichCore(
                        airfoil_sdf    = airfoil_sdf,
                        skin_thickness = skin_thickness,
                        exclusion_sdfs = laminates,
                        x_start        = x_s,
                        x_end          = x_e,
                    )
                self._components[f"core_{k}"] = core

        # ------------------------------------------------------------------
        # TE insert
        # ------------------------------------------------------------------
        if te_insert_x is not None:
            te = TEInsert(
                airfoil_sdf    = airfoil_sdf,
                skin_thickness = skin_thickness,
                x_start        = float(te_insert_x),
            )
            self._components["te_insert"] = te

        # ------------------------------------------------------------------
        # LE insert
        # ------------------------------------------------------------------
        if le_insert_x is not None:
            le_x = 0.0   # default LE at x=0 for unit chord
            try:
                le_x = float(airfoil_sdf.leading_edge[0])
            except AttributeError:
                pass
            le = LEInsert(
                airfoil_sdf    = airfoil_sdf,
                skin_thickness = skin_thickness,
                x_end          = float(le_insert_x),
                le_x           = le_x,
                le_y           = 0.0,
                radius         = le_radius,
            )
            self._components["le_insert"] = le

        # ------------------------------------------------------------------
        # S → B frame: rotate all components to the blade (physical) frame.
        # Every component was built in the chord-aligned S-frame for
        # geometric convenience.  The flapwise-aligned web SDF already has
        # its own internal counter-rotation of -twist_angle so that after
        # this global +twist_angle rotation it ends up vertical (flapwise)
        # in the B-frame — the correct physical behaviour.
        # ------------------------------------------------------------------
        self._components_unrotated = dict(self._components)
        self._skin_outer_boundary_unrotated_sdf = skin_outer_sdf
        self._skin_inner_boundary_unrotated_sdf = skin_inner_sdf
        if abs(self._twist) > 1e-10:
            skin_outer_sdf = rotate_field(skin_outer_sdf, self._twist)
            skin_inner_sdf = rotate_field(skin_inner_sdf, self._twist)
            self._components = {
                lbl: rotate_field(sdf, self._twist)
                for lbl, sdf in self._components.items()
            }
        self._skin_outer_boundary_sdf = skin_outer_sdf
        self._skin_inner_boundary_sdf = skin_inner_sdf

    # ------------------------------------------------------------------
    # Access interface (mirrors BladeSectionGeometry)
    # ------------------------------------------------------------------

    def __getitem__(self, label):
        if label not in self._components:
            raise KeyError(
                f"Unknown component '{label}'. Available: {self.labels}"
            )
        return self._components[label]

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)

    @property
    def labels(self):
        return list(self._components.keys())

    @property
    def airfoil(self):
        return self._af

    @property
    def structural_family(self):
        """Structural spar family: ``A``, ``B``, ``C``, or ``D``."""
        return self._structural_family

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, label, x, y):
        return self._components[label](x, y)

    def eval_all(self, x, y):
        return {label: comp(x, y) for label, comp in self._components.items()}

    def eval_union(self, x, y):
        combined = union_all(list(self._components.values()))
        return combined(x, y)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def d_spar(cls, airfoil_sdf, web_x=0.25, **kwargs):
        """Single-web D-spar (the classical tidal/wind blade layout)."""
        return cls(airfoil_sdf, web_x_positions=[web_x], **kwargs)

    @classmethod
    def twin_web(cls, airfoil_sdf, x_fore=0.20, x_aft=0.50, **kwargs):
        """Two webs, three-core bay partition (LE–web, inter-web, web–TE), box-spar."""
        return cls(airfoil_sdf, web_x_positions=[x_fore, x_aft], **kwargs)

    @classmethod
    def torsion_box(cls, airfoil_sdf, x_fore=0.15, x_mid=0.35, x_aft=0.55, **kwargs):
        """Three webs, four full-span foam bays."""
        return cls(airfoil_sdf,
                   web_x_positions=[x_fore, x_mid, x_aft], **kwargs)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def n_webs(self):
        return sum(1 for k in self._components if k.startswith("web_"))

    @property
    def n_cells(self):
        """Structural shear / foam bay count: ``n_webs + 1`` (matches system layout ``n_cells``).

        Reported for topology; independent of whether ``core_enabled`` built the SDFs.
        """
        return self.n_webs + 1

    def __repr__(self):
        return (
            f"MultiCellSection("
            f"structural_family={self._structural_family!r}, "
            f"n_webs={self.n_webs}, "
            f"n_cells={self.n_cells}, "
            f"twist={np.degrees(self._twist):.1f}°, "
            f"components={self.labels})"
        )
