"""
sections.multicell
==================
MultiCellSection — generalised N-web, (N-1)-cell blade section builder.

Topology
--------
Given N web x-positions [x_0, x_1, …, x_{N-1}]:

    LE  |  x_0  |  x_1  | … |  x_{N-1}  |  TE
        [ cell_0 ][ cell_1 ]   [ cell_{N-2} ]

    - N ShearWeb objects (one per x-position)
    - 1 ContinuousSparCap upper + 1 lower (spanning x_0 → x_{N-1})
    - N-1 SandwichCore objects (cell_i bounded by web_i and web_{i+1})
    - Optional TEInsert  (aft of x_{N-1})
    - Optional LEInsert  (fore of x_0)

Component labels
----------------
    "web_0" … "web_{N-1}"
    "spar_cap_upper", "spar_cap_lower"
    "core_0" … "core_{N-2}"
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
    TEInsert,
    LEInsert,
)
from ..geometry.csg import offset, subtract, intersect, union, union_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inner_y_at_x(airfoil_sdf, skin_thickness, x_query,
                  y_search=None, n_samples=500):
    """Estimate the upper and lower inner-skin y-coordinates at a given x.

    Uses a dense 1-D scan along y at the queried x, finding the sign change
    of phi_inner_skin(x_query, y).

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

    y_vals = np.linspace(y_search[0], y_search[1], n_samples)
    x_vals = np.full_like(y_vals, x_query)

    phi_inner = offset(airfoil_sdf, -skin_thickness)
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
        y0, y1 = y_vals[idx], y_vals[idx + 1]
        p0, p1 = phi_vals[idx], phi_vals[idx + 1]
        return y0 - p0 * (y1 - y0) / (p1 - p0 + 1e-30)

    y_crossings = sorted([_interp_crossing(i) for i in crossings])
    y_bot = y_crossings[0]
    y_top = y_crossings[-1]
    return y_top, y_bot


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiCellSection:
    """N-web, (N-1)-cell generalised blade section.

    Parameters
    ----------
    airfoil_sdf : callable (x, y) → ndarray
        Outer airfoil boundary SDF (e.g. AirfoilSDF instance).
    web_x_positions : list of float
        Chordwise x-coordinates of each web (N values, must be sorted).
    web_thickness : float or list of float
        Web laminate thickness. Scalar → uniform across all webs.
    web_alignment : str or list of str
        "chord_normal" or "flapwise", per web or uniform.
    cap_height : float or tuple (float, float)
        Spar cap laminate height (depth from inner skin).
        Scalar → same for upper and lower.
        Tuple → (upper_height, lower_height).
    skin_thickness : float
        Outer skin laminate thickness.
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

        self._af          = airfoil_sdf
        self._skin_t      = float(skin_thickness)
        self._twist       = float(twist_angle)
        self._components  = {}   # label → callable

        # ------------------------------------------------------------------
        # Outer skin
        # ------------------------------------------------------------------
        skin = OuterSkin(airfoil_sdf, skin_thickness)
        self._components["outer_skin"] = skin

        # ------------------------------------------------------------------
        # Web anchor points
        # ------------------------------------------------------------------
        if web_y_coords is None:
            anchors = []
            for x in xs:
                y_top, y_bot = _inner_y_at_x(
                    airfoil_sdf, skin_thickness, x, y_search=y_search
                )
                anchors.append((y_top, y_bot))
        else:
            anchors = [(float(yt), float(yb)) for yt, yb in web_y_coords]
            if len(anchors) != N:
                raise ValueError(f"web_y_coords must have {N} entries.")

        # ------------------------------------------------------------------
        # Shear webs
        # ------------------------------------------------------------------
        web_sdfs = []
        for i, (x, t, align, (y_top, y_bot)) in enumerate(
            zip(xs, web_thicknesses, web_alignments, anchors)
        ):
            web = ShearWeb(
                airfoil_sdf    = airfoil_sdf,
                skin_thickness = skin_thickness,
                x_top          = x,
                y_top          = y_top,
                x_bot          = x,
                y_bot          = y_bot,
                thickness      = t,
                alignment      = align,
                twist_angle    = float(twist_angle),
            )
            label = f"web_{i}"
            self._components[label] = web
            web_sdfs.append(web)

        # ------------------------------------------------------------------
        # Continuous spar caps (upper + lower)
        # ------------------------------------------------------------------
        x_cap_start = xs[0]
        x_cap_end   = xs[-1]

        cap_upper = ContinuousSparCap(
            airfoil_sdf    = airfoil_sdf,
            skin_thickness = skin_thickness,
            x_start        = x_cap_start,
            x_end          = x_cap_end,
            cap_height     = cap_h_upper,
            surface        = "upper",
            twist_angle    = float(twist_angle),
        )
        cap_lower = ContinuousSparCap(
            airfoil_sdf    = airfoil_sdf,
            skin_thickness = skin_thickness,
            x_start        = x_cap_start,
            x_end          = x_cap_end,
            cap_height     = cap_h_lower,
            surface        = "lower",
            twist_angle    = float(twist_angle),
        )
        self._components["spar_cap_upper"] = cap_upper
        self._components["spar_cap_lower"] = cap_lower

        # ------------------------------------------------------------------
        # Per-cell sandwich cores
        # ------------------------------------------------------------------
        if core_enabled and N >= 2:
            laminates = [cap_upper, cap_lower] + web_sdfs
            for i in range(N - 1):
                core = SandwichCore(
                    airfoil_sdf    = airfoil_sdf,
                    skin_thickness = skin_thickness,
                    exclusion_sdfs = laminates,
                    x_start        = xs[i],
                    x_end          = xs[i + 1],
                )
                self._components[f"core_{i}"] = core

        elif core_enabled and N == 1:
            # Single-web: two half-cells (fore and aft of web)
            laminates = [cap_upper, cap_lower] + web_sdfs
            for i, (x_s, x_e, suffix) in enumerate([
                (None, xs[0], "fore"),
                (xs[0], None, "aft"),
            ]):
                core = SandwichCore(
                    airfoil_sdf    = airfoil_sdf,
                    skin_thickness = skin_thickness,
                    exclusion_sdfs = laminates,
                    x_start        = x_s,
                    x_end          = x_e,
                )
                self._components[f"core_{suffix}"] = core

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
        """Two-web, single-cell box-spar."""
        return cls(airfoil_sdf, web_x_positions=[x_fore, x_aft], **kwargs)

    @classmethod
    def torsion_box(cls, airfoil_sdf, x_fore=0.15, x_mid=0.35, x_aft=0.55, **kwargs):
        """Three-web torsion box (two cells)."""
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
        return max(0, self.n_webs - 1)

    def __repr__(self):
        return (
            f"MultiCellSection("
            f"n_webs={self.n_webs}, "
            f"n_cells={self.n_cells}, "
            f"twist={np.degrees(self._twist):.1f}°, "
            f"components={self.labels})"
        )
