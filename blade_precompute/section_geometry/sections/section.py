"""
sections.section
================
BladeSectionGeometry — top-level assembly of all subcomponents for a blade
cross-section.

For new work prefer MultiCellSection directly; BladeSectionGeometry is
kept as a convenience wrapper that delegates to MultiCellSection internally
and accepts the legacy config-dict API.

New constructor
---------------
    BladeSectionGeometry.from_n_web(airfoil, web_x_positions, ...)

Legacy constructor
------------------
    BladeSectionGeometry(airfoil, config)

Both return an object with the same access API:
    bsg["outer_skin"](X, Y)
    bsg.eval_union(X, Y)
    bsg.labels
"""

from .subcomponents import OuterSkin, SparCap, ShearWeb, SandwichCore
from .multicell import MultiCellSection


class BladeSectionGeometry:
    """Assembled blade cross-section geometry.

    Legacy config-dict API
    ----------------------
    Accepts the same config dict as before.  Internally builds a
    MultiCellSection from the config, so all new features (corrected
    curved spar caps, twist_angle, TE/LE inserts) are available.

    Parameters
    ----------
    airfoil_sdf : AirfoilSDF
    config : dict, optional
        Legacy configuration (see _DEFAULTS for schema).
    twist_angle : float, optional
        Section twist in radians.  Overrides config['twist_angle'] if set.
    """

    _DEFAULTS = {
        "skin_thickness": 0.003,
        "twist_angle":    0.0,
        "spar_cap": {
            "x_start":        0.15,
            "x_end":          0.50,
            "height":         0.015,
        },
        "shear_webs": [
            {"x": 0.15, "y_top":  0.06, "y_bot": -0.06,
             "thickness": 0.004, "alignment": "chord_normal"},
            {"x": 0.50, "y_top":  0.04, "y_bot": -0.04,
             "thickness": 0.004, "alignment": "chord_normal"},
        ],
        "core": {
            "enabled": True,
        },
        "te_insert": None,
        "le_insert": None,
    }

    def __init__(self, airfoil_sdf, config=None, twist_angle=None):
        self._af  = airfoil_sdf
        cfg       = self._merge_config(config or {})

        ta = float(twist_angle) if twist_angle is not None else float(cfg.get("twist_angle", 0.0))

        # Build from config via MultiCellSection
        sc_cfg   = cfg["spar_cap"]
        web_cfgs = cfg.get("shear_webs", [])

        web_xs    = [float(w.get("x", w.get("x_top", 0.25))) for w in web_cfgs]
        web_ts    = [float(w.get("thickness", 0.004)) for w in web_cfgs]
        web_alns  = [w.get("alignment", "chord_normal") for w in web_cfgs]
        web_ys    = [(float(w.get("y_top", 0.06)), float(w.get("y_bot", -0.04)))
                     for w in web_cfgs]

        # cap_height: use x_start/x_end from spar_cap config if they differ from web span
        # MultiCellSection will clip caps to [x_webs[0], x_webs[-1]].
        # If a specific override range is given, we honour it via explicit web_y_coords
        cap_height = float(sc_cfg.get("height", 0.015))

        te_cfg = cfg.get("te_insert")
        le_cfg = cfg.get("le_insert")
        te_x   = float(te_cfg["x_start"]) if te_cfg else None
        le_x   = float(le_cfg["x_end"])   if le_cfg else None

        self._mcs = MultiCellSection(
            airfoil_sdf     = airfoil_sdf,
            web_x_positions = web_xs if web_xs else [sc_cfg["x_start"], sc_cfg["x_end"]],
            web_thickness   = web_ts if web_ts else [0.004, 0.004],
            web_alignment   = web_alns if web_alns else ["chord_normal", "chord_normal"],
            cap_height      = cap_height,
            skin_thickness  = float(cfg["skin_thickness"]),
            twist_angle     = ta,
            web_y_coords    = web_ys if web_ys else None,
            te_insert_x     = te_x,
            le_insert_x     = le_x,
            core_enabled    = cfg.get("core", {}).get("enabled", True),
        )

    # ------------------------------------------------------------------
    # Config merge
    # ------------------------------------------------------------------

    def _merge_config(self, user):
        import copy
        merged = copy.deepcopy(self._DEFAULTS)
        for k, v in user.items():
            if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    # ------------------------------------------------------------------
    # Delegate all access to the underlying MultiCellSection
    # ------------------------------------------------------------------

    def __getitem__(self, label):
        return self._mcs[label]

    def __iter__(self):
        return iter(self._mcs)

    def __len__(self):
        return len(self._mcs)

    @property
    def labels(self):
        return self._mcs.labels

    @property
    def airfoil(self):
        return self._af

    def eval(self, label, x, y):
        return self._mcs.eval(label, x, y)

    def eval_all(self, x, y):
        return self._mcs.eval_all(x, y)

    def eval_union(self, x, y):
        return self._mcs.eval_union(x, y)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_n_web(cls, airfoil_sdf, web_x_positions,
                   web_thickness=0.004,
                   web_alignment="chord_normal",
                   cap_height=0.012,
                   skin_thickness=0.003,
                   twist_angle=0.0,
                   web_y_coords=None,
                   te_insert_x=None,
                   le_insert_x=None,
                   le_radius=None,
                   core_enabled=True,
                   y_search=None,
                   structural_family="D"):
        """Construct directly from N web positions.

        Bypasses the config-dict API and calls MultiCellSection directly.

        Returns
        -------
        BladeSectionGeometry-like object
            Actually returns a MultiCellSection (same interface).
        """
        return MultiCellSection(
            airfoil_sdf     = airfoil_sdf,
            web_x_positions = web_x_positions,
            web_thickness   = web_thickness,
            web_alignment   = web_alignment,
            cap_height      = cap_height,
            skin_thickness  = skin_thickness,
            twist_angle     = twist_angle,
            web_y_coords    = web_y_coords,
            te_insert_x     = te_insert_x,
            le_insert_x     = le_insert_x,
            le_radius       = le_radius,
            core_enabled    = core_enabled,
            y_search        = y_search,
            structural_family = structural_family,
        )

    def __repr__(self):
        return f"BladeSectionGeometry(wrapping {self._mcs!r})"
