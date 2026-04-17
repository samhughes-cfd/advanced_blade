from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry

from .constraints import build_constrained_geometry
from .extract import extract_midline_from_offset_boundaries
from .types import GeometryConstraintSpec, MedialAxisDiagnostics


@dataclass
class ImplicitSectionBuildResult:
    section: SectionDefinition
    diagnostics: dict[str, MedialAxisDiagnostics]


def _segment_dir(seg: np.ndarray) -> np.ndarray:
    v = np.asarray(seg[1] - seg[0], dtype=np.float64)
    n = np.linalg.norm(v) + 1e-30
    return v / n


def _spar_cap_between_webs(web_left: np.ndarray, web_right: np.ndarray, cap: np.ndarray) -> bool:
    a = web_left[0]
    b = web_right[0]
    t = b - a
    den = float(np.dot(t, t)) + 1e-30
    s0 = float(np.dot(cap[0] - a, t) / den)
    s1 = float(np.dot(cap[1] - a, t) / den)
    lo, hi = sorted([s0, s1])
    return lo >= 0.0 and hi <= 1.0


def _closest_point(poly: np.ndarray, p: np.ndarray) -> np.ndarray:
    idx = int(np.argmin(np.linalg.norm(poly - p[None, :], axis=1)))
    return np.asarray(poly[idx], dtype=np.float64)


def _ensure_segment(a: np.ndarray, b: np.ndarray, direction: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if np.linalg.norm(bb - aa) < 1e-9:
        d = np.asarray(direction, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-30)
        bb = aa + 1e-3 * d
    return np.vstack([aa, bb])


def build_section_from_constraints(spec: GeometryConstraintSpec) -> ImplicitSectionBuildResult:
    geom = build_constrained_geometry(spec)

    skin_mid = extract_midline_from_offset_boundaries(
        outer_boundary=geom.skin_outer_s,
        inner_boundary=geom.skin_inner_s,
        strip_width_m=float(spec.skin_thickness),
    )
    skin_pts = skin_mid.midsurface_coords_s
    wl_top = _closest_point(skin_pts, geom.web_left_s[0])
    wl_bot = _closest_point(skin_pts, geom.web_left_s[1])
    wr_top = _closest_point(skin_pts, geom.web_right_s[0])
    wr_bot = _closest_point(skin_pts, geom.web_right_s[1])
    web_left_coords = _ensure_segment(wl_top, wl_bot, np.array([0.0, 1.0], dtype=np.float64))
    web_right_coords = _ensure_segment(wr_top, wr_bot, np.array([0.0, 1.0], dtype=np.float64))
    cap_start = wl_top
    cap_end = wr_top
    cap_coords = _ensure_segment(cap_start, cap_end, np.array([1.0, 0.0], dtype=np.float64))
    web_left_diag = MedialAxisDiagnostics(1, 0, 0, 0.0, 0, [])
    web_right_diag = MedialAxisDiagnostics(1, 0, 0, 0.0, 0, [])
    cap_diag = MedialAxisDiagnostics(1, 0, 0, 0.0, 0, [])

    flap_b = np.array([0.0, 1.0], dtype=np.float64)
    web_left_b = geom.frame.points_s_to_b(geom.web_left_s)
    web_right_b = geom.frame.points_s_to_b(geom.web_right_s)
    ang_tol = 5e-3
    if abs(np.dot(_segment_dir(web_left_b), flap_b)) < 1.0 - ang_tol:
        web_left_diag.notes.append("web_left not parallel to flapwise axis in B frame.")
    if abs(np.dot(_segment_dir(web_right_b), flap_b)) < 1.0 - ang_tol:
        web_right_diag.notes.append("web_right not parallel to flapwise axis in B frame.")
    if not _spar_cap_between_webs(geom.web_left_s, geom.web_right_s, geom.spar_cap_s):
        cap_diag.notes.append("spar cap is not between web stations.")
        raise ValueError("Spar-cap placement constraint violated: cap must remain between webs.")

    subcomponents = [
        SubcomponentGeometry(
            name="skin",
            midsurface_coords=skin_mid.midsurface_coords_s,
            material=spec.materials["skin"],
            thickness=float(spec.skin_thickness),
            strip_width_m=skin_mid.strip_width_m,
        ),
        SubcomponentGeometry(
            name="web_left",
            midsurface_coords=web_left_coords,
            material=spec.materials["web_left"],
            thickness=float(spec.web_width),
            strip_width_m=float(spec.web_width),
        ),
        SubcomponentGeometry(
            name="web_right",
            midsurface_coords=web_right_coords,
            material=spec.materials["web_right"],
            thickness=float(spec.web_width),
            strip_width_m=float(spec.web_width),
        ),
        SubcomponentGeometry(
            name="spar_cap",
            midsurface_coords=cap_coords,
            material=spec.materials["spar_cap"],
            thickness=float(spec.spar_cap_thickness),
            strip_width_m=float(spec.spar_cap_width),
        ),
    ]
    section = SectionDefinition(station_z=float(spec.station_z), subcomponents=subcomponents)
    diagnostics = {
        "skin": skin_mid.diagnostics,
        "web_left": web_left_diag,
        "web_right": web_right_diag,
        "spar_cap": cap_diag,
    }
    return ImplicitSectionBuildResult(section=section, diagnostics=diagnostics)

