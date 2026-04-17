"""Build spanwise ``SectionDefinition`` from design variables and blade geometry."""

from __future__ import annotations

import copy

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial

from ..core.types import DesignVector, OptimBladeGeometry, ThicknessRole


def _infer_role(name: str, explicit: dict[str, ThicknessRole]) -> ThicknessRole:
    if name in explicit:
        return explicit[name]
    n = name.lower()
    if "skin" in n:
        return "skin"
    if "cap" in n:
        return "cap"
    if "web" in n:
        return "web"
    return "fixed"


def _default_norm_polylines(bg: OptimBladeGeometry) -> dict[str, NDArray[np.float64]]:
    h = float(bg.box_height_frac)
    wy = np.asarray(bg.web_positions, dtype=np.float64).ravel()
    if wy.size < 2:
        wy = np.array([-0.35, 0.35], dtype=np.float64)
    y0, y1 = float(wy[0]), float(wy[1])
    return {
        "skin": np.array([[-0.5, 0.0], [0.5, 0.0], [0.5, h], [-0.5, h]], dtype=np.float64),
        "cap_ps": np.array([[y0, 0.0], [y0, h]], dtype=np.float64),
        "cap_ss": np.array([[y1, 0.0], [y1, h]], dtype=np.float64),
        "web_ps": np.array([[y0, 0.0], [y0, h]], dtype=np.float64),
        "web_ss": np.array([[y1, 0.0], [y1, h]], dtype=np.float64),
        "web": np.array([[y0, 0.0], [y0, h]], dtype=np.float64),
        "leading_edge_insert": np.array([[-0.5, 0.0], [-0.5, 0.25 * h]], dtype=np.float64),
    }


def _merged_norm_polylines(bg: OptimBladeGeometry) -> dict[str, NDArray[np.float64]]:
    d = _default_norm_polylines(bg)
    if bg.subcomponent_polylines_norm:
        for k, v in bg.subcomponent_polylines_norm.items():
            d[k] = np.asarray(v, dtype=np.float64)
    return d


def _scale_twist(pts: NDArray[np.float64], chord: float, twist_deg: float) -> NDArray[np.float64]:
    t = np.deg2rad(float(twist_deg))
    c, s = np.cos(t), np.sin(t)
    out = np.zeros_like(pts, dtype=np.float64)
    for k in range(pts.shape[0]):
        yn, zn = float(pts[k, 0]) * chord, float(pts[k, 1]) * chord
        out[k, 0] = c * yn - s * zn
        out[k, 1] = s * yn + c * zn
    return out


class SectionBuilder:
    @staticmethod
    def build(dv: DesignVector, blade_geometry: OptimBladeGeometry) -> list[SectionDefinition]:
        n = int(blade_geometry.z_stations.shape[0])
        if dv.t_skin.shape[0] != n:
            raise ValueError("DesignVector station count must match OptimBladeGeometry.")
        norms = _merged_norm_polylines(blade_geometry)
        roles = blade_geometry.thickness_role
        out: list[SectionDefinition] = []
        for i in range(n):
            chord = float(blade_geometry.chord[i])
            twist = float(blade_geometry.twist[i])
            subs: list[SubcomponentGeometry] = []
            for name in sorted(blade_geometry.subcomponent_materials.keys()):
                mat_template = blade_geometry.subcomponent_materials[name]
                role = _infer_role(name, roles)
                pts_n = norms.get(name)
                if pts_n is None:
                    continue
                pts = _scale_twist(np.asarray(pts_n, dtype=np.float64), chord, twist)
                strip_w = max(0.01 * chord, 1e-4)
                if isinstance(mat_template, LaminateDefinition):
                    lam0 = copy.deepcopy(mat_template)
                    if role == "skin":
                        t_new = float(dv.t_skin[i])
                    elif role == "cap":
                        t_new = float(dv.t_cap[i])
                    elif role == "web":
                        t_new = float(dv.t_web[i])
                    else:
                        t_new = lam0.total_thickness()
                    lam = lam0.scale_thickness(t_new)
                    if role == "cap":
                        b_cap = blade_geometry.cap_shear_lag_width or 0.15 * chord
                        t_skin_ref = float(dv.t_skin[i])
                        lam = lam.apply_shear_lag(b_cap, max(t_skin_ref, 1e-6))
                    subs.append(
                        SubcomponentGeometry(
                            name=name,
                            midsurface_coords=pts,
                            material=lam,
                            thickness=float(lam.total_thickness()),
                            strip_width_m=strip_w,
                        )
                    )
                elif isinstance(mat_template, IsotropicMaterial):
                    mat = copy.deepcopy(mat_template)
                    if role == "skin":
                        th = float(dv.t_skin[i])
                    elif role == "cap":
                        th = float(dv.t_cap[i])
                    elif role == "web":
                        th = float(dv.t_web[i])
                    else:
                        th = 0.004
                    subs.append(
                        SubcomponentGeometry(
                            name=name,
                            midsurface_coords=pts,
                            material=mat,
                            thickness=th,
                            strip_width_m=strip_w,
                        )
                    )
                else:
                    raise TypeError(type(mat_template))
            out.append(SectionDefinition(station_z=float(blade_geometry.z_stations[i]), subcomponents=subs))
        return out
