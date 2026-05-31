"""
Load :class:`~section_model.engine.geometry.SectionDefinition` from mapping specs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional
import warnings

import numpy as np

from blade_precompute._utils.spec_io import load_mapping

from ..engine.geometry import SectionDefinition, SubcomponentGeometry
from ..engine.implicit_section_geometry import GeometryConstraintSpec, build_section_from_constraints
from ..engine.materials import IsotropicMaterial
from .materials_loader import laminate_from_mapping_spec


def load_section_from_spec(
    path: str | Path,
    *,
    ply_library: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> SectionDefinition:
    """Parse a section mapping spec into :class:`~section_model.engine.geometry.SectionDefinition`."""
    p = Path(path)
    data: MutableMapping[str, Any] = load_mapping(p)

    ply_lib: Dict[str, Mapping[str, Any]] = dict(data.get("ply_library") or {})
    if ply_library:
        ply_lib.update(ply_library)

    if "implicit_geometry" in data:
        return _load_implicit_section_from_spec(data, ply_lib)

    subs_raw = data["subcomponents"]
    if not isinstance(subs_raw, dict):
        raise TypeError("subcomponents must be a mapping (dict).")

    materials_block = data.get("materials") or {}
    sub_list: List[SubcomponentGeometry] = []

    for name, spec in subs_raw.items():
        coords = np.asarray(spec["midsurface_coords"], dtype=np.float64)
        thickness = float(spec.get("thickness", 1e-3))
        sw = spec.get("strip_width_m")
        strip_width_m = float(sw) if sw is not None else None
        if strip_width_m is None:
            warnings.warn(
                f"Subcomponent '{name}' has no strip_width_m; defaulting to thickness may bias shear/torsion stiffness.",
                UserWarning,
                stacklevel=2,
            )
        elif strip_width_m <= max(thickness, 1e-12):
            warnings.warn(
                f"Subcomponent '{name}' strip_width_m <= thickness; verify strip idealisation.",
                UserWarning,
                stacklevel=2,
            )

        mat_key = str(spec.get("material", name))
        mb = materials_block.get(mat_key, {})
        lam_spec = dict(mb)
        lam_spec.update(
            {k: v for k, v in spec.items() if k not in ("midsurface_coords", "material", "thickness", "strip_width_m")}
        )
        if "laminate" in spec:
            lam_spec.update(spec["laminate"])
        is_comp = "layup" in lam_spec and "ply_type" in lam_spec
        if is_comp:
            lam = laminate_from_mapping_spec(lam_spec, ply_lib, mat_key)
            sub_list.append(
                SubcomponentGeometry(
                    name=name,
                    midsurface_coords=coords,
                    material=lam,
                    thickness=thickness,
                    strip_width_m=strip_width_m,
                )
            )
        else:
            E = float(spec.get("E", mb.get("E", 70e9)))
            nu = float(spec.get("nu", mb.get("nu", 0.33)))
            rho = float(spec.get("rho", mb.get("rho", 2700.0)))
            sa = float(spec.get("sigma_allow", mb.get("sigma_allow", 270e6)))
            iso = IsotropicMaterial(name=mat_key, E=E, nu=nu, rho=rho, sigma_allow=sa)
            sub_list.append(
                SubcomponentGeometry(
                    name=name,
                    midsurface_coords=coords,
                    material=iso,
                    thickness=thickness,
                    strip_width_m=strip_width_m,
                )
            )

    R_def = data.get("R_deformed")
    R = np.asarray(R_def, dtype=np.float64) if R_def is not None else None
    return SectionDefinition(
        station_z=float(data.get("station_z", 0.0)),
        subcomponents=sub_list,
        R_deformed=R,
    )


def load_section_from_yaml(
    path: str | Path,
    *,
    ply_library: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> SectionDefinition:
    """Backward-compatible YAML loader alias for existing callers."""
    warnings.warn(
        "load_section_from_yaml() is deprecated; use load_section_from_spec().",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_section_from_spec(path, ply_library=ply_library)


def _material_from_block(
    name: str,
    spec: Mapping[str, Any],
    ply_lib: Mapping[str, Mapping[str, Any]],
) -> object:
    is_comp = "layup" in spec and "ply_type" in spec
    if is_comp:
        return laminate_from_mapping_spec(dict(spec), ply_lib, name)
    return IsotropicMaterial(
        name=name,
        E=float(spec.get("E", 70e9)),
        nu=float(spec.get("nu", 0.33)),
        rho=float(spec.get("rho", 2700.0)),
        sigma_allow=float(spec.get("sigma_allow", 270e6)),
    )


def _load_implicit_section_from_spec(
    data: Mapping[str, Any],
    ply_lib: Mapping[str, Mapping[str, Any]],
) -> SectionDefinition:
    ig = dict(data["implicit_geometry"])
    outer = np.asarray(ig["skin_outer_boundary_s"], dtype=np.float64)
    mat_block = dict(ig["materials"])
    mats = {
        "skin": _material_from_block("skin", mat_block["skin"], ply_lib),
        "web_left": _material_from_block("web_left", mat_block.get("web_left", mat_block["skin"]), ply_lib),
        "web_right": _material_from_block("web_right", mat_block.get("web_right", mat_block["skin"]), ply_lib),
        "spar_cap": _material_from_block("spar_cap", mat_block["spar_cap"], ply_lib),
    }
    spec = GeometryConstraintSpec(
        skin_outer_boundary_s=outer,
        skin_thickness=float(ig["skin_thickness"]),
        web_width=float(ig["web_width"]),
        web_stations_s=(float(ig["web_stations_s"][0]), float(ig["web_stations_s"][1])),
        spar_cap_width=float(ig["spar_cap_width"]),
        spar_cap_thickness=float(ig["spar_cap_thickness"]),
        twist_rad=float(ig.get("twist_rad", 0.0)),
        station_z=float(data.get("station_z", ig.get("station_z", 0.0))),
        materials=mats,
        n_samples=int(ig.get("n_samples", 256)),
    )
    return build_section_from_constraints(spec).section
