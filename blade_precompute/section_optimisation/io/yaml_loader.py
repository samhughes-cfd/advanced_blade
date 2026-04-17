"""Load :class:`~design_optimisation.core.types.OptimBladeGeometry` from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from blade_precompute.section_properties.engine.geometry import MaterialAssignment
from blade_precompute.section_properties.engine.materials import IsotropicMaterial
from blade_precompute.section_properties.io.yaml_materials import laminate_from_yaml_spec

from ..core.types import OptimBladeGeometry, ThicknessRole


def _build_isotropic(spec: Mapping[str, Any], mat_key: str) -> IsotropicMaterial:
    if "E" not in spec:
        raise KeyError(f"Isotropic material '{mat_key}' needs E, nu, rho, sigma_allow.")
    return IsotropicMaterial(
        name=str(spec.get("name", mat_key)),
        E=float(spec["E"]),
        nu=float(spec["nu"]),
        rho=float(spec["rho"]),
        sigma_allow=float(spec["sigma_allow"]),
    )


def load_blade_geometry(path: str | Path) -> OptimBladeGeometry:
    """
    Parse YAML with top-level ``blade:`` and optional ``ply_library:``.

    ``blade.twist`` is structural blade twist in **degrees** per ``z_stations`` row (built-in
    section orientation / washout), not angle of attack or global pitch.

    Each ``blade.subcomponents.<name>`` entry is either::

        material: laminate
        ply_type: ...
        layup: [...]

    or::

        material: isotropic
        E: ...
        nu: ...
        rho: ...
        sigma_allow: ...
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ply_lib: dict[str, Any] = dict(raw.get("ply_library", {}))
    blade = raw.get("blade")
    if not isinstance(blade, dict):
        raise ValueError("YAML must contain a mapping 'blade:'.")

    z_stations = np.asarray(blade["z_stations"], dtype=np.float64)
    r_ref = np.asarray(blade["r_ref"], dtype=np.float64)
    kappa0 = np.asarray(blade["kappa0"], dtype=np.float64)
    tau0 = np.asarray(blade["tau0"], dtype=np.float64)
    chord = np.asarray(blade["chord"], dtype=np.float64)
    twist = np.asarray(blade["twist"], dtype=np.float64)
    web_positions = np.asarray(blade.get("web_positions", [-0.35, 0.35]), dtype=np.float64)
    airfoil_profiles = list(blade.get("airfoil_profiles", []))

    subs = blade.get("subcomponents", {})
    if not isinstance(subs, dict):
        raise ValueError("blade.subcomponents must be a mapping.")
    subcomponent_materials: dict[str, MaterialAssignment] = {}
    thickness_role: dict[str, ThicknessRole] = {}
    for name, spec in subs.items():
        if not isinstance(spec, dict):
            continue
        role = str(spec.get("thickness_role", "")).lower()
        if role in ("skin", "cap", "web", "fixed"):
            thickness_role[name] = role  # type: ignore[assignment]
        mtype = str(spec.get("material", "laminate")).lower()
        if mtype not in ("laminate", "isotropic") and "layup" in spec:
            mtype = "laminate"
        if mtype == "laminate":
            subcomponent_materials[name] = laminate_from_yaml_spec(spec, ply_lib, name)
        elif mtype == "isotropic":
            subcomponent_materials[name] = _build_isotropic(spec, name)
        else:
            raise ValueError(f"Unknown material type '{mtype}' for subcomponent '{name}'.")

    cap_w = blade.get("cap_shear_lag_width")
    cap_shear_lag_width = float(cap_w) if cap_w is not None else None
    box_height_frac = float(blade.get("box_height_frac", 0.12))

    return OptimBladeGeometry(
        z_stations=z_stations,
        r_ref=r_ref,
        kappa0=kappa0,
        tau0=tau0,
        chord=chord,
        twist=twist,
        airfoil_profiles=airfoil_profiles,
        web_positions=web_positions,
        subcomponent_materials=subcomponent_materials,
        thickness_role=thickness_role,
        cap_shear_lag_width=cap_shear_lag_width,
        box_height_frac=box_height_frac,
    )
