"""Frozen component → ply-library material index map (orchestration input)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Mapping

import yaml

COMPONENT_MATERIAL_KEYS: FrozenSet[str] = frozenset({"skin", "spar_cap", "shear_web"})


@dataclass(frozen=True)
class ComponentMaterialsMap:
    """Maps high-level structural components to a **0-based material table index**.

    The material table is ``sorted(ply_library.keys())`` from the blade YAML
    (same file as ``BladeDesignProblem.load_geometry``). This matches ply-type
    lookup order used when resolving laminates.
    """

    skin: int
    spar_cap: int
    shear_web: int

    def to_dict(self) -> dict[str, int]:
        return {"skin": int(self.skin), "spar_cap": int(self.spar_cap), "shear_web": int(self.shear_web)}

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "ComponentMaterialsMap":
        missing = COMPONENT_MATERIAL_KEYS - frozenset(str(k) for k in m.keys())
        if missing:
            raise KeyError(f"component_materials missing keys: {sorted(missing)}")
        extra = frozenset(str(k) for k in m.keys()) - COMPONENT_MATERIAL_KEYS
        if extra:
            raise KeyError(f"component_materials unknown keys: {sorted(extra)}")
        skin = int(m["skin"])
        spar_cap = int(m["spar_cap"])
        shear_web = int(m["shear_web"])
        for name, v in (("skin", skin), ("spar_cap", spar_cap), ("shear_web", shear_web)):
            if v < 0:
                raise ValueError(f"{name} index must be >= 0; got {v}.")
        return ComponentMaterialsMap(skin=skin, spar_cap=spar_cap, shear_web=shear_web)


def load_component_materials_json(path: Path) -> ComponentMaterialsMap:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a JSON object at the top level.")
    return ComponentMaterialsMap.from_mapping(raw)


def ply_library_material_table(blade_yaml: Path) -> tuple[str, ...]:
    """Ordered material keys (ply library) for index validation."""
    blade_yaml = Path(blade_yaml)
    doc = yaml.safe_load(blade_yaml.read_text(encoding="utf-8"))
    ply_lib = doc.get("ply_library") or {}
    if not isinstance(ply_lib, dict):
        raise TypeError("ply_library must be a mapping in the blade YAML.")
    return tuple(sorted(ply_lib.keys(), key=str))


def validate_component_indices(blade_yaml: Path, cmap: ComponentMaterialsMap) -> None:
    table = ply_library_material_table(blade_yaml)
    if not table:
        raise ValueError(f"No ply_library entries in {blade_yaml}; cannot validate material indices.")
    n = len(table)
    for role, idx in cmap.to_dict().items():
        if idx >= n:
            raise ValueError(
                f"component_materials.{role}={idx} out of range for ply_library "
                f"(n={n}, keys={list(table)})."
            )
