"""Orchestration: system type registry and component material map."""

from __future__ import annotations

from pathlib import Path

import pytest

from blade_precompute.orchestration import (
    SYSTEM_TYPE_KEYS,
    load_component_materials_json,
    ply_library_material_table,
    resolve_system_type,
    validate_component_indices,
)
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap


def test_resolve_system_type_legacy():
    s = resolve_system_type("legacy")
    assert s.n_webs == 2
    assert s.web_chord_fracs == (0.15, 0.50)


def test_component_materials_roundtrip(tmp_path: Path):
    p = tmp_path / "m.json"
    p.write_text('{"skin": 0, "spar_cap": 0, "shear_web": 0}\n', encoding="utf-8")
    m = load_component_materials_json(p)
    assert m.skin == 0


def test_component_materials_rejects_unknown_key(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text('{"skin": 0, "spar_cap": 0, "shear_web": 0, "extra": 1}\n', encoding="utf-8")
    with pytest.raises(KeyError):
        load_component_materials_json(p)


def test_validate_component_indices_example_blade():
    root = Path(__file__).resolve().parents[1]
    yaml_path = root / "example_blade.yaml"
    cmap = ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0)
    validate_component_indices(yaml_path, cmap)
    table = ply_library_material_table(yaml_path)
    assert "gfrp_ply" in table


def test_system_type_keys_sorted_includes_legacy():
    assert "legacy" in SYSTEM_TYPE_KEYS
