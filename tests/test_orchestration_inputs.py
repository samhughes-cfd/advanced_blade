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
from blade_precompute.orchestration.system_layout import build_section_view
from blade_precompute.section_geometry.engine.implicit_section_geometry import (
    AirfoilSDF,
    SDFGrid,
)


def test_resolve_system_type_2d_cn():
    s = resolve_system_type("2D-CN")
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
    blade_spec_path = root / "example_blade.json"
    if not blade_spec_path.is_file():
        pytest.skip("example_blade.json missing")
    cmap = ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0)
    validate_component_indices(blade_spec_path, cmap)
    table = ply_library_material_table(blade_spec_path)
    assert "gfrp_ply" in table


def test_system_type_keys_sorted_includes_2d_cn():
    assert "2D-CN" in SYSTEM_TYPE_KEYS


@pytest.mark.parametrize("key", ("0A", "0B"))
def test_airfoil_only_section_has_core_0(key: str) -> None:
    layout = resolve_system_type(key)
    af = AirfoilSDF.from_naca("0012", chord=1.0)
    section = build_section_view(af, layout, twist_angle_rad=0.0)
    assert tuple(section.labels) == ("outer_skin", "core_0")
    assert len(section) == 2
    assert section["core_0"] is not None


def test_airfoil_only_core_0_interior_inside_airfoil() -> None:
    layout = resolve_system_type("0A")
    af = AirfoilSDF.from_naca("0012", chord=1.0)
    section = build_section_view(af, layout, twist_angle_rad=0.0)
    grid = SDFGrid.from_airfoil(af, padding=0.05, nx=200, ny=100)
    phi_core = grid.eval(section["core_0"])
    phi_af = grid.eval(af)
    interior = phi_core < 0.0
    assert interior.any()
    assert (phi_af[interior] <= 0.01).all()
