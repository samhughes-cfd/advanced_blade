from __future__ import annotations

from pathlib import Path

import pytest

from blade_precompute.orchestration.precompute.material_library import (
    MaterialRow,
    apply_material_library_to_blade_geometry,
    load_material_library_dat,
    material_resolution_manifest,
    normalize_logical_subcomponent_material_map,
    validate_material_library_bindings,
)
from blade_precompute.section_optimisation.api import BladeDesignProblem
from blade_precompute.section_optimisation.engine.ply_angle_constraints import (
    composite_thickness_m,
    validate_stack_angles_for_role,
)


def _repo() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_material_library_dat_repo_fixture() -> None:
    p = _repo() / "data_library" / "material_library.dat"
    table = load_material_library_dat(p)
    assert 0 in table
    assert table[0].kind == "orthotropic_laminate_ply"
    assert table[0].t_ply == pytest.approx(0.0002)
    assert 1 in table
    assert table[1].kind == "orthotropic_laminate_ply"
    assert table[1].name == "cfrp_ply"
    assert table[2].kind == "isotropic"
    assert table[2].name == "structural_steel"
    assert table[3].kind == "isotropic"
    assert table[3].name == "sandwich_foam"


def test_normalize_logical_subcomponent_material_map_aliases() -> None:
    m = normalize_logical_subcomponent_material_map({"skin": 0, "spar": 2, "web": 3})
    assert m == {"skin": 0, "spar_cap": 2, "shear_web": 3}


def test_validate_material_library_bindings_rejects_unknown_id() -> None:
    table = {0: MaterialRow(0, "a", "orthotropic_laminate_ply", E1=1, E2=1, G12=1, nu12=0.3, rho=1, t_ply=1, Xt=1, Xc=1, Yt=1, Yc=1, S12=1, Zt=1, S13=1, S23=1)}
    logical = {"skin": 0, "spar_cap": 0, "shear_web": 99}
    with pytest.raises(ValueError, match="not found"):
        validate_material_library_bindings(table, logical)


def test_material_resolution_manifest_disabled() -> None:
    m = material_resolution_manifest(material_library_path=None, logical=None, table=None)
    assert m["enabled"] is False


def test_composite_thickness_m() -> None:
    assert composite_thickness_m(n_plies=8, t_ply_m=0.0002) == pytest.approx(8 * 0.0002)


def test_validate_stack_angles_web_rejects_zero() -> None:
    with pytest.raises(ValueError, match="not in allowlist"):
        validate_stack_angles_for_role("web", [0.0], subcomponent="web")


def test_load_material_library_dat_tampered_unit_raises(tmp_path: Path) -> None:
    """Loading a .dat where a unit row has a wrong entry must raise ValueError."""
    good = (_repo() / "data_library" / "material_library.dat").read_text(encoding="utf-8")
    # Replace 'Pa' for E1 with 'MPa' to trigger the validator
    bad = good.replace(
        "# units: -, -, Pa, Pa, Pa, -, kg/m^3, m, Pa, Pa, Pa, Pa, Pa, Pa, Pa, Pa",
        "# units: -, -, MPa, Pa, Pa, -, kg/m^3, m, Pa, Pa, Pa, Pa, Pa, Pa, Pa, Pa",
    )
    assert bad != good, "Tamper string not found — check the units row in material_library.dat"
    tampered = tmp_path / "tampered_material_library.dat"
    tampered.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="unit"):
        load_material_library_dat(tampered)


def test_apply_material_library_preserves_spec_angles(tmp_path: Path) -> None:
    blade_spec = _repo() / "example_blade_10.json"
    if not blade_spec.is_file():
        pytest.skip("example_blade_10.json missing")
    bg = BladeDesignProblem.load_geometry(blade_spec.resolve())
    csv_path = _repo() / "data_library" / "material_library.dat"
    table = load_material_library_dat(csv_path)
    logical = normalize_logical_subcomponent_material_map({"skin": 0, "spar": 0, "web": 0})
    validate_material_library_bindings(
        table, logical, blade_subcomponent_names=frozenset(bg.subcomponent_materials.keys())
    )
    out = apply_material_library_to_blade_geometry(bg, table, logical)
    skin = out.subcomponent_materials["skin"]
    from blade_precompute.section_properties.engine.laminate import LaminateDefinition

    assert isinstance(skin, LaminateDefinition)
    angles = [float(a) for _, a in skin.plies]
    assert angles == [0.0, 45.0, -45.0, 90.0]
    assert skin.plies[0][0].E1 == pytest.approx(float(table[0].E1))
