from __future__ import annotations

from pathlib import Path

import numpy as np

from blade_precompute.section_optimisation.io.blade_geometry_loader import load_blade_geometry
from blade_precompute.section_optimisation.io.yaml_loader import load_blade_geometry as load_blade_geometry_yaml
from blade_precompute.section_optimisation.core.failure import tsai_wu_fi_plies
from blade_precompute.section_properties.io import load_section_from_yaml


def test_blade_geometry_loader_accepts_yaml_specs(tmp_path: Path) -> None:
    spec = tmp_path / "blade.yaml"
    spec.write_text(
        """
ply_library:
  ud:
    E1: 40000000000.0
    E2: 10000000000.0
    G12: 4000000000.0
    nu12: 0.28
    rho: 1900.0
    t_ply: 0.0002
    Xt: 900000000.0
    Xc: 650000000.0
    Yt: 65000000.0
    Yc: 120000000.0
    S12: 75000000.0
    Zt: 45000000.0
    S13: 40000000.0
    S23: 40000000.0
blade:
  z_stations: [0.0, 1.0]
  r_ref: [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
  kappa0: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  chord: [1.0, 0.8]
  twist: [0.0, 0.0]
  web_positions: [-0.3, 0.3]
  subcomponents:
    skin:
      material: laminate
      thickness_role: skin
      ply_type: ud
      layup: [0.0, 90.0]
""",
        encoding="utf-8",
    )

    bg = load_blade_geometry(spec)
    bg_from_legacy = load_blade_geometry_yaml(spec)

    np.testing.assert_allclose(bg.z_stations, np.array([0.0, 1.0]))
    np.testing.assert_allclose(bg_from_legacy.z_stations, bg.z_stations)
    assert "skin" in bg.subcomponent_materials
    assert bg.thickness_role["skin"] == "skin"


def test_legacy_section_yaml_loader_alias_accepts_yaml(tmp_path: Path) -> None:
    spec = tmp_path / "section.yaml"
    spec.write_text(
        """
station_z: 0.0
subcomponents:
  insert:
    midsurface_coords: [[0.0, 0.0], [1.0, 0.0]]
    thickness: 0.002
    material: aluminium
    E: 70000000000.0
    nu: 0.33
    rho: 2700.0
    sigma_allow: 260000000.0
""",
        encoding="utf-8",
    )

    section = load_section_from_yaml(spec)

    assert len(section.subcomponents) == 1
    assert section.subcomponents[0].name == "insert"


def test_legacy_failure_shim_imports_vectorized_tsai_wu() -> None:
    sigma = np.array([[[1.0e6, 2.0e6, 3.0e5]]], dtype=np.float64)
    strength = np.array([[1.0e9]], dtype=np.float64)

    fi = tsai_wu_fi_plies(sigma, strength, strength, strength, strength, strength)

    assert fi.shape == (1, 1)
    assert np.isfinite(fi[0, 0])
