"""Smoke tests for spanwise + material_library :class:`OptimBladeGeometry` (no example blade spec file)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_precompute.orchestration.precompute import build_precompute_orchestration_context
from blade_precompute.orchestration.precompute import (
    LinspaceSpec,
    build_optim_blade_geometry_from_spanwise,
    job_span_z_m,
    linspace_from_spec,
    load_inputs,
    load_material_library_dat,
    normalize_logical_subcomponent_material_map,
    resample_precompute_inputs,
)


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_library"
MAT = DATA / "material_library.dat"
SUB_IDS = {"skin": 0, "spar": 0, "web": 0}


def test_orchestration_builds_without_component_materials_json() -> None:
    """Context builds correctly when component_materials.json is absent (file was deleted)."""
    ctx = build_precompute_orchestration_context(
        data_dir=DATA,
        blade_yaml=None,
        system_type_key="2D-F",
        component_materials_path=None,
        skip_component_index_validation=True,
    )
    assert ctx.system_type_key == "2D-F"
    # stub map should be populated (all zeros) when no JSON file is present
    assert ctx.component_materials.skin == 0
    assert ctx.component_materials.spar_cap == 0
    assert ctx.component_materials.shear_web == 0


@pytest.mark.skipif(not (DATA / "blade_spanwise_distribution.dat").is_file(), reason="data_library spanwise .dat")
@pytest.mark.skipif(not MAT.is_file(), reason="material_library.dat")
def test_build_optim_from_spanwise_matches_stations() -> None:
    inp0 = load_inputs(DATA)
    z_root, z_tip = job_span_z_m(inp0)
    z5 = linspace_from_spec(LinspaceSpec(z_min=z_root, z_max=z_tip, n=5))
    inp = resample_precompute_inputs(inp0, z5)
    table = load_material_library_dat(MAT)
    logical = normalize_logical_subcomponent_material_map(SUB_IDS)
    from blade_precompute.orchestration import resolve_system_type

    layout = resolve_system_type("2D-F")
    bg = build_optim_blade_geometry_from_spanwise(
        inp,
        mat_table=table,
        logical=logical,
        system_layout=layout,
    )
    n = int(inp.span_r_z_m.size)
    assert int(bg.z_stations.size) == n
    assert int(bg.r_ref.shape[0]) == n
    assert len(bg.airfoil_profiles) == n
    assert frozenset(bg.subcomponent_materials) == frozenset({"cap_ps", "skin", "web"})
