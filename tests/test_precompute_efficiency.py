"""Tests for precompute efficiency wiring (parallel workers, design seed from section properties)."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import (
    GridConfig,
    grid_resolution_manifest,
    job_span_z_m,
    LinspaceSpec,
    load_inputs,
    resample_blade_geometry_to_z,
    resample_precompute_inputs,
    warn_geometry_shorter_than_job_span,
)
from blade_precompute.orchestration.precompute.grid import linspace_from_spec
from blade_precompute.orchestration.precompute.stages import (
    section_optimisation_impl,
    section_properties_impl,
)
from blade_precompute.orchestration.system_layout import resolve_system_type
from blade_precompute.section_optimisation.api import BladeDesignProblem


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="2D-CN",
        layout=resolve_system_type("2D-CN"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


def test_grid_config_worker_fields() -> None:
    g = LinspaceSpec(0.0, 1.0, 2)
    s = LinspaceSpec(0.0, 1.0, 2)
    cfg = GridConfig(
        geometry=g,
        structural=s,
        section_plot_station_spec="root",
        n_beam_nodes=10,
        design_n_workers=2,
        section_solve_n_workers=3,
    )
    assert cfg.design_n_workers == 2
    assert cfg.section_solve_n_workers == 3
    man = grid_resolution_manifest(cfg)
    assert man["span_z_source"] == "precompute_inputs.span_r_z_m"
    assert man["n_geometry_stations"] == 2
    assert man["n_structural_stations"] == 2
    # Backward-compatible aliases may be dropped; canonical keys are asserted above.
    assert man["beam_nodes"] == 10
    assert man["beam_png_span_samples"] == 400
    assert man["section_plot_station_spec"] == "root"


def test_section_properties_k6_k7_independent_of_section_solve_workers(tmp_path: Path) -> None:
    root = _repo_root()
    blade_spec = root / "example_blade_10.json"
    if not blade_spec.is_file():
        pytest.skip("example_blade_10.json missing")
    bg = BladeDesignProblem.load_geometry(blade_spec.resolve())
    z = bg.z_stations.ravel()
    sspec = LinspaceSpec(z_min=float(z[0]), z_max=float(z[-1]), n=int(z.shape[0]))
    bg_struct = resample_blade_geometry_to_z(bg, linspace_from_spec(sspec))
    inp = load_inputs(root / "data_library")
    zg0, zg1 = job_span_z_m(inp)
    gspec = LinspaceSpec(z_min=zg0, z_max=zg1, n=int(inp.span_r_z_m.size))
    inp_geom = resample_precompute_inputs(inp, linspace_from_spec(gspec))
    inp_struct = resample_precompute_inputs(inp, linspace_from_spec(sspec))

    out1 = section_properties_impl(
        inp_struct,
        tmp_path / "a",
        blade_yaml=blade_spec,
        section_plot_station_spec="root",
        orchestration=_orch(),
        bg_override=bg_struct,
        section_solve_n_workers=1,
    )
    out2 = section_properties_impl(
        inp_struct,
        tmp_path / "b",
        blade_yaml=blade_spec,
        section_plot_station_spec="root",
        orchestration=_orch(),
        bg_override=bg_struct,
        section_solve_n_workers=2,
    )
    assert np.allclose(out1.K6, out2.K6, rtol=0, atol=0)
    assert np.allclose(out1.K7, out2.K7, rtol=0, atol=0)
    sp_a = tmp_path / "a" / "section_properties"
    station_pngs = list(sp_a.glob("station_*/section_station.png"))
    assert station_pngs, "plot PNGs should be under station_* subdirectories"


def test_seed_stations_makes_first_evaluate_skip_midsurface_solve(tmp_path: Path) -> None:
    root = _repo_root()
    blade_spec = root / "example_blade_10.json"
    if not blade_spec.is_file():
        pytest.skip("example_blade_10.json missing")
    bg = BladeDesignProblem.load_geometry(blade_spec.resolve())
    z = bg.z_stations.ravel()
    sspec = LinspaceSpec(z_min=float(z[0]), z_max=float(z[-1]), n=int(z.shape[0]))
    bg_struct = resample_blade_geometry_to_z(bg, linspace_from_spec(sspec))
    inp = load_inputs(root / "data_library")
    zg0, zg1 = job_span_z_m(inp)
    gspec = LinspaceSpec(z_min=zg0, z_max=zg1, n=int(inp.span_r_z_m.size))
    inp_geom = resample_precompute_inputs(inp, linspace_from_spec(gspec))
    inp_struct = resample_precompute_inputs(inp, linspace_from_spec(sspec))

    sp = section_properties_impl(
        inp_struct,
        tmp_path / "sp",
        blade_yaml=blade_spec,
        section_plot_station_spec="root",
        orchestration=_orch(),
        bg_override=bg_struct,
        section_solve_n_workers=1,
    )
    n_calls: list[int] = []

    def _counting_solve(
        section_defs: list,
        dirty_indices: list[int],
        n_workers: int = 4,
    ) -> dict:
        n_calls.append(len(dirty_indices))
        from blade_precompute.section_optimisation.engine.parallel import solve_dirty_stations as _real

        return _real(section_defs, dirty_indices, n_workers=n_workers)

    with patch(
        "blade_precompute.section_optimisation.engine.evaluator.solve_dirty_stations",
        side_effect=_counting_solve,
    ):
        section_optimisation_impl(
            inp_geom,
            tmp_path / "opt",
            blade_yaml=blade_spec,
            orchestration=_orch(),
            run_blade_optimizer=False,
            design_n_workers=1,
            section_properties=sp,
            seed_section_properties=True,
        )
    assert n_calls, "expected at least one call into solve_dirty_stations"
    assert n_calls[0] == 0, "first evaluate after seed should have no dirty section stations"


def test_job_span_z_m_matches_span_r_z_endpoints() -> None:
    root = _repo_root()
    inp = load_inputs(root / "data_library")
    z0, z1 = job_span_z_m(inp)
    z = inp.span_r_z_m.ravel()
    assert z0 == float(z[0])
    assert z1 == float(z[-1])


def test_warn_geometry_shorter_than_job_span_emits_user_warning() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_geometry_shorter_than_job_span(9.625, 8.0, blade_spec=Path("example_blade_10.json"))
    assert len(rec) == 1
    assert issubclass(rec[0].category, UserWarning)


def test_warn_geometry_shorter_than_job_span_silent_when_span_inside_geometry() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warn_geometry_shorter_than_job_span(8.0, 9.625, blade_spec=None)
    assert len(rec) == 0
