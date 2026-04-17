"""Integration tests: GBT-derived section stations and global beam solve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, SolverOptions
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.section_property_interpolator import (
    SectionPropertyInterpolator,
    section_stiffness_array_from_sequence,
)
from blade_precompute.orchestration.gbt_beam_stations import beam_section_stations_from_gbt
from blade_precompute.section_beam_model.gbt.section_stiffness_export import SectionStiffness
from blade_precompute.section_optimisation.api import BladeDesignProblem
from blade_precompute.section_optimisation.core.types import DesignVector
from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_dv0(n_station: int) -> DesignVector:
    n = int(n_station)
    return DesignVector(
        t_skin=np.full(n, 0.012, dtype=np.float64),
        t_cap=np.full(n, 0.050, dtype=np.float64),
        t_web=np.full(n, 0.015, dtype=np.float64),
    )


@pytest.fixture(scope="module")
def example_blade_yaml() -> Path:
    p = _repo_root() / "example_blade.yaml"
    if not p.is_file():
        pytest.skip(f"Missing {p}")
    return p


def test_section_to_global_pipeline(example_blade_yaml: Path) -> None:
    bg = BladeDesignProblem.load_geometry(example_blade_yaml)
    n = int(bg.z_stations.shape[0])
    dv = _default_dv0(n)
    section_defs = SectionBuilder.build(dv, bg)
    z = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)

    stations, _reports = beam_section_stations_from_gbt(z, tuple(section_defs), bg, n_beam_nodes=12)

    assert all(s.K7 is not None for s in stations)
    assert all(s.K7.shape == (7, 7) for s in stations if s.K7 is not None)
    assert all(float(s.K7[6, 6]) > 0.0 for s in stations if s.K7 is not None)
    for s in stations:
        assert s.K7 is not None
        K7 = np.asarray(s.K7, dtype=np.float64)
        emin = float(np.linalg.eigvalsh(0.5 * (K7 + K7.T)).min())
        assert emin >= -1e-2 * max(float(np.max(np.diag(K7))), 1.0)

    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        tau0=np.asarray(bg.tau0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )
    beam = BeamAnalysis.from_blade_geometry(geom, 12, stations, span_axis=2)
    model = beam.model
    n_nodes = model.n_nodes
    F = np.zeros((n_nodes, 3), dtype=np.float64)
    F[-1, 1] = 1.0
    loads = BeamLoads(
        nodal_F=F,
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    opts = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=12,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = beam.solve_static(loads, options=opts)
    tip = np.asarray(res.nodal_positions[-1] - model.X_ref[-1], dtype=np.float64)
    assert np.all(np.isfinite(tip))
    assert float(np.linalg.norm(tip)) > 0.0


def test_interpolator_monotonic_blade() -> None:
    s = np.linspace(0.0, 10.0, 6, dtype=np.float64)
    ei = np.linspace(5e6, 1e6, 6, dtype=np.float64)
    items = [
        SectionStiffness(EA=1e7, EI_x=float(e), EI_y=2e6, GJ=1e5, GA_x=1e6, GA_y=1e6) for e in ei
    ]
    arr = section_stiffness_array_from_sequence(s, items)
    ip = SectionPropertyInterpolator(s, arr)
    q = np.linspace(0.0, 10.0, 41, dtype=np.float64)
    out = ip.interpolate(q, allow_extrapolation=False)
    ei_q = np.asarray(out.EI_x, dtype=np.float64).ravel()
    assert np.all(np.diff(ei_q) <= 1e-6)
