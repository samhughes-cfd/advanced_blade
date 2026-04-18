"""Integration tests: GBT-derived section stations and global beam solve."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, K7Array, SolverOptions
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.k7_interpolation import K7Interpolator
from blade_precompute.global_beam_model.section_property_interpolator import (
    SectionPropertyInterpolator,
    section_stiffness_array_from_sequence,
)
from blade_precompute.orchestration.gbt_beam_stations import beam_section_stations_from_gbt
from blade_precompute.section_beam_model.gbt import (
    CrossSection,
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    IsotropicMaterial,
    SectionLoads,
    WallDefinition,
    select_modes,
)
from blade_precompute.section_beam_model.gbt.section_stiffness_export import (
    SectionStiffness,
    gbt_to_beam_stiffness,
    gbt_to_k7,
    section_stiffness_to_k6,
)
from blade_precompute.section_optimisation.api import BladeDesignProblem
from blade_precompute.section_optimisation.core.types import DesignVector
from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder

_GBT_MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)


def _make_gbt_box_section(b: float = 0.06, h: float = 0.20) -> CrossSection:
    hb, hh = 0.5 * b, 0.5 * h
    return CrossSection(
        [
            WallDefinition([-hb, -hh], [hb, -hh], _GBT_MAT, n_strips=8, name="bot"),
            WallDefinition([hb, -hh], [hb, hh], _GBT_MAT, n_strips=4, name="right"),
            WallDefinition([hb, hh], [-hb, hh], _GBT_MAT, n_strips=8, name="top"),
            WallDefinition([-hb, hh], [-hb, -hh], _GBT_MAT, n_strips=4, name="left"),
        ]
    )


def _make_gbt_c_section() -> CrossSection:
    return CrossSection(
        [
            WallDefinition([0, 0], [0, -0.1], _GBT_MAT, n_strips=4, name="web"),
            WallDefinition([0, 0], [0.05, 0], _GBT_MAT, n_strips=2, name="top"),
            WallDefinition([0, -0.1], [0.05, -0.1], _GBT_MAT, n_strips=2, name="bot"),
        ]
    )


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


def test_k7_interpolator_symmetry() -> None:
    rng = np.random.default_rng(0)
    n_st = 5
    s = np.linspace(0.0, 1.0, n_st, dtype=np.float64)
    mats: list[np.ndarray] = []
    for _ in range(n_st):
        a = rng.standard_normal((7, 7))
        sk = a @ a.T + np.eye(7)
        mats.append(0.5 * (sk + sk.T))
    entries = np.stack(mats, axis=0)
    arr = K7Array(s=s, entries=entries)
    ip = K7Interpolator(arr)
    zq = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    out = ip.interpolate(zq)
    for i in range(out.entries.shape[0]):
        k = out.entries[i]
        assert np.allclose(k, k.T, atol=1e-12)


def test_k7_interpolator_smooth() -> None:
    """PCHIP matches affine ``K7[3,6](s)`` exactly; smooth nonlinear targets need dense stations."""
    n_st = 20
    s = np.linspace(0.0, 1.0, n_st, dtype=np.float64)
    entries = np.zeros((n_st, 7, 7), dtype=np.float64)
    for i, si in enumerate(s):
        np.fill_diagonal(entries[i], 1e6)
        entries[i, 6, 6] = 1e4
        v = 0.3 * float(si) + 0.02
        entries[i, 3, 6] = v
        entries[i, 6, 3] = v
    arr = K7Array(s=s, entries=entries)
    ip = K7Interpolator(arr)
    zq = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    out = ip.interpolate(zq)
    for i in range(zq.shape[0]):
        zi = float(zq[i])
        pred = float(out.entries[i, 3, 6])
        an = 0.3 * zi + 0.02
        assert abs(pred - an) < 1e-9


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


def _gbt_box_stiffness_k6_k7() -> tuple[np.ndarray, np.ndarray]:
    sec = _make_gbt_box_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6)
    return K6, K7


def test_gbt_to_k7_shape_and_symmetry() -> None:
    K6, K7 = _gbt_box_stiffness_k6_k7()
    assert K7.shape == (7, 7)
    assert np.allclose(K7, K7.T)
    w = np.linalg.eigvalsh(0.5 * (K7 + K7.T))
    assert float(w.min()) >= -1e-6 * max(float(np.max(np.diag(K7))), 1.0)


def test_gbt_to_k7_k6_block() -> None:
    K6, K7 = _gbt_box_stiffness_k6_k7()
    assert np.allclose(K7[:6, :6], K6)


def test_gbt_to_k7_warping_positive() -> None:
    _, K7 = _gbt_box_stiffness_k6_k7()
    assert float(K7[6, 6]) > 0.0


def test_gbt_to_k7_full_vlasov() -> None:
    sec = _make_gbt_c_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=40)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    K6 = section_stiffness_to_k6(st)
    K7 = gbt_to_k7(full, K6, full_vlasov=True)
    assert any(abs(float(K7[i, 6])) > 0.0 for i in range(4))


def test_eiyz_nonzero_in_k6() -> None:
    st = SectionStiffness(
        EA=1e7,
        EI_x=2e6,
        EI_y=3e6,
        GJ=1e5,
        GA_x=1e6,
        GA_y=1e6,
        EIyz=1e6,
    )
    K6 = section_stiffness_to_k6(st)
    assert K6[1, 2] == K6[2, 1] == pytest.approx(-1e6)


def test_eiyz_survives_interpolation() -> None:
    s = np.array([0.0, 1.0], dtype=np.float64)
    base = dict(EA=1e7, EI_x=2e6, EI_y=3e6, GJ=1e5, GA_x=1e6, GA_y=1e6)
    items = [
        SectionStiffness(**base, EIyz=0.0),
        SectionStiffness(**base, EIyz=1e6),
    ]
    arr = section_stiffness_array_from_sequence(s, items)
    ip = SectionPropertyInterpolator(s, arr)
    out = ip.interpolate(np.array([0.5], dtype=np.float64))
    e = float(np.asarray(out.EIyz, dtype=np.float64).ravel()[0])
    assert 0.0 < e < 1e6


def test_gbt_to_beam_stiffness_loads_deprecated() -> None:
    sec = _make_gbt_box_section()
    loads = SectionLoads(N=-1.0)
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gbt_to_beam_stiffness(full, sel, section=sec, loads=loads)
    assert any(w.category is DeprecationWarning for w in rec)
