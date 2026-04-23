"""Integration tests: section stations and global beam solve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import (
    BeamLoads,
    BoundaryCondition,
    K7Array,
    SectionStiffness,
    SolverOptions,
)
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.global_beam_model.k7_interpolation import K7Interpolator
from blade_precompute.global_beam_model.section_property_interpolator import (
    SectionPropertyInterpolator,
    section_stiffness_array_from_sequence,
)
from blade_precompute.section_optimisation.api import BladeDesignProblem


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def example_blade_yaml() -> Path:
    p = _repo_root() / "example_blade.yaml"
    if not p.is_file():
        pytest.skip(f"Missing {p}")
    return p


def test_section_to_global_pipeline(example_blade_yaml: Path) -> None:
    """End-to-end beam static solve using synthetic positive-definite section stiffnesses."""
    bg = BladeDesignProblem.load_geometry(example_blade_yaml)
    z = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    n = int(z.shape[0])
    K6_template = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    K7_template = np.zeros((7, 7), dtype=np.float64)
    K7_template[:6, :6] = K6_template
    K7_template[6, 6] = 1e4
    K6 = np.stack([K6_template.copy() for _ in range(n)], axis=0)
    K7 = np.stack([K7_template.copy() for _ in range(n)], axis=0)
    stations = stations_from_arrays(z, K6, K7)

    assert all(s.K7 is not None for s in stations)
    assert all(s.K7.shape == (7, 7) for s in stations if s.K7 is not None)
    assert all(float(s.K7[6, 6]) > 0.0 for s in stations if s.K7 is not None)
    for s in stations:
        assert s.K7 is not None
        K7m = np.asarray(s.K7, dtype=np.float64)
        emin = float(np.linalg.eigvalsh(0.5 * (K7m + K7m.T)).min())
        assert emin >= -1e-2 * max(float(np.max(np.diag(K7m))), 1.0)

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
    alpha = 5.0 / 6.0
    K6 = np.zeros((6, 6), dtype=np.float64)
    K6[0, 0] = st.EA
    K6[1, 1] = st.EI_x
    K6[2, 2] = st.EI_y
    K6[1, 2] = K6[2, 1] = -float(st.EIyz)
    K6[3, 3] = max(st.GJ, 1e-12)
    K6[4, 4] = alpha * max(st.GA_x, 1e-12)
    K6[5, 5] = alpha * max(st.GA_y, 1e-12)
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
