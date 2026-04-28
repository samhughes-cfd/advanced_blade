"""Integration tests: section stations and global beam solve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.core.types import (
    BeamElement,
    BeamLoads,
    BeamModel,
    BoundaryCondition,
    K7Array,
    SectionStation,
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
def example_blade_spec() -> Path:
    p = _repo_root() / "example_blade_10.json"
    if not p.is_file():
        pytest.skip(f"Missing {p}")
    return p


def test_section_to_global_pipeline(example_blade_spec: Path) -> None:
    """End-to-end beam static solve using synthetic positive-definite section stiffnesses."""
    bg = BladeDesignProblem.load_geometry(example_blade_spec)
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


def _make_random_element_state(rng: np.random.Generator):
    """Return (x1,q1,p1,x2,q2,p2,L0) for a mildly deformed single element."""
    from blade_precompute.global_beam_model.engine.kinematics import (
        exp_so3, update_orientation,
    )
    L0 = 1.5
    x1 = rng.standard_normal(3) * 0.1
    x2 = x1 + np.array([L0, 0.0, 0.0]) + rng.standard_normal(3) * 0.05
    th1 = rng.standard_normal(3) * 0.15
    th2 = rng.standard_normal(3) * 0.15
    q1 = update_orientation(np.array([1.0, 0.0, 0.0, 0.0]), th1)
    q2 = update_orientation(np.array([1.0, 0.0, 0.0, 0.0]), th2)
    p1 = float(rng.standard_normal()) * 0.01
    p2 = float(rng.standard_normal()) * 0.01
    return x1, q1, p1, x2, q2, p2, L0


def test_analytical_B_vs_fd() -> None:
    """Analytical B-matrix columns must match FD columns to within FD tolerance."""
    from blade_precompute.global_beam_model.engine.element import (
        _analytical_B_gp, _apply_endpoint_bump, e7_from_endpoint_states,
        _reuse_R_gp_for_fd_col, _R_interp, _shape,
    )

    rng = np.random.default_rng(42)
    x1, q1, p1, x2, q2, p2, L0 = _make_random_element_state(rng)
    kappa0 = np.zeros(3)
    chi0 = 0.0
    xi_f = 0.3
    N1, N2, dN1, dN2 = _shape(xi_f)
    jac_eps = 1e-5

    B_anal = _analytical_B_gp(x1, q1, x2, q2, L0, dN1, dN2, xi_f)

    R_gp = _R_interp(q1, q2, xi_f)
    B_fd = np.zeros((7, 14), dtype=np.float64)
    for col in range(14):
        r_use = R_gp if _reuse_R_gp_for_fd_col(col) else None
        x1p, q1p, p1p, x2p, q2p, p2p = _apply_endpoint_bump(col, +1.0, jac_eps, x1, q1, p1, x2, q2, p2)
        x1m, q1m, p1m, x2m, q2m, p2m = _apply_endpoint_bump(col, -1.0, jac_eps, x1, q1, p1, x2, q2, p2)
        ep = e7_from_endpoint_states(x1p, q1p, p1p, x2p, q2p, p2p, L0, kappa0, chi0, dN1, dN2, r_use, xi_f)
        em = e7_from_endpoint_states(x1m, q1m, p1m, x2m, q2m, p2m, L0, kappa0, chi0, dN1, dN2, r_use, xi_f)
        B_fd[:, col] = (ep - em) / (2.0 * jac_eps)

    np.testing.assert_allclose(B_anal, B_fd, atol=1e-4, rtol=1e-4,
                               err_msg="Analytical B-matrix differs from FD B-matrix")


def test_extract_buckling_returns_eigenvalues() -> None:
    """extract_buckling=True populates global_buckling_lambdas with n_buckling_modes entries.

    Uses a self-contained synthetic cantilever so the test never needs an external blade spec.
    """
    from blade_precompute.global_beam_model.engine.constitutive import synthesize_K7
    import warnings

    n_nodes = 8
    L = 10.0
    z = np.linspace(0.0, L, n_nodes, dtype=np.float64)
    X_ref = np.zeros((n_nodes, 3), dtype=np.float64)
    X_ref[:, 2] = z

    K6 = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K7 = synthesize_K7(K6)
    stations = [SectionStation(z=float(zi), K6=K6.copy(), K7=K7.copy()) for zi in z]

    elements = [
        BeamElement(node_ids=(i, i + 1), L0=float(z[i + 1] - z[i]), z_mid=float(0.5 * (z[i] + z[i + 1])))
        for i in range(n_nodes - 1)
    ]
    model = BeamModel(X_ref=X_ref, elements=elements, section_stations=stations)
    beam = BeamAnalysis(model)

    F = np.zeros((n_nodes, 3), dtype=np.float64)
    F[-1, 1] = 1.0  # small lateral tip load — well inside linear regime, fast NR convergence
    loads = BeamLoads(
        nodal_F=F,
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    n_modes = 3
    opts = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=4,
        full_fd_hessian=False,  # material-only tangent: fast for test
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
        extract_buckling=True,
        n_buckling_modes=n_modes,
    )
    res = beam.solve_static(loads, options=opts)
    assert res.converged, "beam did not converge — cannot test buckling extraction"
    assert res.global_buckling_lambdas is not None
    lam = np.asarray(res.global_buckling_lambdas, dtype=np.float64)
    assert lam.shape == (n_modes,), f"expected ({n_modes},), got {lam.shape}"
    assert np.all(np.isfinite(lam)), "buckling eigenvalues contain non-finite values"
    assert float(lam[0]) > 0.0, "lowest buckling load factor must be positive"


def _make_synthetic_cantilever(n_nodes: int = 8, L: float = 10.0):
    """Return (BeamAnalysis, BeamLoads) for a straight cantilever with unit tip lateral load."""
    from blade_precompute.global_beam_model.engine.constitutive import synthesize_K7
    import warnings

    z = np.linspace(0.0, L, n_nodes, dtype=np.float64)
    X_ref = np.zeros((n_nodes, 3), dtype=np.float64)
    X_ref[:, 2] = z
    K6 = np.diag([1e8, 1e6, 1e6, 1e5, 1e6, 1e6]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K7 = synthesize_K7(K6)
    stations = [SectionStation(z=float(zi), K6=K6.copy(), K7=K7.copy()) for zi in z]
    elements = [
        BeamElement(node_ids=(i, i + 1), L0=float(z[i + 1] - z[i]), z_mid=float(0.5 * (z[i] + z[i + 1])))
        for i in range(n_nodes - 1)
    ]
    model = BeamModel(X_ref=X_ref, elements=elements, section_stations=stations)
    beam = BeamAnalysis(model)
    F = np.zeros((n_nodes, 3), dtype=np.float64)
    F[-1, 1] = 1.0
    loads = BeamLoads(
        nodal_F=F,
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    return beam, loads


def test_line_search_converges() -> None:
    """line_search=True should converge on the same cantilever as the base solver."""
    beam, loads = _make_synthetic_cantilever()
    opts = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=4,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
        line_search=True,
        line_search_shrink=0.5,
        line_search_max_trials=8,
    )
    res = beam.solve_static(loads, options=opts)
    assert res.converged, "line_search=True: beam did not converge"
    tip = np.asarray(res.nodal_positions[-1] - beam.model.X_ref[-1], dtype=np.float64)
    assert float(np.linalg.norm(tip)) > 0.0


def test_adaptive_load_stepping_converges() -> None:
    """adaptive_load_min_step and adaptive_load_bisect_max should not break convergence."""
    beam, loads = _make_synthetic_cantilever()
    opts = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=6,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
        adaptive_load_min_step=1e-3,
        adaptive_load_bisect_max=8,
    )
    res = beam.solve_static(loads, options=opts)
    assert res.converged, "adaptive load stepping: beam did not converge"
    tip = np.asarray(res.nodal_positions[-1] - beam.model.X_ref[-1], dtype=np.float64)
    assert float(np.linalg.norm(tip)) > 0.0


def test_project_fd_hessian_spd_runs() -> None:
    """project_fd_hessian_spd with full_fd_hessian should converge and yield finite tip displacement."""
    beam, loads = _make_synthetic_cantilever(n_nodes=6)
    opts = SolverOptions(
        max_iter=50,
        tol_res=1e-1,
        tol_res_rel=1e-2,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=2,
        full_fd_hessian=True,
        project_fd_hessian_spd=True,
        fd_hessian_eig_floor_rel=1e-10,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = beam.solve_static(loads, options=opts)
    assert res.converged, "project_fd_hessian_spd: beam did not converge"
    tip = np.asarray(res.nodal_positions[-1] - beam.model.X_ref[-1], dtype=np.float64)
    assert np.all(np.isfinite(tip))


def test_cs_gradient_vs_analytical_B() -> None:
    """CS gradient must match B^T r to near machine precision."""
    from blade_precompute.global_beam_model.core.types import (
        BeamElement, BeamModel, NodeState, SectionStation,
    )
    from blade_precompute.global_beam_model.engine.element import (
        _analytical_B_gp, _shape, _interp_kappa0_N, _interp_chi0_N,
        e7_from_endpoint_states, infer_z_node, _lagrange_gauss,
        element_energy_gradient_cs,
    )
    from blade_precompute.global_beam_model.engine.constitutive import (
        section_resultants_natural, synthesize_K7,
    )
    import warnings

    rng = np.random.default_rng(7)
    x1, q1, p1, x2, q2, p2, L0 = _make_random_element_state(rng)

    K6 = np.diag([1e7, 2e6, 3e6, 1e5, 1e6, 1e6]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K7 = synthesize_K7(K6)

    stations = [
        SectionStation(z=0.0, K6=K6, K7=K7),
        SectionStation(z=2.0, K6=K6, K7=K7),
    ]
    X_ref = np.stack([x1, x2], axis=0)
    el = BeamElement(node_ids=(0, 1), L0=L0, z_mid=L0 / 2)
    model = BeamModel(X_ref=X_ref, elements=[el], section_stations=stations)
    nodes = [
        NodeState(x=x1.copy(), q=q1.copy(), psi=p1),
        NodeState(x=x2.copy(), q=q2.copy(), psi=p2),
    ]

    n_gauss = 2
    xi_w, w_w = _lagrange_gauss(n_gauss)
    ztab = infer_z_node(model)

    # Analytical gradient: sum over GPs of fac * B^T K7 e7
    g_anal = np.zeros(14)
    for xi, w in zip(xi_w, w_w):
        xi_f = float(xi)
        N1, N2, dN1, dN2 = _shape(xi_f)
        kappa0 = _interp_kappa0_N(model, el, N1, N2)
        chi0v = _interp_chi0_N(model, el, N1, N2)
        fac = 0.5 * L0 * float(w)
        B = _analytical_B_gp(x1, q1, x2, q2, L0, dN1, dN2, xi_f)
        e7 = e7_from_endpoint_states(x1, q1, p1, x2, q2, p2, L0, kappa0, chi0v, dN1, dN2, None, xi_f)
        r_nat = section_resultants_natural(K7, e7)
        g_anal += fac * B.T @ r_nat

    g_cs = element_energy_gradient_cs(model, el, nodes, stations, n_gauss)

    np.testing.assert_allclose(g_cs, g_anal, atol=1e-8, rtol=1e-6,
                               err_msg="CS gradient differs from B^T r")
