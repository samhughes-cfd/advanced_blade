from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import yaml

import blade_precompute.beam_model as bm
import blade_precompute.section_properties as sm
from blade_precompute.beam_model.engine.interp import interp_K7, stations_from_arrays
from blade_precompute.design_optimisation.core.verification import ReferenceStation, compute_station_metrics
from blade_precompute.design_optimisation.engine import beam_k7
from blade_analysis.fatigue_damage.core.loads import ResultantHistory
from blade_analysis.fatigue_damage.core.workflows import (
    ExtremeWorkflowSpec,
    OperationalWorkflowSpec,
    validate_shared_calibration,
)
from blade_precompute.section_properties.engine.geometry import SectionDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial
from blade_precompute.section_properties.io.external_results import ExternalSectionResultSolver
from blade_precompute.section_properties.io.yaml_loader import load_section_from_yaml


def test_external_section_solver_npz_and_metrics() -> None:
    z = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    K6 = np.stack([np.eye(6) * (1.0 + i) for i in range(3)], axis=0)
    K7 = np.stack([np.eye(7) * (2.0 + i) for i in range(3)], axis=0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ext.npz"
        np.savez(p, z_stations=z, K6_stack=K6, K7_stack=K7)
        solver = ExternalSectionResultSolver.from_npz(p)
        section = SectionDefinition(station_z=1.2, subcomponents=[])
        res = solver.solve_one(section)
        assert res.K6.shape == (6, 6)
        ref = ReferenceStation("mid", 1.0, K6_ref=K6[1], K7_ref=K7[1])
        m = compute_station_metrics(res.K6, res.K7, ref)
        assert m.rel_K6_fro < 1e-12
        assert m.rel_K7_fro < 1e-12


def test_yaml_loader_warns_missing_strip_width() -> None:
    doc = {
        "station_z": 0.0,
        "ply_library": {
            "ud": {
                "E1": 40e9,
                "E2": 10e9,
                "G12": 4e9,
                "nu12": 0.28,
                "rho": 1900.0,
                "t_ply": 0.0002,
                "Xt": 1e9,
                "Xc": 1e9,
                "Yt": 1e9,
                "Yc": 1e9,
                "S12": 1e9,
                "Zt": 50e6,
                "S13": 40e6,
                "S23": 40e6,
            }
        },
        "materials": {"lam": {"ply_type": "ud", "layup": [0, 90]}},
        "subcomponents": {
            "skin": {
                "midsurface_coords": [[0.0, 0.0], [0.1, 0.0]],
                "thickness": 0.001,
                "material": "lam",
            }
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sec.yaml"
        p.write_text(yaml.safe_dump(doc), encoding="utf-8")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = load_section_from_yaml(p)
        assert any("strip_width_m" in str(item.message) for item in w)


def test_interp_k7_warns_when_synthesised() -> None:
    z = np.array([0.0, 1.0], dtype=np.float64)
    K6 = np.stack([np.eye(6), 2.0 * np.eye(6)], axis=0)
    st = stations_from_arrays(z, K6, None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = interp_K7(np.array([0.3]), st)
    assert any("synthesised warping blocks" in str(item.message) for item in w)


def test_beam_model_supports_four_point_gauss() -> None:
    L = 2.0
    n = 9
    z = np.linspace(0.0, L, n)
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 2] = z
    elems = [bm.BeamElement((i, i + 1), float(z[i + 1] - z[i]), 0.5 * (z[i] + z[i + 1])) for i in range(n - 1)]
    K6 = np.zeros((n, 6, 6), dtype=np.float64)
    for i in range(n):
        np.fill_diagonal(K6[i], [1e9, 8e5, 8e5, 2e5, 4e5, 4e5])
    K7 = np.zeros((n, 7, 7), dtype=np.float64)
    K7[:, :6, :6] = K6
    K7[:, 6, 6] = 1e4
    st = stations_from_arrays(z, K6, K7)
    model = bm.BeamModel(X_ref=X, elements=elems, section_stations=st, span_axis=2, z_node=z)
    loads = bm.BeamLoads(
        nodal_F=np.zeros((n, 3), dtype=np.float64),
        nodal_M=np.zeros((n, 3), dtype=np.float64),
        bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
    )
    loads.nodal_F[-1, 0] = 25.0
    res = bm.solve_static(model, loads, bm.SolverOptions(n_gauss=4, n_load_steps=3, max_iter=25))
    assert res.converged


def test_beam_k7_rotation_override() -> None:
    n = 3
    K7 = np.stack([np.eye(7) for _ in range(n)], axis=0)
    z = np.linspace(0.0, 2.0, n)
    from blade_precompute.design_optimisation.core.types import ExtremeLoads, OptimBladeGeometry

    ext = ExtremeLoads(
        z_stations=z,
        N=np.zeros(n),
        Vy=np.ones(n),
        Vz=np.zeros(n),
        My=np.zeros(n),
        Mz=np.zeros(n),
        T=np.zeros(n),
    )
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=np.zeros((n, 3)),
        kappa0=np.zeros((n, 3)),
        tau0=np.zeros(n),
        chord=np.ones(n),
        twist=np.zeros(n),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2)),
        subcomponent_materials={"dummy": IsotropicMaterial("d", 1.0, 0.3, 1.0, 1.0)},
    )
    R_ovr = np.stack([np.eye(3) for _ in range(n)], axis=0)
    out = beam_k7.solve(K7, ext, bg, nodal_R_override=R_ovr)
    assert out.nodal_R_source == "override"


def test_operational_extreme_workflow_validation() -> None:
    z = np.array([0.0, 1.0], dtype=np.float64)
    t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    zeros = np.zeros((t.shape[0], z.shape[0]), dtype=np.float64)
    hist = ResultantHistory(z_stations=z, time=t, N=zeros, Vy=zeros, Vz=zeros, My=zeros, Mz=zeros, T=zeros, B=zeros)
    extreme = ExtremeWorkflowSpec(z_stations=z, calibration_tag="k7_v1")
    operational = OperationalWorkflowSpec(history=hist, calibration_tag="k7_v1")
    validate_shared_calibration(extreme, operational)


def test_operational_extreme_workflow_validation_fails_on_tag_mismatch() -> None:
    z = np.array([0.0, 1.0], dtype=np.float64)
    t = np.array([0.0, 1.0], dtype=np.float64)
    zeros = np.zeros((t.shape[0], z.shape[0]), dtype=np.float64)
    hist = ResultantHistory(z_stations=z, time=t, N=zeros, Vy=zeros, Vz=zeros, My=zeros, Mz=zeros, T=zeros, B=zeros)
    extreme = ExtremeWorkflowSpec(z_stations=z, calibration_tag="A")
    operational = OperationalWorkflowSpec(history=hist, calibration_tag="B")
    try:
        validate_shared_calibration(extreme, operational)
        assert False, "Expected mismatch to raise"
    except ValueError:
        pass


def test_renamed_packages_import_shims() -> None:
    import blade_precompute.design_optimisation as blade_design_optimization
    import blade_analysis.fatigue_damage as fatigue_damage
    import blade_utilities.recovery as recovery

    assert hasattr(bm, "BeamModel")
    assert hasattr(sm, "SectionAnalysis")
    assert hasattr(recovery, "RecoveryCacheBuilder")
    assert hasattr(fatigue_damage, "FatigueAnalysis")
    assert hasattr(blade_design_optimization, "BladeDesignProblem")
