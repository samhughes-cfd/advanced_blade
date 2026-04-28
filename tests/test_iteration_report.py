"""Tests for optimisation iteration logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blade_precompute.global_beam_model.core.types import BeamSolveResult
from blade_precompute.section_optimisation.core.types import (
    DesignEvaluation,
    DesignProblem,
    DesignVector,
    ExtremeLoads,
    OptimBladeGeometry,
    normalize_stress_recovery,
)
from blade_precompute.section_optimisation.engine.iteration_report import (
    build_optimizer_iteration_payload,
    counts_hashin_unity,
    objective_scalar,
    top_k_hashin_hotspots,
    write_iteration_npz,
    write_iteration_payload_schema,
)


def _minimal_bg(n: int = 3) -> OptimBladeGeometry:
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    r = np.stack([np.zeros(n), np.zeros(n), z], axis=1)
    return OptimBladeGeometry(
        z_stations=z,
        r_ref=r,
        kappa0=np.zeros((n, 3), dtype=np.float64),
        chord=np.ones(n, dtype=np.float64),
        twist=np.zeros(n, dtype=np.float64),
        airfoil_profiles=[],
        web_positions=np.zeros((0, 2), dtype=np.float64),
        subcomponent_materials={},
    )


def _minimal_problem() -> DesignProblem:
    bg = _minimal_bg(3)
    z = np.asarray(bg.z_stations, dtype=np.float64)
    n = int(z.shape[0])
    ex = ExtremeLoads(
        z_stations=z,
        N=np.zeros(n),
        Vy=np.zeros(n),
        Vz=np.zeros(n),
        My=np.zeros(n),
        Mz=np.zeros(n),
        T=np.zeros(n),
    )
    return DesignProblem(blade_geometry=bg, extreme_loads=ex, objective="min_mass")


def test_normalize_stress_recovery_mitc4_only() -> None:
    assert normalize_stress_recovery("mitc4") == "mitc4"
    with pytest.warns(DeprecationWarning):
        assert normalize_stress_recovery("strip_clpt") == "mitc4"
    with pytest.warns(DeprecationWarning):
        assert normalize_stress_recovery("both") == "mitc4"
    with pytest.raises(ValueError, match="Unknown stress_recovery"):
        normalize_stress_recovery("legacy_unknown")


def _minimal_ev(fi_h: np.ndarray) -> DesignEvaluation:
    dv = DesignVector(
        t_skin=np.full(3, 0.01),
        t_cap=np.full(3, 0.02),
        t_web=np.full(3, 0.015),
    )
    return DesignEvaluation(
        dv=dv,
        mass=1.0,
        stiffness_metric=2.0,
        resultants=np.zeros((3, 7), dtype=np.float64),
        fi_hashin=np.asarray(fi_h, dtype=np.float64),
        fi_vm=np.zeros((3, 0), dtype=np.float64),
        max_fi_hashin=float(np.max(fi_h)),
        max_fi_vm=0.0,
    )


def test_objective_scalar_min_mass() -> None:
    ev = _minimal_ev(np.ones((3, 2, 4)))
    assert objective_scalar(ev, "min_mass") == pytest.approx(1.0)


def test_top_k_and_counts() -> None:
    fh = np.array(
        [
            [[10.0, 0.5], [0.2, 0.3]],
            [[0.1, 0.2], [2.0, 0.4]],
        ],
        dtype=np.float64,
    )
    names = ["A", "B"]
    hot = top_k_hashin_hotspots(fh, k=3, composite_names=names)
    assert len(hot) == 3
    assert hot[0]["fi_hashin"] == pytest.approx(10.0)
    assert hot[0]["subcomp_name"] == "A"
    c = counts_hashin_unity(fh)
    assert c["n_stations_max_fi_gt_1"] == 2
    assert c["n_subcomp_max_fi_gt_1"] == 2


def test_build_iteration_payload_jsonable() -> None:
    p = _minimal_problem()
    ev = _minimal_ev(np.array([[[1.5, 0.1], [0.2, 0.3]]], dtype=np.float64))
    out = build_optimizer_iteration_payload(
        ev,
        p,
        iteration=1,
        prev_objective=None,
        hotspot_k=4,
        composite_names=["c0", "c1"],
        isotropic_names=[],
        z_stations_m=np.asarray(p.blade_geometry.z_stations, dtype=np.float64),
        axis_meta_emitted=False,
    )
    assert out["iteration"] == 1
    assert out["slack_ks_hashin"] < 0.0
    assert "top_k_hashin_hotspots" in out
    assert "t_skin" in out
    assert out["n_stations_max_fi_gt_1"] == 1
    assert out["delta_t_skin_max_abs_m"] is None
    assert out["stations_changed_skin"] is None


def test_build_iteration_payload_thickness_deltas_prev_ev() -> None:
    p = _minimal_problem()
    fh = np.ones((3, 1, 2), dtype=np.float64) * 0.3
    prev = _minimal_ev(fh)
    curr_dv = DesignVector(
        t_skin=np.array([0.01, 0.02, 0.01]),
        t_cap=np.array([0.02, 0.02, 0.02]),
        t_web=np.array([0.015, 0.015, 0.015]),
    )
    curr = DesignEvaluation(
        dv=curr_dv,
        mass=1.0,
        stiffness_metric=2.0,
        resultants=np.zeros((3, 7), dtype=np.float64),
        fi_hashin=fh,
        fi_vm=np.zeros((3, 0), dtype=np.float64),
        max_fi_hashin=float(np.max(fh)),
        max_fi_vm=0.0,
    )
    out = build_optimizer_iteration_payload(
        curr,
        p,
        iteration=2,
        prev_objective=0.5,
        prev_ev=prev,
        hotspot_k=2,
        composite_names=None,
        isotropic_names=None,
        z_stations_m=None,
        axis_meta_emitted=True,
    )
    assert out["delta_t_skin_max_abs_m"] == pytest.approx(0.01)
    assert out["delta_t_skin_l2_m"] == pytest.approx(0.01)
    assert out["stations_changed_skin"] == [1]
    assert out["constraint_deltas_max_abs_ks_slack"] is not None
    assert "delta_objective" in out


def test_build_iteration_payload_beam_summary_and_nr_trunc() -> None:
    p = _minimal_problem()
    p.iteration_log_beam_summary = True
    p.iteration_log_beam_nr_history = True
    n_s = 3
    hist: list[dict[str, Any]] = [{"iter": i, "residual_norm": 1.0 / (i + 1), "displacement_norm": 0.1} for i in range(10)]
    beam = BeamSolveResult(
        nodal_positions=np.zeros((2, 3)),
        nodal_rotations=np.zeros((2, 3)),
        nodal_R=np.zeros((2, 3, 3)),
        nodal_warping=np.zeros(2),
        resultants=np.ones((n_s, 7)),
        strains=np.ones((n_s, 7)) * 2.0,
        converged=True,
        n_iterations=4,
        residual_norm=1e-8,
        iteration_history=hist,
    )
    dv = DesignVector(
        t_skin=np.full(3, 0.01),
        t_cap=np.full(3, 0.02),
        t_web=np.full(3, 0.015),
    )
    ev = DesignEvaluation(
        dv=dv,
        mass=1.0,
        stiffness_metric=2.0,
        resultants=np.zeros((3, 7), dtype=np.float64),
        fi_hashin=np.ones((3, 1, 2), dtype=np.float64) * 0.3,
        fi_vm=np.zeros((3, 0), dtype=np.float64),
        max_fi_hashin=0.3,
        max_fi_vm=0.0,
        beam_state=beam,
    )
    out = build_optimizer_iteration_payload(
        ev,
        p,
        iteration=1,
        prev_objective=None,
        hotspot_k=2,
        composite_names=None,
        isotropic_names=None,
        z_stations_m=None,
        axis_meta_emitted=True,
    )
    assert out["beam_converged"] is True
    assert out["beam_n_iterations"] == 4
    assert out["beam_resultants_norm"] > 0.0
    assert out["beam_strains_norm"] > 0.0
    assert "beam_nr_history_trunc" in out
    assert any(isinstance(x, dict) and x.get("_truncated") for x in out["beam_nr_history_trunc"])


def test_iteration_payload_stable_delta_keys() -> None:
    """Guardrail: first-iteration null deltas must remain explicit keys for JSON consumers."""
    p = _minimal_problem()
    ev = _minimal_ev(np.ones((3, 1, 2), dtype=np.float64) * 0.3)
    out = build_optimizer_iteration_payload(
        ev,
        p,
        iteration=1,
        prev_objective=None,
        hotspot_k=1,
        composite_names=None,
        isotropic_names=None,
        z_stations_m=None,
        axis_meta_emitted=True,
    )
    for role in ("skin", "cap", "web"):
        assert f"delta_t_{role}_max_abs_m" in out
        assert f"stations_changed_{role}" in out


def test_build_iteration_payload_k7_spanwise_and_mix_id() -> None:
    p = _minimal_problem()
    p.iteration_log_k7_spanwise = True
    k7 = np.stack([np.eye(7) * float(i + 1) for i in range(3)], axis=0)
    ev = _minimal_ev(np.ones((3, 1, 2), dtype=np.float64) * 0.3)
    out = build_optimizer_iteration_payload(
        ev,
        p,
        iteration=1,
        prev_objective=None,
        hotspot_k=1,
        composite_names=None,
        isotropic_names=None,
        z_stations_m=None,
        axis_meta_emitted=True,
        k7_stack=k7,
        orientation_mix_id='{"cap":{"n_0":1}}',
    )
    assert out["k7_diag_0_min"] == pytest.approx(1.0)
    assert out["orientation_mix_id"] == '{"cap":{"n_0":1}}'


def test_write_iteration_npz_and_schema(tmp_path: Path) -> None:
    write_iteration_payload_schema(tmp_path / "iteration_payload_schema.json")
    assert (tmp_path / "iteration_payload_schema.json").is_file()
    raw = (tmp_path / "iteration_payload_schema.json").read_text(encoding="utf-8")
    assert "beam_nr_residual_tail" in raw
    assert "optimizer.orientation_combo" in raw

    ev = _minimal_ev(np.ones((2, 1, 2)) * 0.5)
    n_s = 2
    beam = BeamSolveResult(
        nodal_positions=np.zeros((2, 3)),
        nodal_rotations=np.zeros((2, 3)),
        nodal_R=np.zeros((2, 3, 3)),
        nodal_warping=np.zeros(2),
        resultants=np.zeros((n_s, 7)),
        strains=np.ones((n_s, 7)) * 0.25,
        converged=True,
        n_iterations=1,
        residual_norm=1e-9,
        iteration_history=[{"residual_norm": 0.5, "displacement_norm": 0.01}],
    )
    ev2 = DesignEvaluation(
        dv=ev.dv,
        mass=ev.mass,
        stiffness_metric=ev.stiffness_metric,
        resultants=ev.resultants,
        fi_hashin=ev.fi_hashin,
        fi_vm=ev.fi_vm,
        max_fi_hashin=ev.max_fi_hashin,
        max_fi_vm=ev.max_fi_vm,
        beam_state=beam,
    )
    pth = tmp_path / "t.npz"
    k7s = np.stack([np.eye(7) for _ in range(n_s)], axis=0)
    write_iteration_npz(
        pth,
        ev=ev2,
        strip={"strip_abd_inv": np.zeros((2, 1, 6, 6))},
        mitc4_station0_panel_abd=np.zeros((1, 6, 6)),
        mitc4_station0_panel_thickness_m=np.array([0.01]),
        mitc4_station0_panel_G_eff=np.array([1e8]),
        mitc4_station0_panel_labels=np.array(["p0"], dtype=object),
        k7_stack=k7s,
        beam_nr_residual_tail=np.array([0.1, 0.05]),
    )
    d = np.load(pth)
    assert "fi_hashin" in d.files
    assert "strip_abd_inv" in d.files
    assert "strains" in d.files
    assert "K7_stack" in d.files
    assert "beam_nr_residual_tail" in d.files
