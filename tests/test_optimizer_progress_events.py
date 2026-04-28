from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_precompute._utils.job_progress import JobProgressReporter
from blade_precompute.section_optimisation.core.types import DesignVector
from blade_precompute.section_optimisation.engine.optimizer import BladeOptimizer


class _EventProgress:
    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[tuple[str, dict]] = []

    def event(self, phase: str, **meta: object) -> None:
        self.calls.append((phase, dict(meta)))


class _JsonlOnlyProgress:
    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[tuple[str, dict]] = []

    def event_jsonl_only(self, phase: str, **meta: object) -> None:
        self.calls.append((phase, dict(meta)))


def test_optimizer_iteration_progress_uses_terminal_event_when_available() -> None:
    progress = _EventProgress()
    fake_self = SimpleNamespace(_progress=progress)
    ev = SimpleNamespace(mass=12.34, max_fi_hashin=0.87, stiffness_metric=456.7)

    BladeOptimizer._emit_optimizer_iteration_progress(
        fake_self,  # type: ignore[arg-type]
        iteration=3,
        elapsed_opt_s=7.89,
        ev=ev,
    )

    assert len(progress.calls) == 1
    phase, meta = progress.calls[0]
    assert phase == "optimizer_slsqp_iteration"
    assert meta["iteration"] == 3
    assert meta["elapsed_since_optimizer_start_s"] == 7.89
    assert meta["mass_kg"] == 12.34
    assert meta["max_fi_hashin"] == 0.87
    assert meta["stiffness_metric"] == 456.7


def test_optimizer_iteration_progress_falls_back_to_jsonl_only() -> None:
    progress = _JsonlOnlyProgress()
    fake_self = SimpleNamespace(_progress=progress)
    ev = SimpleNamespace(mass=1.0, max_fi_hashin=2.0, stiffness_metric=3.0)

    BladeOptimizer._emit_optimizer_iteration_progress(
        fake_self,  # type: ignore[arg-type]
        iteration=1,
        elapsed_opt_s=0.25,
        ev=ev,
    )

    assert len(progress.calls) == 1
    phase, meta = progress.calls[0]
    assert phase == "optimizer_slsqp_iteration"
    assert meta["iteration"] == 1
    assert meta["elapsed_since_optimizer_start_s"] == 0.25


def test_blade_optimizer_emits_terminal_iteration_event_in_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeEvaluator:
        def evaluate(self, _dv: DesignVector):
            fi_hashin = np.array([[[0.2]]], dtype=np.float64)
            fi_vm = np.array([0.1], dtype=np.float64)
            return SimpleNamespace(
                mass=10.0,
                stiffness_metric=20.0,
                fi_hashin=fi_hashin,
                max_fi_hashin=0.2,
                fi_vm=fi_vm,
                max_fi_vm=0.1,
                fi_panel_buckling=None,
                global_buckling_lambdas=None,
            )

    class _FakeResult:
        def __init__(self, x: np.ndarray) -> None:
            self.x = x
            self.success = True
            self.message = "ok"
            self.nit = 1
            self.fun: float = 0.0

    def _fake_minimize(objective, x0, method, bounds, constraints, options, callback):
        del method, bounds, constraints, options
        f0 = float(objective(x0))
        callback(x0)
        r = _FakeResult(np.asarray(x0, dtype=np.float64))
        r.fun = f0
        return r

    monkeypatch.setattr(
        "blade_precompute.section_optimisation.engine.optimizer.minimize",
        _fake_minimize,
    )

    fake_problem = SimpleNamespace(
        ks_rho=35.0,
        ks_rho_buckling=25.0,
        objective="min_mass",
        stress_recovery="mitc4",
        enable_panel_buckling=False,
        enable_global_buckling=False,
        enforce_spanwise_monotone=False,
        optimizer_method="SLSQP",
        optimizer_ftol=1e-5,
        optimizer_n_restarts=0,
        optimizer_multistart_seed=None,
    )
    progress = JobProgressReporter(tmp_path / "job", enabled=True)
    progress.phase_start("blade_optimizer_slsqp", max_iter=1)
    opt = BladeOptimizer(
        fake_problem,
        evaluator=_FakeEvaluator(),
        progress=progress,
        options={"maxiter": 1, "ftol": 1e-5, "disp": False},
    )
    dv0 = DesignVector(
        t_skin=np.array([0.01], dtype=np.float64),
        t_cap=np.array([0.02], dtype=np.float64),
        t_web=np.array([0.03], dtype=np.float64),
    )
    opt.run(dv0)
    progress.phase_end("blade_optimizer_slsqp", success=True, n_iter=1)
    out = capsys.readouterr().out
    assert "[precompute] | event | optimizer_slsqp_iteration |" in out
