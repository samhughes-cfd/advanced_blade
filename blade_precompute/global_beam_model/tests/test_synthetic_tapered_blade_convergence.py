"""Regression: synthetic tapered beam example converges under shared solver/load defaults."""

from __future__ import annotations

import numpy as np

from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import (
    RESIDUAL_THRESHOLD_REGRESSION,
    convergence_verdict,
    run_synthetic_tapered_convergence_case,
    smoke_model,
)


def test_synthetic_tapered_blade_converges() -> None:
    model = smoke_model()
    res = run_synthetic_tapered_convergence_case()
    verdict = convergence_verdict(res, residual_threshold=RESIDUAL_THRESHOLD_REGRESSION)

    assert verdict["history_nonempty"], "expected non-empty Newton iteration_history"
    assert verdict["nodal_positions_finite"]
    assert verdict["converged"], (
        f"expected converged=True, got residual_norm={res.residual_norm} "
        f"n_iterations={res.n_iterations}"
    )
    assert np.isfinite(res.residual_norm)
    assert float(res.residual_norm) < RESIDUAL_THRESHOLD_REGRESSION, (
        f"expected residual_norm < {RESIDUAL_THRESHOLD_REGRESSION}, got {res.residual_norm}"
    )
    assert verdict["ok"]

    tip = res.nodal_positions[-1] - model.X_ref[-1]
    assert float(np.linalg.norm(tip)) > 1e-6, "expected non-trivial tip motion"

    if len(res.iteration_history) >= 2:
        r_last = float(res.iteration_history[-1]["residual_norm"])
        assert r_last <= float(res.residual_norm) * (1.0 + 1e-9)
