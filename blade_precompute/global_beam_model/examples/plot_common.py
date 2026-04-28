"""Shared beam solve for example plotting scripts (synthetic tapered vs GBT blade spec)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

import blade_precompute.global_beam_model as bm
from blade_precompute.global_beam_model.__main__ import _gbt_model
from blade_precompute.global_beam_model.core.types import BeamLoads, BeamSolveResult, SolverOptions
from blade_precompute.global_beam_model.examples.synthetic_tapered_blade import (
    default_beam_loads,
    default_solver_options_for_synthetic_tapered,
    smoke_model as smoke_model_synthetic_tapered,
)

FixtureKind = Literal["synthetic", "gbt"]

_EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = _EXAMPLES_DIR / "output" / "beam_smoke.pdf"
DEFAULT_PNG_DIR = _EXAMPLES_DIR / "output" / "beam_diagnostic_pngs"
DEFAULT_BLADE_SPEC = _EXAMPLES_DIR.parents[2] / "example_blade.json"


def solve_beam_examples_case(
    fixture: FixtureKind,
    *,
    blade_spec: Path | None = None,
    n_nodes: int = 17,
    load_vy: float = 350.0,
    max_iter: int = 110,
    n_load_steps: int = 72,
    verbose: bool = False,
) -> tuple[bm.BeamModel, BeamLoads, BeamSolveResult, SolverOptions]:
    """Build model, loads, and solver options; run ``solve_static``; return ``(model, loads, res, opts)``."""
    spec = Path(DEFAULT_BLADE_SPEC) if blade_spec is None else Path(blade_spec)

    if fixture == "synthetic":
        model = smoke_model_synthetic_tapered()
        loads = default_beam_loads(model, q_y_Npm=float(load_vy))
        opts = default_solver_options_for_synthetic_tapered(
            max_iter=int(max_iter),
            n_load_steps=int(n_load_steps),
            verbose=bool(verbose),
        )
    else:
        model = _gbt_model(spec, int(n_nodes))
        n = model.n_nodes
        q_line = np.zeros((len(model.elements), 3), dtype=np.float64)
        q_line[:, 1] = float(load_vy)
        loads = bm.BeamLoads(
            nodal_F=np.zeros((n, 3)),
            nodal_M=np.zeros((n, 3)),
            distributed_q=q_line,
            bcs=[bm.BoundaryCondition(0, tuple(range(7)))],
        )
        # Match ``test_precompute_example_blade10_beam_converges`` / precompute-style NR.
        opts = SolverOptions(
            max_iter=int(max_iter),
            tol_res=5e-2,
            tol_res_rel=5e-3,
            tol_du=1e-6,
            n_gauss=2,
            n_load_steps=int(n_load_steps),
            full_fd_hessian=False,
            spin_stabilization=1e-5,
            warping_stabilization=1e-3,
            relax_factor=0.9,
            line_search=False,
            tol_res_rel_rhs=0.035,
            cap_floor_rel=0.055,
            verbose=bool(verbose),
        )

    res = bm.BeamAnalysis(model).solve_static(loads, options=opts)
    return model, loads, res, opts
