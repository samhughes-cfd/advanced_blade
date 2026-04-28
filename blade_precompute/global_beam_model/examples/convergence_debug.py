"""Interpret Newton / load-step history and document solver knobs that drive convergence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.global_beam_model.core.types import BeamSolveResult, SolverOptions


def convergence_driver_notes() -> str:
    return """Key drivers for this 7-DOF corotated beam + proportional loading solver
---------------------------------------------------------------------------
0. Load-substep bisection after a failed NR
   Each proportional increment may be bisected. Retries must start from the last
   *converged* equilibrium at the current load factor ``prev``, not from a corrupted
   post-NR state; otherwise ``tol_res_rel_rhs`` can accept huge residuals and the ramp
   becomes unstable.

1. n_load_steps (proportional load factor lambda: 0 -> 1)
   Usually the strongest lever. Too few steps -> NR diverges or stalls on stiff
   geometric/coupled response. Increase (e.g. 48 -> 72 -> 96) before raising max_iter.

2. relax_factor (NR under-relaxation on the free-DOF increment)
   Values 0.85-0.95 stabilise difficult increments. Lower if residual spikes mid-ramp.

3. max_iter
   Cap per load increment. If non-converged and history shows oscillation, prefer more
   load steps or lower relax_factor before only raising max_iter.

4. Tolerance mix: tol_res, tol_res_rel, optional tol_res_rel_rhs, cap_floor_rel
   Stagnation handling uses these; precompute tests use tol_res_rel_rhs and cap_floor_rel
   for ill-conditioned tangents.

5. full_fd_hessian=False
   Analytic / reduced tangent path is typically faster and more stable here than full FD Hessian.

6. spin_stabilization, warping_stabilization
   Small positive values regularise the 7-DOF warping beam; zero can destabilise thin/tapered cases.

7. line_search
   Optional merit backtracking; try if residual fails to decrease with relax_factor < 1.

8. Mesh / geometry
   Initial curvature (kappa0), taper, and follower-like effects make the path harder; compare
   iteration_history |residual| across load ramp (plot_iteration_history / beam_iteration_history.png).
"""


def _history_tail(hist: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if not hist or k <= 0:
        return []
    return [dict(h) for h in hist[-k:]]


def _solver_options_public_dict(opts: SolverOptions) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for field in (
        "max_iter",
        "tol_res",
        "tol_res_rel",
        "tol_du",
        "n_gauss",
        "n_load_steps",
        "full_fd_hessian",
        "relax_factor",
        "line_search",
        "tol_res_rel_rhs",
        "cap_floor_rel",
        "spin_stabilization",
        "warping_stabilization",
        "adaptive_load_min_step",
        "adaptive_load_bisect_max",
    ):
        if hasattr(opts, field):
            d[field] = getattr(opts, field)
    return d


def build_convergence_report(
    *,
    fixture: str,
    res: BeamSolveResult,
    opts: SolverOptions,
    extra: dict[str, Any] | None = None,
    history_tail_k: int = 24,
) -> dict[str, Any]:
    hist = res.iteration_history or []
    rn = np.array([float(h.get("residual_norm", 0.0)) for h in hist], dtype=np.float64)
    report: dict[str, Any] = {
        "fixture": fixture,
        "converged": bool(res.converged),
        "n_iterations_total": int(res.n_iterations),
        "residual_norm_final": float(res.residual_norm),
        "history_len": len(hist),
        "residual_first": float(rn[0]) if rn.size else None,
        "residual_last": float(rn[-1]) if rn.size else None,
        "residual_max": float(np.max(rn)) if rn.size else None,
        "history_tail": _history_tail(hist, history_tail_k),
        "solver_options": _solver_options_public_dict(opts),
        "hints": convergence_driver_notes(),
    }
    if extra:
        report["extra"] = extra
    if not res.converged and rn.size >= 2:
        # Simple heuristic: blowing up tail
        tail = rn[-5:] if rn.size >= 5 else rn
        report["residual_tail_increasing"] = bool(tail[-1] > tail[0] * 1.1)
    return report


def write_convergence_artifacts(
    out_dir: Path,
    *,
    fixture: str,
    res: BeamSolveResult,
    opts: SolverOptions,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_convergence_report(fixture=fixture, res=res, opts=opts, extra=extra)
    json_path = (out_dir / "convergence_debug.json").resolve()
    txt_path = (out_dir / "convergence_debug.txt").resolve()
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_lines = [
        report["hints"],
        "",
        f"fixture={fixture}",
        f"converged={report['converged']}",
        f"residual_norm_final={report['residual_norm_final']}",
        f"n_iterations_total={report['n_iterations_total']}",
        f"history_len={report['history_len']}",
        f"residual_first={report.get('residual_first')}",
        f"residual_last={report.get('residual_last')}",
        f"residual_max={report.get('residual_max')}",
        "",
        "solver_options:",
        json.dumps(report["solver_options"], indent=2),
    ]
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    return json_path, txt_path
