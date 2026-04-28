"""Gradient-based NL optimiser (SciPy ``SLSQP`` / ``trust-constr``) with KS failure constraints.

Population heuristics (differential evolution, genetic algorithms) are **not** the default: each
outer iteration already runs a full structural pipeline, so black-box global search is usually
orders of magnitude more expensive unless you add parallel evaluation budgets, coarse surrogates, or
similar. Prefer fixing the constraint set (see ``mitc4`` + Hashin), ``trust-constr`` / multi-start, or
feasibility-first two-phase steps before global methods.

Group J — Buckling Constraints
-------------------------------
J.2  Panel buckling KS:   ``c_ks_panel_buckling = 1 - KS(fi_buck, rho_buck) >= 0``
     Separate ``ks_rho_buckling`` (default 25) so that the aggregation is sharper
     than the strength KS (rho=35).
J.4  Global buckling:     ``c_global_buckling = lambda_crit - lambda_min_safe >= 0``
     Currently a stub; requires Group H CoupledFEResultantDriver + J.3.
J.5  Both are logged per-iteration and persisted in the history dicts.

Group L — Outer-inner Orientation Loop
----------------------------------------
L.4  ``run_with_orientation`` enumerates ``OrientationMix`` combos (outer) and runs
     the inner SLSQP over thicknesses (inner) for each combo.  Parallel via
     ``ProcessPoolExecutor`` across combos.
L.8  The optimal mix per role and enumeration cost table are stored in
     ``OptimisationResult.orientation_result`` and persisted to ``optimizer_iterations.npz``.
L.9  Spanwise monotone thickness: linear SLSQP inequalities
     ``c_mono[role,i] = t_role[i] - t_role[i+1] >= 0`` added to ``run()``.
     ``enforce_spanwise_monotone`` from ``DesignProblem`` controls this; default True.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .aggregation import ks_aggregate
from .evaluator import DesignEvaluator
from ..core.types import DesignProblem, DesignVector, OptimisationResult
from blade_precompute._utils.run_logging import RunLogger

# Numerical floor for log objective when maximizing stiffness/mass.
_EPS_LOG = 1e-30


def _build_monotone_constraints(dv0: DesignVector) -> list[dict]:
    """Build spanwise monotone thickness SLSQP inequalities (L.9).

    For each role (skin, cap, web) and adjacent station pair (i, i+1):
        c_mono[role, i] = t_role[i] - t_role[i+1] >= 0
    """
    n = int(dv0.t_skin.shape[0])
    if n < 2:
        return []

    x_dummy = dv0.to_flat()
    n_x = len(x_dummy)
    x_dummy_dv = DesignVector.from_flat(x_dummy, n)

    # Layout: DesignVector.to_flat() packs [t_skin[0..n-1], t_cap[0..n-1], t_web[0..n-1]]
    # Determine offset by matching the known-zero approach.
    try:
        x_test = dv0.to_flat()
        offsets = {}
        for role, arr in [("skin", dv0.t_skin), ("cap", dv0.t_cap), ("web", dv0.t_web)]:
            for j in range(n):
                val = float(arr[j])
                for k in range(len(x_test)):
                    if abs(float(x_test[k]) - val) < 1e-15:
                        if role not in offsets:
                            offsets[role] = k - j
                            break
    except Exception:
        offsets = {"skin": 0, "cap": n, "web": 2 * n}

    skin_off = offsets.get("skin", 0)
    cap_off = offsets.get("cap", n)
    web_off = offsets.get("web", 2 * n)

    cons = []
    for role, off in [("skin", skin_off), ("cap", cap_off), ("web", web_off)]:
        for i in range(n - 1):
            idx_i = off + i
            idx_j = off + i + 1

            def _c_mono(x: np.ndarray, _i: int = idx_i, _j: int = idx_j) -> float:
                return float(x[_i]) - float(x[_j])

            cons.append({"type": "ineq", "fun": _c_mono})

    return cons


def _sample_x0_in_bounds(
    bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random start inside box ``bounds`` (one scalar per design variable)."""
    b = list(bounds)
    return np.array(
        [float(rng.uniform(float(lo), float(hi))) for lo, hi in b],
        dtype=np.float64,
    )


class BladeOptimizer:
    def __init__(
        self,
        problem: DesignProblem,
        method: str = "SLSQP",
        options: dict[str, Any] | None = None,
        *,
        evaluator: DesignEvaluator | None = None,
        run_log: RunLogger | None = None,
        orientation_mix_id: str | None = None,
        progress: Any | None = None,
        on_after_evaluation: Callable[[Any, DesignVector, int], None] | None = None,
    ):
        self.problem = problem
        self.method = method
        self.options = options if options is not None else {"maxiter": 80, "ftol": 1e-6, "disp": False}
        self.evaluator = evaluator if evaluator is not None else DesignEvaluator(problem)
        self._last_x: np.ndarray | None = None
        self._last_ev: Any = None
        self._executed: bool = False
        self._result: OptimisationResult | None = None
        self._run_log = run_log
        self._progress = progress
        self._t_opt_start: float | None = None
        self._dv_best_so_far: DesignVector | None = None
        self._iter_prev_objective: float | None = None
        self._iteration_axis_meta_logged: bool = False
        self._orientation_mix_id = orientation_mix_id
        self._on_after_evaluation = on_after_evaluation

    def _emit_optimizer_iteration_progress(
        self,
        *,
        iteration: int,
        elapsed_opt_s: float,
        ev: Any,
    ) -> None:
        """Emit a concise per-iteration optimiser progress milestone.

        Terminal visibility is intentional here so long SLSQP phases show
        subprocess liveness without requiring users to tail JSONL artifacts.
        """
        if self._progress is None or not getattr(self._progress, "enabled", True):
            return
        payload = {
            "iteration": int(iteration),
            "elapsed_since_optimizer_start_s": round(float(elapsed_opt_s), 3),
            "mass_kg": float(ev.mass),
            "max_fi_hashin": float(ev.max_fi_hashin),
            "stiffness_metric": float(ev.stiffness_metric),
        }
        emit = getattr(self._progress, "event", None)
        if callable(emit):
            emit("optimizer_slsqp_iteration", **payload)
            return
        # Backward-compat fallback for thin mock progress objects.
        emit_jsonl = getattr(self._progress, "event_jsonl_only", None)
        if callable(emit_jsonl):
            emit_jsonl("optimizer_slsqp_iteration", **payload)

    def _ev(self, x: np.ndarray, n_station: int):
        if self._last_x is not None and np.allclose(x, self._last_x, rtol=1e-14, atol=1e-14):
            return self._last_ev
        dv = DesignVector.from_flat(x, n_station)
        self._last_ev = self.evaluator.evaluate(dv)
        self._last_x = x.copy()
        return self._last_ev

    def run(self, dv0: DesignVector) -> OptimisationResult:
        n_station = int(dv0.t_skin.shape[0])
        x0 = dv0.to_flat()
        bounds = dv0.get_bounds()
        rho = float(self.problem.ks_rho)
        rho_buck = float(self.problem.ks_rho_buckling)
        method_use = str(
            getattr(self.problem, "optimizer_method", None) or self.method
        )
        n_restarts = max(0, int(getattr(self.problem, "optimizer_n_restarts", 0)))
        seed_ms = getattr(self.problem, "optimizer_multistart_seed", None)
        rng = np.random.default_rng(seed_ms)
        merged_options = dict(self.options)
        merged_options["ftol"] = float(
            getattr(
                self.problem,
                "optimizer_ftol",
                merged_options.get("ftol", 1e-5),
            )
        )

        obj_mode = str(self.problem.objective)

        def objective(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            if obj_mode == "max_specific_stiffness":
                return math.log(ev.mass + _EPS_LOG) - math.log(ev.stiffness_metric + _EPS_LOG)
            return float(ev.mass)

        def c_ks_hashin(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            return 1.0 - ks_aggregate(ev.fi_hashin, rho)

        def c_ks_vm(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            return 1.0 - ks_aggregate(ev.fi_vm, rho)

        def c_ks_mitc4(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            if ev.fi_mitc4 is None or ev.fi_mitc4.size == 0:
                return 1.0
            return 1.0 - ks_aggregate(ev.fi_mitc4, rho)

        stress_recovery = str(getattr(self.problem, "stress_recovery", "mitc4"))
        ineq_ks_ids: list[str] = []
        cons: list[dict] = []
        # MITC4 shell stress-index KS and Hashin KS (MITC4 N,M + CLPT with K7 fallback); see DesignProblem.stress_recovery.
        cons.append({"type": "ineq", "fun": c_ks_mitc4})
        cons.append({"type": "ineq", "fun": c_ks_hashin})
        ineq_ks_ids.extend(["ks_mitc4", "ks_hashin_laminate_clpt"])
        cons.append({"type": "ineq", "fun": c_ks_vm})
        ineq_ks_ids.append("ks_vm")

        # J.2 — panel buckling KS constraint
        if self.problem.enable_panel_buckling:
            def c_ks_panel_buckling(x: np.ndarray) -> float:
                ev = self._ev(x, n_station)
                if ev.fi_panel_buckling is None or ev.fi_panel_buckling.size == 0:
                    return 1.0  # No composite panels — constraint trivially satisfied
                return 1.0 - ks_aggregate(ev.fi_panel_buckling, rho_buck)

            cons.append({"type": "ineq", "fun": c_ks_panel_buckling})
            ineq_ks_ids.append("ks_panel_buckling")

        # J.4 — global buckling constraint (stub: active only when enable_global_buckling and driver provides lambdas)
        if self.problem.enable_global_buckling:
            lam_safe = float(self.problem.global_buckling_lambda_min)

            def c_global_buckling(x: np.ndarray) -> float:
                ev = self._ev(x, n_station)
                if ev.global_buckling_lambdas is None:
                    return 1.0  # Not computed — do not constrain
                lambda_crit = float(np.min(ev.global_buckling_lambdas))
                return lambda_crit - lam_safe

            cons.append({"type": "ineq", "fun": c_global_buckling})
            ineq_ks_ids.append("global_buckling_lambda")

        # L.9 — spanwise monotone thickness constraints
        mono_n = 0
        if self.problem.enforce_spanwise_monotone and n_station >= 2:
            mono_n = 3 * (n_station - 1)
            mono_cons = _build_monotone_constraints(dv0)
            cons.extend(mono_cons)

        if self._run_log is not None:
            _hsrc = "mitc4_shell_nm_clpt_ply"
            _vsrc = "mitc4_shell_nm_clpt_vm"
            self._run_log.info_event(
                "optimizer.setup",
                scipy_method=method_use,
                optimizer_ftol=float(merged_options.get("ftol", 1e-5)),
                maxiter=merged_options.get("maxiter"),
                n_restarts_planned=n_restarts,
                multistart_seed=seed_ms,
                slsqp_scalar_ineq_ks_ids=ineq_ks_ids,
                slsqp_spanwise_monotone_ineq_count=int(mono_n),
                stress_recovery_mode=stress_recovery,
                hashin_constraint_fi_source=_hsrc,
                von_mises_constraint_fi_source=_vsrc,
            )

        def _execute_minimize(
            x_start: np.ndarray,
        ) -> tuple[Any, list[Any], int, str]:
            from blade_precompute.section_optimisation.engine.iteration_report import (
                beam_nr_residual_tail_array,
                build_optimizer_iteration_payload,
                collect_k7_stack_for_npz,
                collect_section_clpt_for_npz,
                objective_scalar,
                write_iteration_npz,
            )

            current_hist: list[Any] = []

            def _cb(xk: np.ndarray) -> None:
                ev = self._ev(xk, n_station)
                current_hist.append(ev)
                prev_ev = current_hist[-2] if len(current_hist) >= 2 else None
                self._dv_best_so_far = DesignVector.from_flat(
                    np.asarray(xk, dtype=np.float64).copy(), n_station
                )
                ih = int(np.argmax(ev.fi_hashin))
                sh = ev.fi_hashin.shape
                si, ci, pi = np.unravel_index(ih, sh)
                ratio = ev.stiffness_metric / max(ev.mass, _EPS_LOG)
                ks_h = ks_aggregate(ev.fi_hashin, rho)
                has_vm = ev.fi_vm.size > 0
                ks_vm = ks_aggregate(ev.fi_vm, rho) if has_vm else None
                ks_pb = (
                    ks_aggregate(ev.fi_panel_buckling, rho_buck)
                    if ev.fi_panel_buckling is not None
                    else float("nan")
                )
                lambda_crit = (
                    float(np.min(ev.global_buckling_lambdas))
                    if ev.global_buckling_lambdas is not None
                    else float("nan")
                )
                vm_str = f"{ks_vm:.4f}" if ks_vm is not None else "n/a"
                _it = int(len(current_hist))
                _elapsed_opt = 0.0
                if self._t_opt_start is not None:
                    _elapsed_opt = float(time.perf_counter() - self._t_opt_start)
                print(
                    f"  iter={_it} (+{_elapsed_opt:.1f}s opt) mass={ev.mass:.4f} kg  "
                    f"S_int={ev.stiffness_metric:.4g}  "
                    f"S/m={ratio:.4g}  max_Hashin={ev.max_fi_hashin:.4f}  (st={si}, sub={ci}, ply={pi})  "
                    f"max_VM={ev.max_fi_vm:.4f}  "
                    f"KS_Hashin={ks_h:.4f}  KS_VM={vm_str}  "
                    f"KS_PanelBuck={ks_pb:.3f}  lambda_crit={lambda_crit:.3f}"
                )
                self._emit_optimizer_iteration_progress(
                    iteration=_it,
                    elapsed_opt_s=_elapsed_opt,
                    ev=ev,
                )
                if self._on_after_evaluation is not None:
                    dv_cb = DesignVector.from_flat(
                        np.asarray(xk, dtype=np.float64).copy(), n_station
                    )
                    self._on_after_evaluation(ev, dv_cb, int(len(current_hist)))
                if self._run_log is not None:
                    ref_r = self.evaluator._caches[0].result
                    comp_names = list(ref_r.composite_subcomp_names) if ref_r is not None else None
                    iso_names = list(ref_r.isotropic_subcomp_names) if ref_r is not None else None
                    z_m = (
                        np.asarray(self.problem.blade_geometry.z_stations, dtype=np.float64)
                        if self.problem.blade_geometry is not None
                        else None
                    )
                    hotspot_k = int(getattr(self.problem, "iteration_hotspot_k", 10))
                    want_k7_json = bool(getattr(self.problem, "iteration_log_k7_spanwise", False))
                    want_k7_npz = bool(getattr(self.problem, "iteration_dump_npz", False))
                    k7_stack = (
                        collect_k7_stack_for_npz(self.evaluator)
                        if (want_k7_json or want_k7_npz)
                        else None
                    )
                    extra = build_optimizer_iteration_payload(
                        ev,
                        self.problem,
                        iteration=int(len(current_hist)),
                        prev_objective=self._iter_prev_objective,
                        prev_ev=prev_ev,
                        hotspot_k=hotspot_k,
                        composite_names=comp_names,
                        isotropic_names=iso_names,
                        z_stations_m=z_m,
                        axis_meta_emitted=self._iteration_axis_meta_logged,
                        k7_stack=k7_stack,
                        orientation_mix_id=self._orientation_mix_id,
                    )
                    if "composite_subcomp_names" in extra:
                        self._iteration_axis_meta_logged = True
                    self._iter_prev_objective = float(
                        extra.get("objective_value", objective_scalar(ev, obj_mode))
                    )

                    if bool(getattr(self.problem, "iteration_dump_npz", False)):
                        npz_path = (
                            self._run_log.package_output_dir
                            / "arrays"
                            / f"iter_{len(current_hist):04d}.npz"
                        )
                        strip = collect_section_clpt_for_npz(self.evaluator)
                        m0 = getattr(self.evaluator, "_mitc4_clt_station0", {}) or {}
                        tail_k = int(
                            getattr(self.problem, "iteration_beam_nr_residual_tail_k", 8)
                        )
                        nr_tail = beam_nr_residual_tail_array(ev, k=tail_k)
                        write_iteration_npz(
                            npz_path,
                            ev=ev,
                            strip=strip if strip else None,
                            mitc4_station0_panel_abd=m0.get("mitc4_panel_abd"),
                            mitc4_station0_panel_thickness_m=m0.get("mitc4_panel_thickness_m"),
                            mitc4_station0_panel_G_eff=m0.get("mitc4_panel_G_eff"),
                            mitc4_station0_panel_labels=m0.get("mitc4_panel_labels"),
                            k7_stack=k7_stack,
                            beam_nr_residual_tail=nr_tail,
                        )
                        self._run_log.log_artefact(
                            npz_path, "optimizer_iteration_npz", iteration=int(len(current_hist))
                        )

                    self._run_log.info_event("optimizer.iteration", **extra)

            self._last_x = None
            self._last_ev = None
            self._iter_prev_objective = None
            self._iteration_axis_meta_logged = False
            self._t_opt_start = time.perf_counter()
            res = minimize(
                objective,
                x_start,
                method=method_use,
                bounds=bounds,
                constraints=cons,
                options=merged_options,
                callback=_cb,
            )
            n_it = int(res.nit) if hasattr(res, "nit") and res.nit is not None else len(
                current_hist
            )
            return res, current_hist, n_it, str(res.message)

        best_res: Any = None
        best_hist: list[Any] = []
        best_f = float("inf")
        best_attempt = 0
        best_nit = 0
        best_msg = ""
        for att in range(n_restarts + 1):
            x_s = x0 if att == 0 else _sample_x0_in_bounds(bounds, rng)
            res, hist_att, n_it, msg = _execute_minimize(x_s)
            fv = float(res.fun)
            if best_res is None or fv < best_f:
                best_f = fv
                best_res = res
                best_hist = hist_att
                best_attempt = att
                best_nit = n_it
                best_msg = msg
        res = best_res
        if res is None:  # pragma: no cover
            raise RuntimeError("BladeOptimizer multistart produced no result")
        msg = best_msg
        if n_restarts > 0:
            msg = (
                f"{msg}  [multistart: n_restarts={n_restarts}, best_attempt={best_attempt} "
                f"by objective value]"
            )
        self._dv_best_so_far = DesignVector.from_flat(res.x, n_station)
        dv_opt = self._dv_best_so_far
        result = OptimisationResult(
            success=bool(res.success),
            message=msg,
            dv_opt=dv_opt,
            evaluations=best_hist,
            n_iter=best_nit,
            dv_best_so_far=self._dv_best_so_far,
        )
        self._result = result
        self._executed = True
        return result

    def run_with_orientation(
        self,
        dv0: DesignVector,
        *,
        n_workers_outer: int = 1,
    ) -> OptimisationResult:
        """Outer-inner optimisation with discrete orientation enumeration (L.4).

        Outer loop: enumerate all feasible ``OrientationMix`` combinations per role
        (from ``DesignProblem.orientation_bounds``).  Each combo runs the inner
        SLSQP (``run()``) over continuous thicknesses with the orientation baked in.

        The globally best (combo, inner-result) pair is returned as the final
        ``OptimisationResult``.  An enumeration cost table (n combos tried,
        best cost per combo) is appended to ``OptimisationResult.orientation_result``
        for retrospective analysis (L.8).

        Parameters
        ----------
        dv0
            Initial design vector (orientation will be overridden by each combo).
        n_workers_outer
            ``ProcessPoolExecutor`` workers for outer enumeration (1 = serial).
        """
        from .orientation_mix import enumerate_feasible_mixes, OrientationMix
        from ..core.types import OrientationBounds

        p = self.problem
        ob = p.orientation_bounds
        if ob is None:
            # No orientation bounds configured — fall back to standard run
            return self.run(dv0)

        combos: list[dict[str, OrientationMix]] = []
        roles_with_bounds = [r for r in ("skin", "cap", "web") if r in ob]
        if not roles_with_bounds:
            return self.run(dv0)

        # Build Cartesian product of feasible mixes per role
        from itertools import product as iproduct

        per_role: dict[str, list[OrientationMix]] = {}
        for role in roles_with_bounds:
            bounds_r: OrientationBounds = ob[role]
            per_role[role] = list(
                enumerate_feasible_mixes(
                    role,
                    n_half_min=bounds_r.n_half_min,
                    n_half_max=bounds_r.n_half_max,
                    n_biax_min=bounds_r.n_biax_min,
                    n_0_min=bounds_r.n_0_min,
                    n_90_min=bounds_r.n_90_min,
                )
            )

        role_keys = list(per_role.keys())
        role_lists = [per_role[r] for r in role_keys]

        for combo_tuple in iproduct(*role_lists):
            combo = {role_keys[i]: combo_tuple[i] for i in range(len(role_keys))}
            combos.append(combo)

        if not combos:
            return self.run(dv0)

        best_cost: float = float("inf")
        best_result: OptimisationResult | None = None
        enumeration_table: list[dict[str, Any]] = []

        def _run_combo(combo: dict[str, OrientationMix]) -> tuple[float, OptimisationResult, dict[str, Any]]:
            mix_rep = {r: m.as_dict() for r, m in combo.items()}
            mix_id = json.dumps(mix_rep, sort_keys=True)
            evaluator_combo = DesignEvaluator(p, run_log=self._run_log)
            evaluator_combo._abd_cache = self.evaluator._abd_cache  # share ABD cache (L.5)
            opt_combo = BladeOptimizer(
                p,
                method=str(p.optimizer_method),
                options=self.options,
                evaluator=evaluator_combo,
                run_log=self._run_log,
                orientation_mix_id=mix_id,
                progress=self._progress,
                on_after_evaluation=self._on_after_evaluation,
            )
            result_c = opt_combo.run(dv0)
            final_ev = result_c.evaluations[-1] if result_c.evaluations else None
            cost = float(final_ev.mass) if final_ev is not None else float("inf")
            return cost, result_c, mix_rep

        if n_workers_outer <= 1:
            for combo_i, combo in enumerate(combos):
                cost_c, result_c, mix_rep = _run_combo(combo)
                enumeration_table.append({"mix": mix_rep, "cost": cost_c, "success": result_c.success})
                if self._run_log is not None:
                    fe = result_c.evaluations[-1] if result_c.evaluations else None
                    self._run_log.info_event(
                        "optimizer.orientation_combo",
                        mix=mix_rep,
                        combo_index=int(combo_i),
                        n_combos_total=len(combos),
                        n_inner_iters=int(result_c.n_iter),
                        inner_success=bool(result_c.success),
                        final_mass_kg=float(fe.mass) if fe is not None else None,
                        message=str(result_c.message),
                    )
                if cost_c < best_cost:
                    best_cost = cost_c
                    best_result = result_c
        else:
            # Parallel outer enumeration (ProcessPoolExecutor)
            with ProcessPoolExecutor(max_workers=n_workers_outer) as pool:
                futures = [pool.submit(_run_combo, c) for c in combos]
                for combo_i, (combo, fut) in enumerate(zip(combos, futures)):
                    try:
                        cost_c, result_c, mix_rep = fut.result()
                    except Exception as exc:
                        cost_c = float("inf")
                        result_c = OptimisationResult(
                            success=False,
                            message=str(exc),
                            dv_opt=dv0,
                            evaluations=[],
                            n_iter=0,
                        )
                        mix_rep = {r: m.as_dict() for r, m in combo.items()}
                    enumeration_table.append({"mix": mix_rep, "cost": cost_c, "success": result_c.success})
                    if self._run_log is not None:
                        fe = result_c.evaluations[-1] if result_c.evaluations else None
                        self._run_log.info_event(
                            "optimizer.orientation_combo",
                            mix=mix_rep,
                            combo_index=int(combo_i),
                            n_combos_total=len(combos),
                            n_inner_iters=int(result_c.n_iter),
                            inner_success=bool(result_c.success),
                            final_mass_kg=float(fe.mass) if fe is not None else None,
                            message=str(result_c.message),
                        )
                    if cost_c < best_cost:
                        best_cost = cost_c
                        best_result = result_c

        if best_result is None:
            best_result = OptimisationResult(
                success=False,
                message="No feasible orientation combo found.",
                dv_opt=dv0,
                evaluations=[],
                n_iter=0,
            )

        # L.8: attach enumeration cost table for retrospective analysis
        best_result_with_orient = OptimisationResult(
            success=best_result.success,
            message=best_result.message,
            dv_opt=best_result.dv_opt,
            evaluations=best_result.evaluations,
            n_iter=best_result.n_iter,
            orientation_result={  # type: ignore[call-arg]
                "n_combos_tried": len(combos),
                "best_cost": float(best_cost),
                "enumeration_table": enumeration_table,
            } if hasattr(OptimisationResult, "orientation_result") else None,
        )
        self._result = best_result_with_orient
        self._executed = True
        return best_result_with_orient

    def execute(self, dv0: DesignVector) -> "BladeOptimizer":
        """Orchestrator-style alias for compatibility with API conventions."""
        self.run(dv0)
        return self

    def get_results(self) -> OptimisationResult:
        """Return the most recent optimisation result from `run()` / `execute()`."""
        if not self._executed or self._result is None:
            raise RuntimeError("BladeOptimizer.execute() or BladeOptimizer.run() must be called first.")
        return self._result
