"""SLSQP driver with KS failure constraints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .aggregation import ks_aggregate
from .evaluator import DesignEvaluator
from ..core.types import DesignProblem, DesignVector, OptimisationResult

# Numerical floor for log objective when maximizing stiffness/mass.
_EPS_LOG = 1e-30


class BladeOptimizer:
    def __init__(
        self,
        problem: DesignProblem,
        method: str = "SLSQP",
        options: dict[str, Any] | None = None,
    ):
        self.problem = problem
        self.method = method
        self.options = options if options is not None else {"maxiter": 80, "ftol": 1e-6, "disp": False}
        self.evaluator = DesignEvaluator(problem)
        self._last_x: np.ndarray | None = None
        self._last_ev: Any = None

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
        hist: list = []
        rho = float(self.problem.ks_rho)

        def _cb(xk: np.ndarray) -> None:
            ev = self._ev(xk, n_station)
            hist.append(ev)
            itw = int(np.argmax(ev.fi_tw))
            sh = ev.fi_tw.shape
            si, ci, pi = np.unravel_index(itw, sh)
            ratio = ev.stiffness_metric / max(ev.mass, _EPS_LOG)
            print(
                f"  iter mass={ev.mass:.4f} kg  S_int={ev.stiffness_metric:.4g}  S/m={ratio:.4g}  "
                f"max_TW={ev.max_fi_tw:.4f}  (st={si}, sub={ci}, ply={pi})  max_VM={ev.max_fi_vm:.4f}  "
                f"KS_TW={ks_aggregate(ev.fi_tw, rho):.4f}  KS_VM={ks_aggregate(ev.fi_vm, rho):.4f}"
            )

        obj_mode = str(self.problem.objective)

        def objective(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            if obj_mode == "max_specific_stiffness":
                return math.log(ev.mass + _EPS_LOG) - math.log(ev.stiffness_metric + _EPS_LOG)
            return float(ev.mass)

        def c_ks_tw(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            return 1.0 - ks_aggregate(ev.fi_tw, rho)

        def c_ks_vm(x: np.ndarray) -> float:
            ev = self._ev(x, n_station)
            return 1.0 - ks_aggregate(ev.fi_vm, rho)

        cons = [
            {"type": "ineq", "fun": c_ks_tw},
            {"type": "ineq", "fun": c_ks_vm},
        ]
        if self.problem.enable_tier3_delam:

            def c_ks_del(x: np.ndarray) -> float:
                ev = self._ev(x, n_station)
                if ev.fi_delam is None:
                    return 1.0
                return 1.0 - ks_aggregate(ev.fi_delam, rho)

            cons.append({"type": "ineq", "fun": c_ks_del})

        self._last_x = None
        self._last_ev = None
        res = minimize(
            objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=cons,
            options=self.options,
            callback=_cb,
        )
        dv_opt = DesignVector.from_flat(res.x, n_station)
        return OptimisationResult(
            success=bool(res.success),
            message=str(res.message),
            dv_opt=dv_opt,
            evaluations=hist,
            n_iter=int(res.nit) if hasattr(res, "nit") else len(hist),
        )
