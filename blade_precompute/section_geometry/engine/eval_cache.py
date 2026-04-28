"""Station-local cache for grid-evaluated component SDF fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import os

import numpy as np
from .csg_ir import compile_expr


def _eikonal_redistance_2d(
    phi: np.ndarray, dx: float, dy: float, *, n_iter: int = 12, dt: float = 0.15
) -> np.ndarray:
    """
    Few iterations of a first-order reinitialization ``φ_t + s₀(|∇φ| - 1) = 0``
    (Sussman et al. style) to bring |∇φ| → 1 without external dependencies.
    """
    f = np.asarray(phi, dtype=np.float64, copy=True)
    s0 = np.sign(np.clip(f, a_min=-1e-15, a_max=1e-15))
    s0 = np.where(s0 == 0.0, 1.0, s0)
    for _ in range(int(n_iter)):
        gy = np.gradient(f, float(dy), axis=0)
        gx = np.gradient(f, float(dx), axis=1)
        gmag = np.sqrt(gx * gx + gy * gy) + 1e-20
        f = f - float(dt) * s0 * (gmag - 1.0)
    return f


@dataclass
class SectionEvalCache:
    """Memoize ``grid.eval`` outputs by ``(grid_id, label)``."""

    _phi_by_key: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    _expr_cache_by_grid: dict[int, dict[Any, np.ndarray]] = field(default_factory=dict)

    def get(self, grid: Any, label: str) -> np.ndarray | None:
        return self._phi_by_key.get((id(grid), str(label)))

    def set(self, grid: Any, label: str, phi: np.ndarray) -> np.ndarray:
        arr = np.asarray(phi, dtype=float)
        self._phi_by_key[(id(grid), str(label))] = arr
        return arr

    def get_or_eval(self, label: str, sdf_callable: Any, grid: Any) -> np.ndarray:
        return self.get_or_eval_with_owner(label, sdf_callable, grid, owner=None)

    def get_or_eval_with_owner(
        self, label: str, sdf_callable: Any, grid: Any, owner: Any | None
    ) -> np.ndarray:
        key = (id(grid), str(label))
        cached = self._phi_by_key.get(key)
        if cached is not None:
            return cached

        phi = None
        use_ir = os.getenv("SECTION_GEOMETRY_USE_CSG_IR", "0").lower() in {"1", "true", "yes"}
        if use_ir:
            expr = None
            host = getattr(owner, "_mcs", owner) if owner is not None else None
            if owner is not None:
                if str(label) in {"_skin_outer_boundary", "_skin_inner_boundary"}:
                    if str(label) == "_skin_outer_boundary":
                        expr = getattr(host, "_skin_outer_boundary_expr", None)
                    else:
                        expr = getattr(host, "_skin_inner_boundary_expr", None)
                else:
                    comp = getattr(host, "_components", {}).get(str(label))
                    expr = getattr(comp, "_expr", None) if comp is not None else None
            else:
                expr = getattr(sdf_callable, "_expr", None)
            if expr is not None:
                gid = id(grid)
                expr_cache = self._expr_cache_by_grid.setdefault(gid, {})
                phi = np.asarray(compile_expr(expr, grid, cache=expr_cache), dtype=float)

        if owner is not None:
            twist = float(getattr(owner, "_twist", 0.0))
            if abs(twist) > 1e-10 and hasattr(grid, "rotated_eval"):
                base_components = getattr(owner, "_components_unrotated", None)
                if isinstance(base_components, dict) and label in base_components:
                    phi = np.asarray(grid.rotated_eval(base_components[label], twist), dtype=float)
                elif str(label) == "_skin_outer_boundary":
                    base_sdf = getattr(owner, "_skin_outer_boundary_unrotated_sdf", None)
                    if base_sdf is not None:
                        phi = np.asarray(grid.rotated_eval(base_sdf, twist), dtype=float)
                elif str(label) == "_skin_inner_boundary":
                    base_sdf = getattr(owner, "_skin_inner_boundary_unrotated_sdf", None)
                    if base_sdf is not None:
                        phi = np.asarray(grid.rotated_eval(base_sdf, twist), dtype=float)
        if phi is None:
            phi = np.asarray(grid.eval(sdf_callable), dtype=float)
        reflag = os.getenv("SECTION_GEOMETRY_REDISTANCE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if reflag and phi.ndim == 2 and hasattr(grid, "dx") and hasattr(grid, "dy"):
            try:
                phi = _eikonal_redistance_2d(
                    phi, float(grid.dx), float(grid.dy)
                )
            except Exception:
                pass
        self._phi_by_key[key] = phi
        return phi

    def clear(self) -> None:
        self._phi_by_key.clear()
        self._expr_cache_by_grid.clear()

