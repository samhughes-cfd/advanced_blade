"""Datatypes for blade sizing optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import MaterialAssignment
from blade_precompute.section_properties.core.types import SectionSolveResult


ThicknessRole = Literal["skin", "cap", "web", "fixed"]

OptimizationObjective = Literal["min_mass", "max_specific_stiffness"]


@dataclass
class DesignVector:
    t_skin: NDArray[np.float64]
    t_cap: NDArray[np.float64]
    t_web: NDArray[np.float64]
    t_skin_bounds: tuple[float, float] = (0.002, 0.050)
    t_cap_bounds: tuple[float, float] = (0.005, 0.100)
    t_web_bounds: tuple[float, float] = (0.003, 0.060)

    def to_flat(self) -> NDArray[np.float64]:
        return np.concatenate([self.t_skin.ravel(), self.t_cap.ravel(), self.t_web.ravel()])

    @staticmethod
    def from_flat(x: NDArray[np.float64], n_station: int) -> DesignVector:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != 3 * n_station:
            raise ValueError(f"Expected {3*n_station} values, got {x.size}.")
        return DesignVector(
            t_skin=x[0:n_station].copy(),
            t_cap=x[n_station : 2 * n_station].copy(),
            t_web=x[2 * n_station : 3 * n_station].copy(),
        )

    def get_bounds(self) -> list[tuple[float, float]]:
        n = self.t_skin.shape[0]
        lo_s, hi_s = self.t_skin_bounds
        lo_c, hi_c = self.t_cap_bounds
        lo_w, hi_w = self.t_web_bounds
        return [(lo_s, hi_s)] * n + [(lo_c, hi_c)] * n + [(lo_w, hi_w)] * n


@dataclass
class OptimBladeGeometry:
    z_stations: NDArray[np.float64]
    r_ref: NDArray[np.float64]
    kappa0: NDArray[np.float64]
    tau0: NDArray[np.float64]
    chord: NDArray[np.float64]
    twist: NDArray[np.float64]  # structural blade twist [deg]; built section orientation, not α or pitch DOF
    airfoil_profiles: list[Any]
    web_positions: NDArray[np.float64]
    subcomponent_materials: dict[str, MaterialAssignment]
    """Maps subcomponent name → base laminate or isotropic (thickness scaled via DV)."""
    thickness_role: dict[str, ThicknessRole] = field(default_factory=dict)
    """If empty, inferred: skin→skin, cap_ps/cap_ss→cap, web→web, else fixed."""
    cap_shear_lag_width: float | None = None
    """Optional cap width ``b_cap`` [m] for Reissner shear lag; else derived from chord."""
    box_height_frac: float = 0.12
    """Normalized section box height (fraction of chord) for default strip layout."""
    subcomponent_polylines_norm: dict[str, NDArray[np.float64]] | None = None
    """Optional normalized ``(y,z)`` polylines per subcomponent name; else built-in box."""


@dataclass
class ExtremeLoads:
    z_stations: NDArray[np.float64]
    N: NDArray[np.float64]
    Vy: NDArray[np.float64]
    Vz: NDArray[np.float64]
    My: NDArray[np.float64]
    Mz: NDArray[np.float64]
    T: NDArray[np.float64]
    B: NDArray[np.float64] | None = None

    def bimoment(self) -> NDArray[np.float64]:
        if self.B is not None:
            return np.asarray(self.B, dtype=np.float64)
        return np.zeros_like(self.N, dtype=np.float64)


@dataclass
class StationCache:
    t_skin: float
    t_cap: float
    t_web: float
    result: SectionSolveResult | None = None
    dirty: bool = True

    def is_stale(self, dv: DesignVector, i: int, tol: float = 1e-9) -> bool:
        return (
            abs(self.t_skin - dv.t_skin[i]) > tol
            or abs(self.t_cap - dv.t_cap[i]) > tol
            or abs(self.t_web - dv.t_web[i]) > tol
        )


@dataclass
class DesignEvaluation:
    """``stiffness_metric``: integrated ``trace(K7)`` along span; constitutive proxy, not compliance."""

    dv: DesignVector
    mass: float
    stiffness_metric: float
    resultants: NDArray[np.float64]
    fi_tw: NDArray[np.float64]
    fi_vm: NDArray[np.float64]
    fi_delam: NDArray[np.float64] | None
    max_fi_tw: float
    max_fi_vm: float
    max_fi_delam: float | None


@dataclass
class OptimisationResult:
    success: bool
    message: str
    dv_opt: DesignVector
    evaluations: list[DesignEvaluation]
    n_iter: int


@dataclass
class DesignProblem:
    """
    ``objective``: ``min_mass`` minimizes blade mass; ``max_specific_stiffness`` maximizes
    stiffness/mass (integrated ``trace(K7)`` / mass) via a log objective in the optimizer.
    """

    blade_geometry: OptimBladeGeometry
    extreme_loads: ExtremeLoads
    solver: Any = None
    objective: OptimizationObjective = "min_mass"
    ks_rho: float = 50.0
    enable_tier3_delam: bool = False
    n_workers: int = 4
    composite_subcomp_idx: NDArray[np.int64] | None = None
    isotropic_subcomp_idx: NDArray[np.int64] | None = None
