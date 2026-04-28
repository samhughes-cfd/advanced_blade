"""Datatypes for blade sizing optimisation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from blade_precompute.global_beam_model.engine.axial_loading import AxialLoadingConfig

StressRecoveryMode = Literal["mitc4"]
BeamDriverMode = Literal["prescribed", "global_beam", "coupled_fe"]


def normalize_stress_recovery(value: str) -> str:
    """Map legacy values to the only supported mode and warn once per process semantics via caller.

    Returns
    -------
    mitc4
    """
    s = (value or "").strip()
    if s in ("strip_clpt", "both"):
        warnings.warn(
            f"stress_recovery {s!r} is removed; using 'mitc4' (see DesignProblem).",
            DeprecationWarning,
            stacklevel=3,
        )
        return "mitc4"
    if s == "mitc4" or s == "":
        return "mitc4"
    raise ValueError(f"Unknown stress_recovery {value!r}; use 'mitc4'.")

BeamSectionStiffnessSource = Literal["section_properties"]

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_geometry.laminate_thickness_limits import (
    MIN_REALISTIC_SKIN_LAMINATE_THICKNESS_M,
    MIN_REALISTIC_SPAR_LAMINATE_THICKNESS_M,
    MIN_REALISTIC_WEB_LAMINATE_THICKNESS_M,
)
from blade_precompute.section_properties.engine.geometry import MaterialAssignment
from blade_precompute.section_properties.core.types import SectionSolveResult


ThicknessRole = Literal["skin", "cap", "web", "fixed"]

OptimisationObjective = Literal["min_mass", "max_specific_stiffness"]


@dataclass(frozen=True)
class OrientationBounds:
    """Integer ply-count bounds per subcomponent role for the outer orientation enumeration.

    ``n_half`` is the half-stack ply count (total stack = 2 * n_half, symmetric about midplane).
    ``n_biax_min`` enforces at least one ±45 pair in the half-stack for shear resistance.
    All minimums default to 0 except ``n_biax_min = 1`` (at least one ±45 pair).
    """

    n_half_min: int = 2
    n_half_max: int = 10
    n_0_min: int = 0
    n_biax_min: int = 1
    n_90_min: int = 0

    def t_bounds_from_n_half(self, t_half_single: float) -> tuple[float, float]:
        """Translate integer N_half bounds into continuous t_role bounds for SLSQP.

        ``t_half_single`` is the half-stack thickness for one repeat unit
        (from ``t_half_for_mix`` in orientation_mix.py).
        """
        return (
            float(self.n_half_min) * 2.0 * t_half_single,
            float(self.n_half_max) * 2.0 * t_half_single,
        )


def objective_from_str(value: str) -> OptimisationObjective:
    key = (value or "").strip().lower().replace("_", "-")
    if key in ("min-mass", "minmass"):
        return "min_mass"
    if key in ("max-specific-stiffness", "max-stiffness-mass", "specific-stiffness"):
        return "max_specific_stiffness"
    raise ValueError(f"Unknown objective {value!r}; use min-mass or max-specific-stiffness.")


@dataclass
class DesignVector:
    t_skin: NDArray[np.float64]
    t_cap: NDArray[np.float64]
    t_web: NDArray[np.float64]
    t_skin_bounds: tuple[float, float] = (MIN_REALISTIC_SKIN_LAMINATE_THICKNESS_M, 0.050)
    t_cap_bounds: tuple[float, float] = (MIN_REALISTIC_SPAR_LAMINATE_THICKNESS_M, 0.100)
    t_web_bounds: tuple[float, float] = (MIN_REALISTIC_WEB_LAMINATE_THICKNESS_M, 0.060)

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
    run_global_beam: bool = True
    """If False, skip Tier-A global beam static solve (section-only iteration)."""
    beam_section_stiffness_source: BeamSectionStiffnessSource = "section_properties"
    """``section_properties`` for :class:`SectionStation` ``K6``/``K7`` (from strip section solver)."""
    resolved_dv: "DesignVector | None" = None
    """When set by :func:`apply_dv_to_bg`, downstream stages use this DV instead of the default
    initial vector (``default_dv0``).  Carries the converged / best-so-far design vector
    into post-optimisation re-renders of section_properties / global_beam_model."""
    radial_r_m: NDArray[np.float64] | None = None
    """Hub-centred radial distance [m] (spanwise table ``radial_pos``); for centrifugal axial `q_x`."""


def apply_dv_to_bg(bg: "OptimBladeGeometry", dv: "DesignVector") -> "OptimBladeGeometry":
    """Return a copy of *bg* carrying *dv* as the canonical resolved design vector.

    All geometry fields (chord, twist, kappa0, airfoil profiles, web positions,
    subcomponent_materials, etc.) are preserved unchanged.  The returned object
    has ``resolved_dv`` set to *dv* so that downstream stages
    (``section_properties_impl``, ``beam_model_impl``) use the converged
    thicknesses instead of the ``default_dv0`` initial vector.

    Neither *bg* nor *dv* are mutated.

    Example
    -------
    ::

        bg_final = apply_dv_to_bg(bg_struct, dv_resolved)
        # pass bg_final as bg_override to SectionPropertiesStage / BeamModelStage
    """
    import copy

    bg_final = copy.copy(bg)
    bg_final.resolved_dv = dv
    return bg_final


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


@dataclass(frozen=True)
class DistributedLoadCurves:
    """Distributed spanwise hydrodynamic / operating loads (same convention as :class:`PrecomputeInputs`).

    Used with ``beam_driver='global_beam'`` to build :class:`BeamLoads` identically
    to :func:`~blade_precompute.orchestration.precompute.stages.beam_model_impl`.
    Interpolate onto beam element midpoints; ``z`` on ``loads_r_z_m`` need not
    match structural section spacing.
    """

    loads_r_z_m: NDArray[np.float64]
    q_y_Npm: NDArray[np.float64]
    q_z_Npm: NDArray[np.float64]
    m_x_Nmpm: NDArray[np.float64]
    q_x_Npm: NDArray[np.float64] | None = None
    """Spanwise (along reference line) line load [N/m] (centrifugal + gravity); added to ``distributed_q[:, span_axis]``."""


@dataclass
class DesignEvaluation:
    """``stiffness_metric``: integrated ``trace(K7)`` along span; constitutive proxy, not compliance.

    Panel and global buckling fields are populated when ``DesignProblem.enable_panel_buckling``
    is True (Group J). ``tip_deflection`` is populated when a coupled FE driver is used (Group H).
    """

    dv: DesignVector
    mass: float
    stiffness_metric: float
    resultants: NDArray[np.float64]
    fi_hashin: NDArray[np.float64]
    fi_vm: NDArray[np.float64]
    max_fi_hashin: float
    max_fi_vm: float
    # --- Group J: buckling ---
    fi_panel_buckling: NDArray[np.float64] | None = None
    """shape (n_stations, n_edges) panel BI per station-edge; None when buckling disabled."""
    global_buckling_lambdas: NDArray[np.float64] | None = None
    """Lowest N_modes global buckling eigenvalues (lambda_crit); None when not computed."""
    # --- Group G/H: tip deflection ---
    tip_deflection: float | None = None
    """Flapwise tip deflection [m] from coupled FE result; None when using prescribed driver."""
    # --- Group H: beam state cache ---
    beam_state: Any | None = None
    """Last ``BeamSolveResult`` (or ``PrescribedResultantBeamState``) for downstream use."""
    # --- Group J beam K7 condition stats ---
    k7_cond_stats: dict[str, float] | None = None
    """K7 condition number stats per evaluation for I.7 debug provenance."""
    # --- In-loop MITC4 shell stress index (with Hashin from N,M via mitc4_shell_fi_batch) ---
    fi_mitc4: NDArray[np.float64] | None = None
    """Per-station stress index from MITC4 section recovery; shape ``(n_stations,)``."""
    max_fi_mitc4: float | None = None
    """``max(fi_mitc4)`` when MITC4 path runs; else None."""


@dataclass
class OptimisationResult:
    success: bool
    message: str
    dv_opt: DesignVector
    evaluations: list[DesignEvaluation]
    n_iter: int
    # L.8: orientation enumeration cost table (populated by BladeOptimizer.run_with_orientation)
    orientation_result: dict[str, Any] | None = None
    # Last accepted iterate from SLSQP callback; populated regardless of convergence status.
    # Use dv_opt if success=True, dv_best_so_far otherwise, for post-optimisation re-renders.
    dv_best_so_far: "DesignVector | None" = None


@dataclass
class DesignProblem:
    """
    ``objective``: ``min_mass`` minimizes blade mass; ``max_specific_stiffness`` maximizes
    stiffness/mass (integrated ``trace(K7)`` / mass) via a log objective in the optimizer.

    Warping BC convention (I.3): warping DOF ``psi`` is clamped at z=0 (root), free at z=L (tip).
    No end bimoment load applied at z=L.  Shell K77 homogenisation must use the same clamped
    warping BC on panel end nodes when computing the 7th K7 column (see homogenisation.py I.6).

    Pre-loop vs in-loop K7 (I.11): the pre-loop ``BeamModelStage`` uses strip K7 from
    ``SectionPropertiesStage`` (informational only). The in-loop coupled FE driver computes
    updated K7 from shell homogenisation per SLSQP evaluation.  Both K7 stacks are persisted
    in summary.json under ``pre_loop_k7_stack`` / ``in_loop_k7_stack`` for provenance comparison.

    Single-LC assumption (K.1): the optimiser uses one extreme-load ``.dat`` file. Multi-LC
    envelope is out of scope. The SLS tip-deflection bound (``sls_tip_frac``) stabilises the
    hydrodynamic loading envelope, justifying the assumption of invariant hydrodynamics.
    """

    blade_geometry: OptimBladeGeometry
    extreme_loads: ExtremeLoads
    solver: Any = None
    objective: OptimisationObjective = "min_mass"
    ks_rho: float = 35.0
    n_workers: int = 4
    composite_subcomp_idx: NDArray[np.int64] | None = None
    isotropic_subcomp_idx: NDArray[np.int64] | None = None
    # --- Group G: SLS tip deflection ---
    sls_tip_frac: float | None = None
    """Flapwise tip deflection limit as fraction of blade length (e.g. 0.10 = 10%)."""
    # --- Group H: beam driver ---
    beam_driver: str = "prescribed"
    """``'prescribed'`` (Tier-B: tabulated ``ExtremeLoads`` resultants).
    ``'global_beam'`` / ``'coupled_fe'``: same Tier-A global beam as :func:`beam_model_impl`
    with :class:`DistributedLoadCurves` (equilibrated resultants from ``solve_static``).
    """
    distributed_loads: DistributedLoadCurves | None = None
    """Required when ``beam_driver`` is ``global_beam`` or ``coupled_fe``."""
    axial_loading: "AxialLoadingConfig | None" = None
    """When set with ``global_beam``/``coupled_fe``, in-loop :class:`GlobalBeamResultantDriver` uses it for ``q_x``."""
    n_beam_nodes: int = 50
    """Global beam FE nodes; used only for ``global_beam`` driver."""
    stress_recovery: StressRecoveryMode = "mitc4"
    """MITC4 ``ShellPanelResultants`` (N, M) drive ``clpt_ply_failure_indices`` into mapped
    ``fi_hashin`` rows and mapped isotropic ``fi_vm`` (von Mises vs ``sigma_allow``); unmapped
    rows fall back to K7/CLPT strip recovery. Per-station ``fi_mitc4`` is an auxiliary shell scalar."""
    optimizer_method: str = "SLSQP"
    """``scipy.optimize.minimize`` method: ``SLSQP`` (default) or ``trust-constr``."""
    optimizer_ftol: float = 1e-5
    """Function tolerance for SciPy optimiser options (``ftol`` for SLSQP, same key for trust-constr)."""
    optimizer_n_restarts: int = 0
    """Number of extra multistarts from random points in design bounds (0 = only ``dv0``)."""
    optimizer_multistart_seed: int | None = None
    """RNG seed for multistart sampling; ``None`` uses non-reproducible draws."""
    mitc4_n_elements_per_panel: int = 10
    """MITC4 elements per panel along contour for ``run_section_with_mitc4_shell``."""
    # --- Group I.3: warping BC convention (documented above) ---
    # --- Group I.7: K7 condition monitoring ---
    k7_cond_warn_threshold: float = 1e10
    """Log a RunLogger warning when cond(K7) exceeds this value at any station."""
    # --- Fix 1: stress projection diagnostics ---
    debug_stress_projection: bool = False
    """When True, log resultants / strains / comp_res / max sigma on iteration 0.
    Enable via ``main_precompute.py`` for a single evaluation to verify units."""
    # --- Group J: panel + global buckling ---
    enable_panel_buckling: bool = False
    """When True, compute orthotropic panel buckling BI per station-edge (J.1-J.2)."""
    ks_rho_buckling: float = 25.0
    """KS aggregation parameter for panel buckling constraint (sharper than strength)."""
    enable_global_buckling: bool = False
    """When True, solve (K_t - lambda K_g) phi = 0 at each coupled FE evaluation (J.3-J.4)."""
    global_buckling_lambda_min: float = 1.5
    """Global buckling safety factor: c_global_buckling = lambda_crit - lambda_min_safe >= 0."""
    n_global_buckling_modes: int = 5
    """Number of lowest global buckling eigenvalues to extract (J.3)."""
    # --- Group L: orientation design variable ---
    orientation_bounds: "dict[ThicknessRole, OrientationBounds] | None" = None
    """Per-role integer ply-count bounds for outer orientation enumeration (L.3-L.6).
    When None, orientation is fixed from the YAML template (existing behaviour)."""
    # --- Group L.9: spanwise monotone thickness ---
    enforce_spanwise_monotone: bool = True
    """Enforce t_role[i] >= t_role[i+1] (ply-drop only towards tip) via linear SLSQP constraints."""
    # --- Iteration diagnostics (orchestration / RunLogger) ---
    iteration_dump_npz: bool = False
    """When True, write ``section_optimisation/arrays/iter_XXXX.npz`` each SLSQP callback."""
    iteration_hotspot_k: int = 10
    """Number of largest Hashin FI entries to list in ``optimizer.iteration`` JSON."""
    iteration_emit_schema: bool = True
    """When True, write ``section_optimisation/iteration_payload_schema.json`` once per optimisation run."""
    iteration_delta_thickness_tol_m: float = 1e-9
    """Station counts as changed when ``abs(Δt)`` exceeds this (per-role design deltas in ``optimizer.iteration``)."""
    iteration_log_constraint_deltas: bool = True
    """When True and ``prev_ev`` is present, log ``constraint_deltas_max_abs_ks_slack`` vs previous iterate."""
    iteration_log_beam_summary: bool = True
    """When True and ``beam_state`` is a :class:`~blade_precompute.global_beam_model.core.types.BeamSolveResult`,
    log scalar beam convergence / norms (cheap)."""
    iteration_log_beam_nr_history: bool = False
    """When True, append truncated Newton–Raphson ``iteration_history`` to JSON (bounded: first + last 3)."""
    iteration_log_k7_spanwise: bool = False
    """When True and a ``K7_stack`` is supplied to the payload builder, log per-diagonal min/median/max across span."""
    iteration_beam_nr_residual_tail_k: int = 8
    """When ``iteration_dump_npz`` is True, store up to this many trailing NR ``residual_norm`` scalars in the NPZ."""

    def __post_init__(self) -> None:
        self.stress_recovery = normalize_stress_recovery(str(self.stress_recovery))  # type: ignore[assignment]
