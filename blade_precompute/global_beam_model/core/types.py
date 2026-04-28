"""
global_beam_model/core/types.py
================================
Geometry, spanwise section data (``K6`` / ``K7``), loads, solver I/O.

**Level 2 (out of scope):** section shape distortion under large torsion
(typically >~15°) is not modelled; use shell-based section analysis.

Strain six-vector ``e_sec`` follows ``section_model`` / ``SectionProps``::

    e_sec = [ε₀, κ_y, κ_z, γ_t, γ_s_y, γ_s_z]

Beam **resultants** (seven-vector) use::

    [N, Vy, Vz, My, Mz, T, B]

Spanwise naming convention:

- ``z`` denotes the blade span station coordinate used by solvers and outputs.
- ``s`` denotes interpolation abscissa vectors in tabulated stiffness arrays.

Use :func:`global_beam_model.engine.constitutive.resultants_to_recovery6` to map the first six
entries to ``section_model`` order ``[N, My, Mz, T, Vy, Vz]`` for downstream section
recovery tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class NodeState:
    """
    Deformed configuration at one node.

    ``q`` is unit quaternion ``[w,x,y,z]`` (scalar-first); ``v_spatial = R(q) @ v_ref``.
    ``psi`` is the Vlasov warping amplitude [m²] (scalar seventh DOF).
    ``spin_accum`` stores the sum of Newton spin increments (spatial) for output.
    """

    x: NDArray[np.float64]
    q: NDArray[np.float64]
    psi: float = 0.0
    spin_accum: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def copy(self) -> "NodeState":
        return NodeState(
            x=self.x.copy(),
            q=self.q.copy(),
            psi=float(self.psi),
            spin_accum=self.spin_accum.copy(),
        )


@dataclass
class SectionStation:
    """
    Spanwise tabulated section data.

    ``K6`` / ``K7`` are in the elastic-centroid section frame. When ``K7`` is
    omitted, the solver synthesises ``K7 = block_diag(K6, K₍ww₎)`` with
    ``K₍ww₎ = max(K6[3,3], 1e-6)`` and zero ``k_w`` coupling.
    """

    z: float
    K6: NDArray[np.float64]
    K7: Optional[NDArray[np.float64]] = None
    M6: Optional[NDArray[np.float64]] = None
    elastic_center: Optional[NDArray[np.float64]] = None
    mass_center: Optional[NDArray[np.float64]] = None
    shear_center: Optional[NDArray[np.float64]] = None


@dataclass(frozen=True)
class SectionStiffness:
    """Classical extension, bending, torsion, and shear stiffness scalars for one section [SI]."""

    EA: float
    EI_x: float
    EI_y: float
    GJ: float
    GA_x: float
    GA_y: float
    #: Bending–bending coupling (e.g. from GBT): enters ``K6`` as ``-EIyz`` off-diagonal.
    EIyz: float = 0.0


@dataclass(frozen=True)
class SectionStiffnessArray:
    """
    Tabulated classical section stiffnesses along span coordinate ``s`` [m].

    Each array has shape ``(n_stations,)``; ``s`` must be strictly increasing.
    """

    s: NDArray[np.float64]
    EA: NDArray[np.float64]
    EI_x: NDArray[np.float64]
    EI_y: NDArray[np.float64]
    GJ: NDArray[np.float64]
    GA_x: NDArray[np.float64]
    GA_y: NDArray[np.float64]
    #: Omitted at construction becomes zeros matching ``s`` (frozen-safe in ``__post_init__``).
    EIyz: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=np.float64).ravel()
        n = s.size
        object.__setattr__(self, "s", s)
        if self.EIyz is None:
            object.__setattr__(self, "EIyz", np.zeros_like(s, dtype=np.float64))
        for name in ("EA", "EI_x", "EI_y", "GJ", "GA_x", "GA_y", "EIyz"):
            a = np.asarray(getattr(self, name), dtype=np.float64).ravel()
            if a.shape[0] != n:
                raise ValueError(f"SectionStiffnessArray.{name} length {a.shape[0]} != len(s)={n}.")
            object.__setattr__(self, name, a)
        if n >= 2 and np.any(np.diff(s) <= 0):
            raise ValueError("SectionStiffnessArray.s must be strictly increasing.")


@dataclass(frozen=True)
class K7Array:
    """
    Tabulated ``(7, 7)`` section stiffness ``K7`` along span coordinate ``s`` [m].

    ``entries`` has shape ``(n_stations, 7, 7)``; ``s`` must be strictly increasing
    when ``n_stations >= 2``.
    """

    s: NDArray[np.float64]
    entries: NDArray[np.float64]

    def __post_init__(self) -> None:
        s = np.asarray(self.s, dtype=np.float64).ravel()
        e = np.asarray(self.entries, dtype=np.float64)
        if e.ndim != 3 or e.shape[1:] != (7, 7):
            raise ValueError(f"K7Array.entries must have shape (n_stations, 7, 7), got {e.shape}.")
        if e.shape[0] != s.size:
            raise ValueError(f"K7Array: len(s)={s.size} != entries.shape[0]={e.shape[0]}.")
        if s.size >= 2 and np.any(np.diff(s) <= 0):
            raise ValueError("K7Array.s must be strictly increasing.")


@dataclass
class BeamElement:
    """Two-node straight reference element in the reference configuration."""

    node_ids: Tuple[int, int]
    L0: float
    z_mid: float
    #: Optional classical stiffness at element mid-span (diagnostics only; solver uses ``SectionStation``).
    debug_section_stiffness: Any = None


@dataclass
class BeamModel:
    """
    Discrete beam on a reference polyline.

    ``z_node[i]`` is arc-length at node ``i`` when provided; otherwise inferred
    from element ``L0`` starting at 0.
    ``kappa0_node`` and ``chi0_node`` prescribe the stress-free reference
    (precurvature / initial warping rate) in the **material frame** matching the
    discrete Reissner curvature measure ``Ω`` at Gauss points.
    """

    X_ref: NDArray[np.float64]
    elements: List[BeamElement]
    section_stations: Optional[List[SectionStation]] = None
    span_axis: int = 2
    z_node: Optional[NDArray[np.float64]] = None
    kappa0_node: Optional[NDArray[np.float64]] = None
    chi0_node: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.z_node is None and self.elements:
            n = int(self.X_ref.shape[0])
            z = np.zeros(n, dtype=np.float64)
            for el in self.elements:
                i, j = el.node_ids
                z[j] = z[i] + el.L0
            self.z_node = z

    @property
    def n_nodes(self) -> int:
        return int(self.X_ref.shape[0])

    @classmethod
    def from_blade_geometry(
        cls,
        geometry: "BladeGeometry",
        n_nodes: int,
        section_stations: List[SectionStation],
        *,
        span_axis: int = 2,
        align_section_stations: bool = False,
    ) -> "BeamModel":
        from ..engine.blade_geometry import beam_model_from_blade_geometry

        return beam_model_from_blade_geometry(
            geometry,
            n_nodes,
            section_stations,
            span_axis=span_axis,
            align_section_stations=align_section_stations,
        )


@dataclass
class BoundaryCondition:
    """
    Homogeneous Dirichlet BCs.

    Local DOF indices per node: ``0..2`` translations, ``3..5`` spins, ``6`` warping.
    """

    node_id: int
    fixed_dofs: Tuple[int, ...]


@dataclass
class BeamLoads:
    """Dead loads in the **global** frame plus optional boundary conditions.

    Load convention (I.4)
    ----------------------
    Loads from DLC envelope tabulation are **non-follower** forces/moments in
    the **undeformed global frame** (``frame = 'undeformed_global'``).
    The in-loop adapter (``build_extreme_load_beam_loads`` in ``beam_k7.py``)
    must set ``frame`` accordingly.  Do NOT borrow ``BeamLoads`` from
    ``BeamModelStage`` operating-load instances — they may carry a different
    loading convention and/or be mutated by the operating solver.

    Warping BC (I.3)
    ----------------
    No bimoment end loads are applied at ``z=L`` (tip).  Warping DOF ``psi``
    is clamped (psi=0) at the root node via ``bcs`` with ``fixed_dofs=(6,)``
    at ``node_id=0``.
    """

    nodal_F: NDArray[np.float64]
    nodal_M: NDArray[np.float64]
    distributed_q: Optional[NDArray[np.float64]] = None
    distributed_mz: Optional[NDArray[np.float64]] = None
    """Distributed torsion about local x per unit arc length [N·m/m], per element or scalar.

    The load is assembled onto rotational DOF 3 (torsion), not warping DOF 6.
    """
    bcs: List[BoundaryCondition] = field(default_factory=list)
    #: ``'undeformed_global'`` (default): ``nodal_F``/``nodal_M`` in global frame;
    #  ``'node_corotated'``: body-fixed with current node rotation ``R(q)`` (follower-style).
    frame: str = "undeformed_global"


# Backward compatibility
LoadCase = BeamLoads


@dataclass
class SolverOptions:
    """
    Newton–Raphson options for the 7-DOF beam static solve.

    **Tangent.** By default ``full_fd_hessian`` is True: the element Hessian is built by
    finite-differencing the element energy gradient (more consistent NR tangent, much more
    costly than the material-only tangent). Set ``full_fd_hessian`` False for the elastic
    ``BᵀK₇B`` (material) stiffness from a finite-difference ``B`` matrix, which is faster but
    omits the geometric (stress-dependent) part of the exact Hessian for nonlinear ``e7(q)``.

    When using the FD Hessian path, nested differencing can yield a nearly indefinite reduced
    matrix. When ``project_fd_hessian_spd`` is True, eigenvalues of the reduced ``K_ff`` are
    floored using ``fd_hessian_eig_floor_rel`` before the linear solve.
    """

    max_iter: int = 35
    tol_res: float = 1e-8
    tol_res_rel: float = 1e-6
    tol_du: float = 1e-10
    n_gauss: int = 2
    fd_eps: float = 1e-7
    hess_eps: float = 1e-6
    full_fd_hessian: bool = True
    #: When True and ``full_fd_hessian``, eigenvalues of reduced ``K_ff`` are floored so the
    #: tangent is SPD (nested FD Hessian can be slightly indefinite).
    project_fd_hessian_spd: bool = True
    #: Minimum eigenvalue relative to ``max(eig, 1)`` when projecting (see ``project_fd_hessian_spd``).
    fd_hessian_eig_floor_rel: float = 1e-10
    n_load_steps: int = 1
    #: When a proportional load sub-step does not converge, halve the step until this minimum.
    adaptive_load_min_step: float = 1e-5
    #: Maximum number of load-sub-step halvings (prevents infinite loops on hard targets).
    adaptive_load_bisect_max: int = 32
    tangent_rho: float = 0.0
    relax_factor: float = 1.0
    spin_stabilization: float = 0.0
    warping_stabilization: float = 0.0
    accept_stagnation: bool = True
    stagnation_window: int = 4
    verbose: bool = False
    #: If set, ``tol_mixed`` also includes ``tol_res_rel_rhs * rhs_ref`` where ``rhs_ref``
    #: is ``||rhs_f||`` on the first NR iteration of each load increment (backward compatible:
    #: ``None`` disables).
    tol_res_rel_rhs: Optional[float] = None
    #: Floor on stagnation ``cap`` as a fraction of ``||F_ext_full||`` (``0`` = legacy).
    cap_floor_rel: float = 0.0
    #: Backtracking line search on the free DOF increment (merit = reduced residual norm).
    line_search: bool = False
    line_search_shrink: float = 0.5
    line_search_min_scale: float = 0.05
    line_search_max_trials: int = 8
    #: Grow ``tangent_rho`` when the reduced residual fails to decrease across iterations.
    adaptive_tangent_rho: bool = False
    adaptive_rho_growth: float = 2.0
    adaptive_rho_max: float = 0.1
    # Arc-length (optional; load stepping used when disabled)
    use_arc_length: bool = False
    #: Spherical Riks: arc constraint ``||u-u0||^2 + c^2(λ-λ0)^2 = s^2``; ``c`` weights load vs displacements.
    arc_length_scale_lambda: float = 1.0
    arc_length_target: float = 0.0
    arc_length_max_iter: int = 8
    #: Finite-difference step for ``∂F_ext/∂q`` when using follower (corotated) loads in the tangent.
    follower_jacobian_eps: float = 1e-5
    # Cheap diagonal max/min ratio on ``K_ff`` (post floor when projection is on). For exact
    # ``cond`` via SVD, set ``log_full_condition_number`` True (much slower in NR).
    log_full_condition_number: bool = False
    # J.3: global buckling eigenvalue extraction after convergence
    extract_buckling: bool = False
    """When True, solve a generalized buckling problem at the converged state and
    store eigenvalues/modeshapes.  The tangent is ``K_t`` (analytic: ``B^T K7 B`` plus
    stress-stiffness, or full FD if ``full_fd_hessian``).  The geometric mass matrix
    is the assembled stress (initial) geometric stiffness from ``r_m ∂²e_m/∂q∂qᵀ``."""
    n_buckling_modes: int = 5
    """Number of lowest buckling eigenvalues to extract (J.3)."""
    #: Use complex-step (h=1e-20j) gradient as primary NR force vector. Set False to fall
    #: back to the legacy central-difference gradient (for debugging only).
    use_cs_gradient: bool = True


SolveOptions = SolverOptions


@dataclass
class BeamSolveResult:
    nodal_positions: NDArray[np.float64]  # (n_node, 3)
    nodal_rotations: NDArray[np.float64]  # (n_node, 3) total rotation vector vs ref
    nodal_R: NDArray[np.float64]  # (n_node, 3, 3)
    nodal_warping: NDArray[np.float64]  # (n_node,)
    resultants: NDArray[np.float64]  # (n_station, 7)
    strains: NDArray[np.float64]  # (n_station, 7)
    converged: bool
    n_iterations: int
    residual_norm: float
    iteration_history: List[Dict[str, float]]  # Per-iteration diagnostics (residual, du, step metadata).
    # Legacy / debug
    nodes: Optional[List[NodeState]] = None
    reactions: Optional[Dict[Tuple[int, int], float]] = None
    z_stations_out: Optional[NDArray[np.float64]] = None
    # Gauss→nodal projection (spanwise, one row per mesh node)
    z_nodal_out: Optional[NDArray[np.float64]] = None
    strains_nodal: Optional[NDArray[np.float64]] = None
    resultants_nodal: Optional[NDArray[np.float64]] = None
    # Section grid (aligned with section_properties ``station_z``)
    z_section_recovery: Optional[NDArray[np.float64]] = None
    section_stress_voigt_gp: Optional[NDArray[np.float64]] = None  # (n_s, 3) ply |σ| max envelope, GP path
    section_stress_voigt_nodal: Optional[NDArray[np.float64]] = None
    section_strain_maxabs_gp: Optional[NDArray[np.float64]] = None  # (n_s, 6) laminate strain envelope
    section_strain_maxabs_nodal: Optional[NDArray[np.float64]] = None
    section_hashin_fi_max_gp: Optional[NDArray[np.float64]] = None  # (n_s,)
    section_hashin_fi_max_nodal: Optional[NDArray[np.float64]] = None
    section_von_mises_fi_max_gp: Optional[NDArray[np.float64]] = None  # (n_s,) isotropic subcomponents
    section_von_mises_fi_max_nodal: Optional[NDArray[np.float64]] = None
    # Section-frame ply stress envelope (``blade_utilities.recovery.apply_section_stress_operator``)
    section_stress_voigt_secframe_gp: Optional[NDArray[np.float64]] = None  # (n_s, 3)
    section_stress_voigt_secframe_nodal: Optional[NDArray[np.float64]] = None
    # Spanwise d(FI)/dz via bundle ``D_z`` (``blade_utilities.recovery.apply_span_derivative``)
    section_d_hashin_fi_dz_gp: Optional[NDArray[np.float64]] = None
    section_d_hashin_fi_dz_nodal: Optional[NDArray[np.float64]] = None
    # Hashin FI max over composite subcomponents per (station, ply); shape (n_s, n_ply_max)
    section_hashin_fi_ply_envelope_gp: Optional[NDArray[np.float64]] = None
    section_hashin_fi_ply_envelope_nodal: Optional[NDArray[np.float64]] = None
    # J.3: global buckling stiffness matrices (populated when SolverOptions.extract_buckling=True)
    tangent_stiffness: Optional[NDArray[np.float64]] = None
    """(n_free, n_free) reduced tangent stiffness K_ff at converged state (J.3)."""
    geometric_stiffness: Optional[NDArray[np.float64]] = None
    """(n_free, n_free) geometric (stress) stiffness K_g at converged state (J.3)."""
    global_buckling_lambdas: Optional[NDArray[np.float64]] = None
    """Lowest N global buckling eigenvalues lambda from (K_t - lambda K_g)phi=0 (J.3)."""
    global_buckling_modeshapes: Optional[NDArray[np.float64]] = None
    """(N_modes, n_nodes, 7) global buckling modeshapes (J.5)."""


SolveResult = BeamSolveResult


@runtime_checkable
class BeamSolverProtocol(Protocol):
    def solve(
        self,
        model: BeamModel,
        loads: BeamLoads,
        options: SolverOptions,
    ) -> BeamSolveResult: ...


def default_initial_state(model: BeamModel) -> List[NodeState]:
    out: List[NodeState] = []
    for i in range(model.n_nodes):
        x = model.X_ref[i].copy()
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out.append(
            NodeState(
                x=x,
                q=q,
                psi=0.0,
                spin_accum=np.zeros(3, dtype=np.float64),
            )
        )
    return out
