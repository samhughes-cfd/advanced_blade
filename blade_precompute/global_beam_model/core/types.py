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

Use :func:`global_beam_model.engine.constitutive.resultants_to_recovery6` to map the first six
entries to ``section_model`` order ``[N, My, Mz, T, Vy, Vz]`` for downstream section
recovery tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

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


@dataclass
class BeamElement:
    """Two-node straight reference element in the reference configuration."""

    node_ids: Tuple[int, int]
    L0: float
    z_mid: float


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
    """Dead loads in the **global** frame plus optional boundary conditions."""

    nodal_F: NDArray[np.float64]
    nodal_M: NDArray[np.float64]
    distributed_q: Optional[NDArray[np.float64]] = None
    distributed_mz: Optional[NDArray[np.float64]] = None
    """Distributed torsion about local x per unit arc length [N·m/m], per element or scalar."""
    bcs: List[BoundaryCondition] = field(default_factory=list)


# Backward compatibility
LoadCase = BeamLoads


@dataclass
class SolverOptions:
    max_iter: int = 35
    tol_res: float = 1e-8
    tol_res_rel: float = 1e-6
    tol_du: float = 1e-10
    n_gauss: int = 2
    fd_eps: float = 1e-7
    hess_eps: float = 1e-6
    full_fd_hessian: bool = False
    n_load_steps: int = 1
    tangent_rho: float = 0.0
    relax_factor: float = 1.0
    spin_stabilization: float = 0.0
    warping_stabilization: float = 0.0
    accept_stagnation: bool = True
    stagnation_window: int = 4
    verbose: bool = False
    # Arc-length (optional; load stepping used when disabled)
    use_arc_length: bool = False
    arc_length_target: float = 0.0
    arc_length_max_iter: int = 8


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
    iteration_history: List[Dict[str, float]]
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
    section_tsai_wu_fi_max_gp: Optional[NDArray[np.float64]] = None  # (n_s,)
    section_tsai_wu_fi_max_nodal: Optional[NDArray[np.float64]] = None
    section_von_mises_fi_max_gp: Optional[NDArray[np.float64]] = None  # (n_s,) isotropic subcomponents
    section_von_mises_fi_max_nodal: Optional[NDArray[np.float64]] = None
    section_delamination_fi_max_gp: Optional[NDArray[np.float64]] = None  # (n_s,) Tier-3 only
    section_delamination_fi_max_nodal: Optional[NDArray[np.float64]] = None
    # Section-frame ply stress envelope (``blade_utilities.recovery.apply_section_stress_operator``)
    section_stress_voigt_secframe_gp: Optional[NDArray[np.float64]] = None  # (n_s, 3)
    section_stress_voigt_secframe_nodal: Optional[NDArray[np.float64]] = None
    # Spanwise d(FI)/dz via bundle ``D_z`` (``blade_utilities.recovery.apply_span_derivative``)
    section_d_tsai_wu_fi_dz_gp: Optional[NDArray[np.float64]] = None
    section_d_tsai_wu_fi_dz_nodal: Optional[NDArray[np.float64]] = None
    # Tsai–Wu FI max over composite subcomponents per (station, ply); shape (n_s, n_ply_max)
    section_tsai_wu_fi_ply_envelope_gp: Optional[NDArray[np.float64]] = None
    section_tsai_wu_fi_ply_envelope_nodal: Optional[NDArray[np.float64]] = None


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
