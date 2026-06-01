"""Tier-A global beam resultant driver for optimisation (distributed loads, equilibrated R)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.api import BeamAnalysis
from blade_precompute.global_beam_model.engine.axial_loading import (
    AxialLoadingConfig,
    q_x_distributed,
)
from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, SolverOptions
from blade_precompute.global_beam_model.engine.constitutive import (
    beam_resultants_to_section_recovery_order,
)
from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z
from blade_precompute.global_beam_model.engine.solver import solve_static
from blade_precompute.section_optimisation.core.types import (
    DistributedLoadCurves,
    ExtremeLoads,
    OptimBladeGeometry,
)
from blade_precompute.section_optimisation.engine.beam_k7 import PrescribedResultantBeamState


def default_global_beam_solver_options() -> SolverOptions:
    """Match :func:`~blade_precompute.orchestration.precompute.stages.beam_model_impl` defaults."""
    return SolverOptions(
        max_iter=110,
        tol_res=5e-2,
        tol_res_rel=5e-3,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=72,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        relax_factor=0.9,
        verbose=False,
        tol_res_rel_rhs=0.035,
        cap_floor_rel=0.055,
        line_search=False,
    )


def build_beam_loads_distributed(
    geom: BladeGeometry,
    model: Any,
    curves: DistributedLoadCurves,
) -> BeamLoads:
    """Construct distributed ``BeamLoads`` from precompute-style curves (undeformed frame)."""
    n_nodes = int(model.n_nodes)
    n_elem = int(len(model.elements))
    z_mid = np.asarray([el.z_mid for el in model.elements], dtype=np.float64)

    qy = np.interp(z_mid, curves.loads_r_z_m, curves.q_y_Npm)
    qz = np.interp(z_mid, curves.loads_r_z_m, curves.q_z_Npm)
    mx = np.interp(z_mid, curves.loads_r_z_m, curves.m_x_Nmpm)
    # Global line load; spanwise beam axis = ``model.span_axis`` (default 2 = global z).
    sa = int(getattr(model, "span_axis", 2))
    if sa not in (0, 1, 2):
        raise ValueError(f"span_axis must be 0..2, got {sa}.")

    distributed_q = np.zeros((n_elem, 3), dtype=np.float64)
    distributed_q[:, 1] = qy
    distributed_q[:, 2] = qz
    if curves.q_x_Npm is not None:
        q_ax = np.interp(
            z_mid, curves.loads_r_z_m, np.asarray(curves.q_x_Npm, dtype=np.float64)
        )
        # Centrifugal + gravity line load along the 1D reference (same global axis as span).
        distributed_q[:, sa] = distributed_q[:, sa] + q_ax

    return BeamLoads(
        nodal_F=np.zeros((n_nodes, 3), dtype=np.float64),
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        distributed_q=distributed_q,
        distributed_mz=np.asarray(mx, dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )


def _radial_m_on_stations(blade_geometry: OptimBladeGeometry, n: int) -> NDArray[np.float64]:
    if blade_geometry.radial_r_m is not None:
        r = np.asarray(blade_geometry.radial_r_m, dtype=np.float64).ravel()
        if int(r.shape[0]) == n:
            return r
    rr = np.asarray(blade_geometry.r_ref, dtype=np.float64)
    if int(rr.shape[0]) != n:
        raise ValueError("r_ref row count must match station count when radial_r_m is unset.")
    return np.sqrt(rr[:, 0] ** 2 + rr[:, 1] ** 2)


def _interp_nodal_R_on_z(
    z_nodes: NDArray[np.float64],
    R_nodes: NDArray[np.float64],
    z_q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linearly interpolate each entry of ``(n_node,3,3)`` nodal_R to query z (1D)."""
    zn = np.asarray(z_nodes, dtype=np.float64).ravel()
    R = np.asarray(R_nodes, dtype=np.float64)
    zq = np.asarray(z_q, dtype=np.float64).ravel()
    out = np.zeros((zq.shape[0], 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            out[:, a, b] = np.interp(zq, zn, R[:, a, b])
    return out


@dataclass
class GlobalBeamResultantDriver:
    """Equilibrium resultants from Tier-A :class:`BeamAnalysis` (same loads as precompute beam stage)."""

    distributed_loads: DistributedLoadCurves
    n_beam_nodes: int
    solver_options: SolverOptions | None = None
    axial_cfg: AxialLoadingConfig | None = None

    def drive(
        self,
        K7_stack: NDArray[np.float64],
        extreme_loads: ExtremeLoads,
        blade_geometry: OptimBladeGeometry,
        *,
        K6_stack: NDArray[np.float64] | None = None,
        mass_per_length: NDArray[np.float64] | None = None,
    ) -> PrescribedResultantBeamState:
        del extreme_loads  # Equilibrium uses distributed_loads only (ExtremeLoads kept for API parity).
        n_s = int(blade_geometry.z_stations.shape[0])
        if K7_stack.shape[0] != n_s:
            raise ValueError("K7_stack first axis must match blade_geometry.z_stations count.")
        if K6_stack is None:
            K6_stack = K7_stack[:, :6, :6].copy()
        else:
            if K6_stack.shape[0] != n_s:
                raise ValueError("K6_stack first axis must match number of stations.")

        z = np.asarray(blade_geometry.z_stations, dtype=np.float64).ravel()
        stations = stations_from_arrays(z, K6_stack, K7_stack)

        geom = BladeGeometry(
            z_stations=np.asarray(blade_geometry.z_stations, dtype=np.float64),
            r_ref=np.asarray(blade_geometry.r_ref, dtype=np.float64),
            kappa0=np.asarray(blade_geometry.kappa0, dtype=np.float64),
            chord=np.asarray(blade_geometry.chord, dtype=np.float64),
            twist=np.asarray(blade_geometry.twist, dtype=np.float64),
            airfoil_profiles=list(blade_geometry.airfoil_profiles),
            web_positions=np.asarray(blade_geometry.web_positions, dtype=np.float64),
            subcomponent_materials=dict(blade_geometry.subcomponent_materials),
            chi0=None,
        )

        analysis = BeamAnalysis.from_blade_geometry(
            geom, int(self.n_beam_nodes), stations, span_axis=2
        )
        model = analysis.model
        curves = self.distributed_loads
        if self.axial_cfg is not None and self.axial_cfg.enabled and mass_per_length is not None:
            zq = np.asarray(curves.loads_r_z_m, dtype=np.float64).ravel()
            mu_b = np.asarray(mass_per_length, dtype=np.float64).ravel()
            r_b = _radial_m_on_stations(blade_geometry, n_s)
            if mu_b.shape[0] != n_s:
                raise ValueError("mass_per_length length must match blade station count.")
            mu_q = np.interp(zq, z, mu_b)
            r_q = np.interp(zq, z, r_b)
            qx = q_x_distributed(zq, r_q, mu_q, self.axial_cfg)
            curves = replace(curves, q_x_Npm=qx)
        loads = build_beam_loads_distributed(geom, model, curves)
        opt = self.solver_options if self.solver_options is not None else default_global_beam_solver_options()
        res = solve_static(model, loads, options=opt)

        if not bool(res.converged):
            raise RuntimeError(
                "Global beam solve did not converge; refusing to use partial resultants "
                f"(residual_norm={float(res.residual_norm):.3e}, "
                f"n_iterations={int(res.n_iterations)})."
            )
        if res.z_stations_out is None or res.resultants is None or res.z_stations_out.size < 1:
            raise RuntimeError("Global beam solve returned empty resultants for sampling.")
        R_beam = sample_resultants_at_z(z, res.z_stations_out, res.resultants)
        if R_beam.shape[0] != n_s or R_beam.shape[1] != 7:
            raise ValueError(f"Sampled resultants have shape {R_beam.shape}, expected ({n_s}, 7).")
        R_at = beam_resultants_to_section_recovery_order(R_beam)

        z_nodal = res.z_nodal_out
        if z_nodal is not None and res.nodal_R is not None and z_nodal.size == res.nodal_R.shape[0]:
            nodal_R = _interp_nodal_R_on_z(z_nodal, res.nodal_R, z)
        else:
            # Fallback: small-curvature frame from kappa0 (matches Tier-B)
            from blade_precompute.global_beam_model.engine.kinematics import rotmat_from_small_curvature

            nodal_R = np.zeros((n_s, 3, 3), dtype=np.float64)
            for i in range(n_s):
                nodal_R[i] = rotmat_from_small_curvature(np.asarray(blade_geometry.kappa0[i], dtype=np.float64))

        tip_disp = np.asarray(res.nodal_positions[-1] - model.X_ref[-1], dtype=np.float64)
        return PrescribedResultantBeamState(
            resultants=R_at.astype(np.float64),
            nodal_R=nodal_R.astype(np.float64),
            nodal_R_source="global_beam_tier_a",
            beam_solve=res,
            tip_displacement_m=tip_disp,
        )
