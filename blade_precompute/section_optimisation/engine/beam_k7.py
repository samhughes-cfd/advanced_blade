"""
Linear ``K7`` beam driver for optimisation (Tier B).

The extreme-load envelope is taken as **prescribed internal resultants** at
tabulated stations (common for ultimate blade checks). The optional seventh
component is the bimoment ``B`` (defaults to zero unless ``ExtremeLoads.B`` is
set). ``nodal_R`` applies a level-1 rigid rotation from ``kappa0`` at each station
via :func:`global_beam_model.engine.kinematics.rotmat_from_small_curvature`.

Load convention (I.4)
---------------------
``BeamLoads`` produced by the in-loop adapter (``extreme_loads_to_beam_loads``)
are **non-follower** loads in the **undeformed global frame**.  This matches the
convention under which DLC envelopes are tabulated.  Do NOT borrow or mutate
``BeamLoads`` instances from ``BeamModelStage`` (operating loads) in the
optimisation loop (I.5).

Warping BC convention (I.3)
----------------------------
The warping DOF ``psi`` is clamped (``psi = 0``) at the root (z=0) and free at
the tip (z=L).  No end bimoment loads are applied at z=L.  The same convention
must be used in shell K77 homogenisation (I.6 / F2.1) so that ``K7[6,6]``
produced by the shell matches the beam's warping stiffness definition.

NR warm-start auto-fallback (I.8)
-----------------------------------
When using the coupled FE driver (``CoupledFEResultantDriver``, Group H), if
``solve_static`` returns ``converged=False`` from a warm start the driver must
retry from cold-start before raising.  See ``solve_static_with_warm_start_fallback``
below for the helper contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.engine.kinematics import rotmat_from_small_curvature

from ..core.types import ExtremeLoads, OptimBladeGeometry


# ---------------------------------------------------------------------------
# I.1 — Bimoment derivative utility
# ---------------------------------------------------------------------------

def bimoment_derivative_z(
    B: NDArray[np.float64],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Central-difference estimate of ``dB/dz`` along the span.

    Used to pass the correct bimoment rate into the shell FI recovery (I.1).
    Without this, web shear flow is missing the bimoment-rate term and shear-web
    FI is biased low.

    Parameters
    ----------
    B : (n_s,) bimoment array [N·m²].
    z : (n_s,) station coordinates [m], strictly increasing.

    Returns
    -------
    dB_dz : (n_s,) array [N·m].
        Central differences at interior points; forward/backward at the ends.
    """
    B = np.asarray(B, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    n = B.shape[0]
    if n < 2:
        return np.zeros_like(B)
    dBdz = np.empty_like(B)
    # Interior: central differences
    dBdz[1:-1] = (B[2:] - B[:-2]) / np.maximum(z[2:] - z[:-2], 1e-12)
    # Boundaries: one-sided
    dBdz[0] = (B[1] - B[0]) / max(float(z[1] - z[0]), 1e-12)
    dBdz[-1] = (B[-1] - B[-2]) / max(float(z[-1] - z[-2]), 1e-12)
    return dBdz


# ---------------------------------------------------------------------------
# I.2 — Section-frame rotation helper
# ---------------------------------------------------------------------------

def section_frame_rotation_matrix(
    twist_deg: float,
    kappa0: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the 3×3 rotation ``R_section_from_beam = R_twist · R_kappa0``.

    The coupled FE returns resultants in the beam-element frame (which already
    includes ``nodal_R`` from the NR solution).  The shell FI recovery expects
    resultants in the **section principal frame** (twisted + kappa0-rotated).
    Apply this rotation to the 3×3 force/moment sub-vector before calling
    ``run_section_both``.

    Parameters
    ----------
    twist_deg
        Structural blade twist at this station [deg].
    kappa0
        Pre-curvature vector ``[kx, ky, kz]`` at this station [1/m].

    Returns
    -------
    R : (3, 3) float64 rotation matrix.
    """
    theta = np.deg2rad(float(twist_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R_twist = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    R_kappa0 = rotmat_from_small_curvature(np.asarray(kappa0, dtype=np.float64))
    return R_kappa0 @ R_twist


# ---------------------------------------------------------------------------
# I.5 — In-loop BeamLoads builder (non-follower, undeformed global frame)
# ---------------------------------------------------------------------------

def build_extreme_load_beam_loads(
    extreme_loads: ExtremeLoads,
    blade_geometry: OptimBladeGeometry,
    beam_model: Any,
) -> Any:
    """Build ``BeamLoads`` from ``ExtremeLoads`` for the in-loop coupled FE driver.

    The resulting loads are **non-follower** forces in the **undeformed global frame**
    (I.4 convention), consistent with how DLC envelopes are tabulated.

    This function must NEVER share or mutate ``BeamLoads`` from ``BeamModelStage``
    (which carries operating loads: gravity + aero).  Always call this function from
    ``CoupledFEResultantDriver._build_extreme_load_beam_loads`` (I.5).

    Implementation note (Group H)
    ------------------------------
    The full implementation differentiates prescribed resultant profiles
    ``(N, Vy, Vz, My, Mz, T)(z)`` into equivalent distributed nodal loads on
    the ``BeamModel`` discretisation.  This stub returns a zero-load placeholder
    that will be replaced by the H0.1 adapter when Group H is implemented.

    Parameters
    ----------
    extreme_loads
        Prescribed spanwise resultants from the ``.dat`` file.
    blade_geometry
        Optimisation blade geometry (stations, z-coordinates, chord, twist).
    beam_model
        Assembled ``BeamModel`` skeleton (nodes, elements, X_ref).

    Returns
    -------
    ``BeamLoads`` with ``frame='undeformed_global'`` tag (non-follower, I.4).
    """
    from blade_precompute.global_beam_model.core.types import BeamLoads

    n_nodes = int(beam_model.n_nodes)
    nodal_F = np.zeros((n_nodes, 3), dtype=np.float64)
    nodal_M = np.zeros((n_nodes, 3), dtype=np.float64)

    warnings.warn(
        "build_extreme_load_beam_loads: returning zero-load stub. "
        "Full H0.1 implementation (resultant differentiation into distributed loads) "
        "must replace this before using CoupledFEResultantDriver.",
        stacklevel=2,
    )
    return BeamLoads(nodal_F=nodal_F, nodal_M=nodal_M)


# ---------------------------------------------------------------------------
# I.8 — NR warm-start auto-fallback helper
# ---------------------------------------------------------------------------

def solve_static_with_warm_start_fallback(
    model: Any,
    loads: Any,
    options: Any,
    *,
    initial_nodes: Any | None = None,
    bcs: Any | None = None,
    logger: Any | None = None,
) -> Any:
    """Wrap ``solve_static`` with a warm-start + cold-start auto-fallback (I.8).

    If a warm-started solve returns ``converged=False``, the driver retries
    once from cold-start.  If the cold-start also diverges, it raises
    ``RuntimeError`` with the diagnostic payload (H3.3).

    Parameters
    ----------
    model : BeamModel
    loads : BeamLoads
    options : SolverOptions
    initial_nodes
        Warm-start node state list; when None a cold-start is performed.
    bcs
        Explicit boundary conditions (may be None).
    logger
        Optional :class:`RunLogger` for fallback event logging.

    Returns
    -------
    ``BeamSolveResult``
    """
    from blade_precompute.global_beam_model.engine.solver import solve_static

    # ---- attempt 1: warm start (or cold if no initial_nodes) ----
    warm_start_used = initial_nodes is not None

    result = solve_static(model, loads, options, bcs=bcs)
    if result.converged:
        return result

    prev_res = float(result.residual_norm)
    nr_warm = int(result.n_iterations)

    if not warm_start_used:
        raise RuntimeError(
            f"solve_static (cold-start) did not converge. "
            f"residual_norm={prev_res:.3e}, n_iter={nr_warm}. "
            "Check K7 condition number and applied loads."
        )

    # ---- fallback: cold-start ----
    if logger is not None:
        try:
            logger.warn_event(
                "beam_solver.warm_start_fallback",
                warm_residual_norm=prev_res,
                nr_iters_warm=nr_warm,
                message="Warm start did not converge; retrying from cold-start.",
            )
        except Exception:
            pass

    cold_result = solve_static(model, loads, options, bcs=bcs)
    if cold_result.converged:
        return cold_result

    raise RuntimeError(
        f"solve_static failed on both warm-start and cold-start. "
        f"warm residual={prev_res:.3e} ({nr_warm} iters), "
        f"cold residual={float(cold_result.residual_norm):.3e} "
        f"({int(cold_result.n_iterations)} iters). "
        "Halt: downstream shell FI and buckling results would be garbage."
    )


@dataclass
class PrescribedResultantBeamState:
    """Beam driver output normalised to section/K7 order ``[N, My, Mz, T, Vy, Vz, B]``."""

    resultants: NDArray[np.float64]
    nodal_R: NDArray[np.float64]
    nodal_R_source: str = "small_curvature_kappa0"
    # Populated by GlobalBeamResultantDriver for tip deflection / provenance
    beam_solve: Any | None = None
    tip_displacement_m: NDArray[np.float64] | None = None
    """Tip node translational displacement vs reference [m], shape ``(3,)``; None for Tier-B."""


class PrescribedResultantDriver:
    """Default Tier-B driver using prescribed internal resultants."""

    __slots__ = ()

    def drive(
        self,
        K7_stack: NDArray[np.float64],
        extreme_loads: ExtremeLoads,
        blade_geometry: OptimBladeGeometry,
        *,
        K6_stack: NDArray[np.float64] | None = None,
        mass_per_length: NDArray[np.float64] | None = None,
    ) -> PrescribedResultantBeamState:
        del K6_stack, mass_per_length
        return solve(K7_stack, extreme_loads, blade_geometry)


def solve(
    K7_stack: NDArray[np.float64],
    extreme_loads: ExtremeLoads,
    blade_geometry: OptimBladeGeometry,
    *,
    nodal_R_override: NDArray[np.float64] | None = None,
) -> PrescribedResultantBeamState:
    """
    Parameters
    ----------
    K7_stack
        ``(n_s, 7, 7)`` section stiffness tables (used for consistency checks;
        internal resultants follow ``ExtremeLoads`` directly in this driver).
    """
    n_s = int(blade_geometry.z_stations.shape[0])
    if K7_stack.shape[0] != n_s:
        raise ValueError("K7_stack first axis must match number of stations.")
    B = extreme_loads.bimoment()
    R = np.stack(
        [
            extreme_loads.N,
            extreme_loads.My,
            extreme_loads.Mz,
            extreme_loads.T,
            extreme_loads.Vy,
            extreme_loads.Vz,
            B,
        ],
        axis=1,
    ).astype(np.float64)
    if nodal_R_override is not None:
        nodal_R = np.asarray(nodal_R_override, dtype=np.float64)
        if nodal_R.shape != (n_s, 3, 3):
            raise ValueError("nodal_R_override must have shape (n_station, 3, 3).")
        return PrescribedResultantBeamState(
            resultants=R, nodal_R=nodal_R, nodal_R_source="override"
        )
    nodal_R = np.zeros((n_s, 3, 3), dtype=np.float64)
    for i in range(n_s):
        nodal_R[i] = rotmat_from_small_curvature(blade_geometry.kappa0[i])
    return PrescribedResultantBeamState(
        resultants=R, nodal_R=nodal_R, nodal_R_source="small_curvature_kappa0"
    )
