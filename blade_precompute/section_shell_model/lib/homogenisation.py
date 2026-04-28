"""
Cross-sectional stiffness homogenisation from the MITC4/CLPT shell model (Groups F2.1, I.6).

K7 is extracted via the energy bilinear form K7[m,n] = U_m^T @ K_global @ U_n,
where U_m is the unit-mode displacement field for beam mode m applied across a
unit-length (L_x = 1 m) shell slice, and K_global is the unconstrained MITC4
stiffness assembled by ``solve_global_coupled_mitc4(..., return_assembly_data=True)``.

The 7 beam modes and their DOF prescriptions on top-layer nodes (bottom = reference, zero):

  Mode 0  ε₀      : u_x = 1               (uniform axial extension)
  Mode 1  κ_y     : u_x = -(z - z_e)      (bending about y-axis)
  Mode 2  κ_z     : u_x = +(y - y_e)      (bending about z-axis)
  Mode 3  γ_t     : u_s = 1               (uniform shear / torsion)
  Mode 4  γ_s_y   : u_s = tz              (transverse shear y, tz = z-component of tangent)
  Mode 5  γ_s_z   : u_s = -ty             (transverse shear z, ty = y-component of tangent)
  Mode 6  ψ'      : u_x = ω̂(s)            (Vlasov warping; from section_vlasov)

All other DOFs (w, β_s, β_x) are zero — K7 depends only on the membrane
in-plane stiffness (A-matrix), consistent with the strip solver convention.

K7[6,6] verification (F2.2)
----------------------------
After computing K7, verify::

    abs(K7[6,6] - E_ref * I_omega_E) / (E_ref * I_omega_E) < 0.05

where ``I_omega_E`` is from ``SectionVlasovResult``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


# DOF indices per node — must match global_mitc4_assembly.py constants.
_U_X = 0
_U_S = 1
_W = 2
_BETA_S = 3
_BETA_X = 4
_NDOF_NODE = 5


# ---------------------------------------------------------------------------
# Section centroid helpers
# ---------------------------------------------------------------------------

def compute_elastic_centroid_from_panels(panels: list[Any]) -> tuple[float, float]:
    """A11-weighted elastic centroid (y_e, z_e) from panel laminates and geometry.

    Uses the A₁₁ entry of each panel's ABD matrix as the axial stiffness weight,
    consistent with the strip-solver convention.
    """
    from examples.section_stress_model.lib.laminate_clpt import abd_stack  # type: ignore[import-untyped]

    sum_A11_y = 0.0
    sum_A11_z = 0.0
    sum_A11 = 0.0
    for p in panels:
        try:
            A_mat, _, _ = abd_stack(p.lam.build_plies())
            A11 = float(A_mat[0, 0])
        except Exception:
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_p = np.asarray(p.s, dtype=float)
        if len(s_p) < 2 or len(nodes_yz) < 2:
            continue
        L_p = float(abs(s_p[-1] - s_p[0]))
        y_mid = float(np.mean(nodes_yz[:, 0]))
        z_mid = float(np.mean(nodes_yz[:, 1]))
        sum_A11_y += A11 * L_p * y_mid
        sum_A11_z += A11 * L_p * z_mid
        sum_A11 += A11 * L_p

    if sum_A11 < 1e-30:
        return 0.0, 0.0
    return float(sum_A11_y / sum_A11), float(sum_A11_z / sum_A11)


def section_mass_properties(panels: list[Any]) -> dict[str, float]:
    """Mass properties per unit spanwise length from panel laminates.

    Returns a dict with keys:
      ``mass_per_length`` [kg/m], ``y_mass_centre`` [m], ``z_mass_centre`` [m].
    """
    mass_per_length = 0.0
    sum_rho_L_y = 0.0
    sum_rho_L_z = 0.0
    sum_rho_L = 0.0
    for p in panels:
        try:
            rho = float(getattr(p.lam, "rho", None) or getattr(p, "rho", 0.0))
            t = float(p.lam.t)
        except Exception:
            continue
        nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
        s_p = np.asarray(p.s, dtype=float)
        if len(s_p) < 2 or len(nodes_yz) < 2:
            continue
        L_p = float(abs(s_p[-1] - s_p[0]))
        y_mid = float(np.mean(nodes_yz[:, 0]))
        z_mid = float(np.mean(nodes_yz[:, 1]))
        mass_per_length += rho * t * L_p
        sum_rho_L_y += rho * t * L_p * y_mid
        sum_rho_L_z += rho * t * L_p * z_mid
        sum_rho_L += rho * t * L_p

    denom = max(sum_rho_L, 1e-30)
    return {
        "mass_per_length": float(mass_per_length),
        "y_mass_centre": float(sum_rho_L_y / denom),
        "z_mass_centre": float(sum_rho_L_z / denom),
    }


# ---------------------------------------------------------------------------
# Unit-mode displacement matrix
# ---------------------------------------------------------------------------

def _build_unit_mode_matrix(
    node_meta: Any,
    y_e: float,
    z_e: float,
    omega_hat: NDArray | None,
) -> NDArray[np.float64]:
    """Build (n_gdof, 7) matrix U of unit beam-mode displacement fields.

    Only top-layer nodes (``node_meta.is_top == True``) carry non-zero entries.
    Bottom-layer nodes represent the clamped reference face (zero displacement).
    MPC virtual nodes with NaN coordinates are skipped.
    """
    n_nodes = node_meta.n_nodes
    n_gdof = n_nodes * _NDOF_NODE
    U = np.zeros((n_gdof, 7), dtype=np.float64)

    yz = node_meta.yz
    is_top = node_meta.is_top
    tan = node_meta.tangent_yz

    for gn in range(n_nodes):
        if not is_top[gn]:
            continue
        if np.isnan(yz[gn, 0]):
            continue  # virtual MPC cluster node — no direct physical coords

        y_i = float(yz[gn, 0])
        z_i = float(yz[gn, 1])
        ty_i = float(tan[gn, 0]) if not np.isnan(tan[gn, 0]) else 0.0
        tz_i = float(tan[gn, 1]) if not np.isnan(tan[gn, 1]) else 0.0
        base = gn * _NDOF_NODE

        # Axial DOF (u_x) modes
        U[base + _U_X, 0] = 1.0             # ε₀: uniform axial
        U[base + _U_X, 1] = -(z_i - z_e)   # κ_y: bending about y
        U[base + _U_X, 2] = (y_i - y_e)    # κ_z: bending about z
        if omega_hat is not None and not np.isnan(omega_hat[gn]):
            U[base + _U_X, 6] = float(omega_hat[gn])  # ψ': warping

        # Tangential shear DOF (u_s) modes
        U[base + _U_S, 3] = 1.0    # γ_t: uniform torsion shear
        U[base + _U_S, 4] = tz_i   # γ_s_y: shear-y (tz component of tangent)
        U[base + _U_S, 5] = -ty_i  # γ_s_z: shear-z (-ty component)

    return U


# ---------------------------------------------------------------------------
# K7 extraction
# ---------------------------------------------------------------------------

def compute_section_K7_from_shell(
    K_global: sp.spmatrix,
    node_meta: Any,
    y_e: float,
    z_e: float,
    omega_hat: NDArray | None = None,
    *,
    run_log: Any | None = None,
    k7_cond_warn_threshold: float = 1e10,
) -> NDArray[np.float64]:
    """Compute 7×7 cross-section stiffness via energy bilinear form K7 = U^T K_global U.

    Parameters
    ----------
    K_global
        Unconstrained global MITC4 stiffness matrix, shape ``(n_gdof, n_gdof)``.
        Obtained from ``solve_global_coupled_mitc4(..., return_assembly_data=True)``.
    node_meta
        :class:`~blade_precompute.section_shell_model.lib.global_mitc4_assembly.GlobalNodeMeta`
        returned alongside ``K_global``.
    y_e, z_e
        Elastic centroid coordinates [m] — e.g. from
        :func:`compute_elastic_centroid_from_panels`.
    omega_hat
        Normalized Vlasov warping function ω̂ evaluated at each global node
        (same indexing as ``node_meta``).  When ``None``, column/row 6 of K7
        is left as zero (K6 only).
    run_log
        Optional run logger for diagnostics.
    k7_cond_warn_threshold
        Warn if ``cond(K7) > threshold``.

    Returns
    -------
    K7 : (7, 7) float64
        Section stiffness matrix in the beam reference frame.
    """
    U = _build_unit_mode_matrix(node_meta, y_e, z_e, omega_hat)

    # K7 = U^T @ K_global @ U  (energy bilinear form, no linear solve needed)
    KU = K_global @ U     # (n_gdof, 7)  — sparse × dense
    K7: NDArray[np.float64] = U.T @ KU  # (7, 7)

    cond = float(np.linalg.cond(K7))
    if cond > k7_cond_warn_threshold:
        msg = (
            f"K7 condition number {cond:.2e} exceeds threshold "
            f"{k7_cond_warn_threshold:.2e} — check elastic centroid alignment."
        )
        if run_log is not None:
            try:
                run_log.warn_event("homogenisation.k7_ill_conditioned", cond=cond)
            except Exception:
                pass
        else:
            warnings.warn(msg, stacklevel=2)

    return K7


# ---------------------------------------------------------------------------
# K7[6,6] verification
# ---------------------------------------------------------------------------

def verify_k7_warping_stiffness(
    K7: NDArray[np.float64],
    I_omega_E: float,
    E_ref: float,
    *,
    tol: float = 0.05,
    run_log: Any | None = None,
    station_index: int | None = None,
) -> dict[str, float]:
    """Verify K7[6,6] against thin-wall Vlasov estimate ``E_ref * I_omega_E`` (F2.2).

    Returns a diagnostics dict with ``K7_66``, ``vlasov_EIomega``, and ``ratio``.
    Emits a warning when the relative error exceeds ``tol``.
    """
    k7_66 = float(K7[6, 6])
    vlasov_ei = float(E_ref) * float(I_omega_E)
    ratio = abs(k7_66 - vlasov_ei) / max(abs(vlasov_ei), 1e-30)
    diag = {
        "K7_66": k7_66,
        "vlasov_EIomega": vlasov_ei,
        "ratio": ratio,
        "within_tol": bool(ratio < tol),
    }
    if ratio >= tol:
        msg = (
            f"K7[6,6]={k7_66:.4g} differs from Vlasov EI_omega={vlasov_ei:.4g} "
            f"by {ratio*100:.1f}% (tol={tol*100:.0f}%) at station {station_index}. "
            "Check unit-warping-strain BC alignment (I.6)."
        )
        if run_log is not None:
            try:
                run_log.warn_event(
                    "homogenisation.k7_warping_mismatch",
                    station=station_index,
                    **{k: float(v) if isinstance(v, (int, float)) else v for k, v in diag.items()},
                )
            except Exception:
                pass
        else:
            warnings.warn(msg, stacklevel=2)
    return diag
