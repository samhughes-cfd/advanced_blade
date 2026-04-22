"""
Adapter from :mod:`examples.section_stress_model.multi_cell_blade_section` to shell DTOs.

Entry points
------------
run_section_with_shell_mapping()
    MVP: thin-wall recovery → Nx, Nxy only; Ny/M placeholders.

run_section_with_mitc4_shell()
    Improved: thin-wall recovery → Nx, Nxy; E-weighted Vlasov warping correction
    applied to Nx; MITC4 panel solve → all 6 resultants with MITC4 provenance.

run_section_both()
    Calls run_section() once; returns (mvp_bundle, mitc4_bundle) sharing thin-wall result.

check_panel_equilibrium()
    Post-solve compatibility: compare Nx/Nxy at shared boundaries between consecutive panels.

Adds ``examples/section_stress_model`` to ``sys.path`` when importing the recovery module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .panel_mitc4_model import solve_panel_mitc4
from .global_mitc4_assembly import solve_global_coupled_mitc4
from .section_vlasov import SectionVlasovResult, compute_section_vlasov
from .types import FieldProvenance, ProvenanceKind, SectionShellRecoveryBundle, ShellPanelResultants


def _stress_model_root() -> Path:
    # lib/ -> section_shell_model/ -> examples/
    return Path(__file__).resolve().parents[2] / "section_stress_model"


def _ensure_stress_imports():
    root = _stress_model_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def run_section_with_shell_mapping(
    airfoil: np.ndarray,
    spars: list[float],
    *,
    skin_lam: Any | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    B: float = 0.0,
    dB_dx: float = 0.0,
    reference_panel_index: int = 0,
    reference_station_index: int | None = None,
) -> SectionShellRecoveryBundle:
    """
    Run closed-cell section recovery and build shell handoff for one panel station.

    Parameters
    ----------
    reference_panel_index, reference_station_index
        Panel and optional contour station index for :func:`panel_station_shell_resultants`.
    """
    _ensure_stress_imports()
    from multi_cell_blade_section import run_section  # type: ignore[import-untyped]

    out = run_section(
        airfoil,
        spars,
        skin_lam=skin_lam,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        B=B,
        dB_dx=dB_dx,
    )
    (
        panels,
        booms,
        webs_geom,
        q_tot,
        sig_p,
        sig_b,
        q0,
        props,
        y_sc,
        z_sc,
        areas,
        I_omega,
        gamma_y,
        gamma_z,
        GA_y,
        GA_z,
        q_primary,
        q_warp,
    ) = out

    ref = panel_station_shell_resultants(
        panels,
        q_tot,
        sig_p,
        panel_index=reference_panel_index,
        station_index=reference_station_index,
    )

    return SectionShellRecoveryBundle(
        panels=panels,
        booms=booms,
        webs_geom=webs_geom,
        q_tot=q_tot,
        sig_p=sig_p,
        sig_b=sig_b,
        q0=q0,
        props=props,
        y_sc=y_sc,
        z_sc=z_sc,
        areas=list(areas),
        I_omega=I_omega,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
        GA_y=GA_y,
        GA_z=GA_z,
        q_primary=q_primary,
        q_warp=q_warp,
        reference_resultants=ref,
    )


def panel_station_shell_resultants(
    panels: Any,
    q_tot: list,
    sig_p: list,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
) -> ShellPanelResultants:
    """
    Map thin-wall ``sigma_xx`` and ``q`` at one station to shell resultants.

    MVP:
    - ``Nx = sigma_xx * t``, ``Nxy = q`` (shear flow equals resultant shear per width).
    - ``Ny = Mx = My = Mxy = 0`` with :class:`ProvenanceKind.PLACEHOLDER`.
    - ``Qx``, ``Qy`` reserved (None).
    """
    p = panels[panel_index]
    npt = len(p.s)
    if npt < 2:
        raise ValueError("Panel has insufficient stations for shell mapping.")
    j = npt // 2 if station_index is None else int(station_index)
    j = max(0, min(npt - 1, j))

    sig_xx = float(sig_p[panel_index][j])
    q_here = float(q_tot[panel_index][j])
    t_wall = float(p.lam.t)
    tau_xy = q_here / max(t_wall, 1e-30)
    nx = sig_xx * t_wall
    nxy = q_here

    label = getattr(p, "label", "") or f"panel_{panel_index}"

    prov = {
        "Nx": FieldProvenance(ProvenanceKind.DERIVED, "sigma_xx * t from thin-wall recovery"),
        "Nxy": FieldProvenance(ProvenanceKind.DERIVED, "shear flow q [N/m]"),
        "Ny": FieldProvenance(
            ProvenanceKind.PLACEHOLDER,
            "MVP: Ny not recovered; thin-wall axial model uses sigma_yy ~ 0",
        ),
        "Mx": FieldProvenance(
            ProvenanceKind.PLACEHOLDER,
            "MVP: no bending moment from thickness direction",
        ),
        "My": FieldProvenance(
            ProvenanceKind.PLACEHOLDER,
            "MVP: no bending moment from thickness direction",
        ),
        "Mxy": FieldProvenance(
            ProvenanceKind.PLACEHOLDER,
            "MVP: twisting moment not recovered at laminate level",
        ),
        "Qx": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / higher-order future"),
        "Qy": FieldProvenance(ProvenanceKind.RESERVED, "FSDT / higher-order future"),
    }

    return ShellPanelResultants(
        Nx=nx,
        Ny=0.0,
        Nxy=nxy,
        Mx=0.0,
        My=0.0,
        Mxy=0.0,
        Qx=None,
        Qy=None,
        provenance=prov,
        sigma_xx_pa=sig_xx,
        tau_xy_pa=tau_xy,
        q_n_per_m=q_here,
        thickness_m=t_wall,
        panel_label=str(label),
        panel_index=panel_index,
        station_index=j,
    )


# ---------------------------------------------------------------------------
# Internal MITC4 solve helper (shared by run_section_with_mitc4_shell and run_section_both)
# ---------------------------------------------------------------------------

def _mitc4_solve_from_thin_wall(
    airfoil: np.ndarray,
    panels: list,
    webs_geom: Any,
    q_tot: list,
    sig_p: list,
    B: float,
    dB_dx: float,
    n_elements_per_panel: int,
    reference_panel_index: int,
) -> tuple[list[list[ShellPanelResultants]], list[dict], ShellPanelResultants | None, SectionVlasovResult]:
    """
    Run Vlasov + MITC4 solve from unpacked run_section() output.

    Returns (all_panel_results, all_panel_diagnostics, reference_resultants, vlasov).
    """
    _ensure_stress_imports()
    from lib.laminate_clpt import abd_stack  # type: ignore[import-untyped]

    # Vlasov warping correction (pass webs_geom for multi-cell Batho correction)
    t_web_mean = float(np.mean([p.lam.t for p in panels if len(p.s) >= 2])) if panels else None
    vlasov = compute_section_vlasov(
        airfoil, panels, B, dB_dx,
        webs_geom=list(webs_geom),
        t_web=t_web_mean,
    )

    # Per-panel MITC4 solve
    all_panel_results: list[list[ShellPanelResultants]] = []
    all_panel_diag: list[dict] = []
    for pi, p in enumerate(panels):
        if len(p.s) < 2:
            all_panel_results.append([])
            all_panel_diag.append({})
            continue

        # Build ABD from ply stack
        plies = p.lam.build_plies()
        A_mat, B_mat, D_mat = abd_stack(plies)
        ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])

        thickness = float(p.lam.t)
        nu = float(p.lam.nu)
        E_p = float(p.lam.E)
        G_eff = E_p / (2.0 * (1.0 + nu))

        # Nx: thin-wall axial + Vlasov warping correction at each station
        sig_xx_arr = np.asarray(sig_p[pi], dtype=float)
        is_web = "web" in str(getattr(p, "label", "")).lower()
        if is_web:
            # Web panels are not parameterised on the outer skin outline;
            # do not add any interpolated warping correction.
            sig_w_nodes = np.zeros_like(sig_xx_arr)
        elif pi < len(vlasov.sigma_omega) and len(vlasov.sigma_omega[pi]) > 0:
            s_mids = vlasov.panel_s_mids[pi]
            sig_w_mids = vlasov.sigma_omega[pi]
            if len(s_mids) > 0 and len(p.s) >= 2:
                sig_w_nodes = np.interp(p.s, np.r_[0.0, s_mids, p.s[-1]],
                                        np.r_[sig_w_mids[0], sig_w_mids, sig_w_mids[-1]])
            else:
                sig_w_nodes = np.zeros_like(sig_xx_arr)
        else:
            sig_w_nodes = np.zeros_like(sig_xx_arr)

        sig_total = sig_xx_arr + sig_w_nodes
        Nx_panel = sig_total * thickness
        Nxy_panel = np.asarray(q_tot[pi], dtype=float)

        # Spar BCs: panel boundaries (s=0 and s=s_max)
        spar_s = [0.0, float(p.s[-1])]

        label = getattr(p, "label", "") or f"panel_{pi}"
        nodes_yz = np.asarray(p.nodes) if hasattr(p, "nodes") and p.nodes is not None else None
        solve_out = solve_panel_mitc4(
            ABD=ABD,
            thickness=thickness,
            G_eff=G_eff,
            s_panel=p.s,
            Nx_panel=Nx_panel,
            Nxy_panel=Nxy_panel,
            spar_s_coords=spar_s,
            panel_label=str(label),
            panel_index=pi,
            nodes_yz=nodes_yz,
            n_elements=n_elements_per_panel,
            return_diagnostics=True,
        )
        results, diag = solve_out
        all_panel_results.append(results)
        all_panel_diag.append(diag)

    # Reference resultant: centre element of reference panel
    ref: ShellPanelResultants | None = None
    if reference_panel_index < len(all_panel_results):
        panel_res = all_panel_results[reference_panel_index]
        if panel_res:
            ref = panel_res[len(panel_res) // 2]

    return all_panel_results, all_panel_diag, ref, vlasov


def _mitc4_global_coupled_solve_from_thin_wall(
    airfoil: np.ndarray,
    panels: list,
    webs_geom: Any,
    q_tot: list,
    sig_p: list,
    B: float,
    dB_dx: float,
    n_elements_per_panel: int,
    reference_panel_index: int,
    global_bc_mode: str = "legacy",
    interface_constraint_mode: str = "shared",
) -> tuple[list[list[ShellPanelResultants]], list[dict], ShellPanelResultants | None, SectionVlasovResult]:
    """
    Global-coupled MITC4 solve with shared interface nodes in one system.
    """
    t_web_mean = float(np.mean([p.lam.t for p in panels if len(p.s) >= 2])) if panels else None
    vlasov = compute_section_vlasov(
        airfoil, panels, B, dB_dx,
        webs_geom=list(webs_geom),
        t_web=t_web_mean,
    )

    Nx_panels: list[np.ndarray] = []
    Nxy_panels: list[np.ndarray] = []
    for pi, p in enumerate(panels):
        if len(p.s) < 2:
            Nx_panels.append(np.array([]))
            Nxy_panels.append(np.array([]))
            continue
        thickness = float(p.lam.t)
        sig_xx_arr = np.asarray(sig_p[pi], dtype=float)
        is_web = "web" in str(getattr(p, "label", "")).lower()
        if is_web:
            sig_w_nodes = np.zeros_like(sig_xx_arr)
        elif pi < len(vlasov.sigma_omega) and len(vlasov.sigma_omega[pi]) > 0:
            s_mids = vlasov.panel_s_mids[pi]
            sig_w_mids = vlasov.sigma_omega[pi]
            if len(s_mids) > 0 and len(p.s) >= 2:
                sig_w_nodes = np.interp(
                    p.s, np.r_[0.0, s_mids, p.s[-1]], np.r_[sig_w_mids[0], sig_w_mids, sig_w_mids[-1]]
                )
            else:
                sig_w_nodes = np.zeros_like(sig_xx_arr)
        else:
            sig_w_nodes = np.zeros_like(sig_xx_arr)
        sig_total = sig_xx_arr + sig_w_nodes
        Nx_panels.append(sig_total * thickness)
        Nxy_panels.append(np.asarray(q_tot[pi], dtype=float))

    all_panel_results, all_panel_diag = solve_global_coupled_mitc4(
        panels,
        Nx_panels,
        Nxy_panels,
        n_elements_per_panel=n_elements_per_panel,
        bc_mode=global_bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )

    ref: ShellPanelResultants | None = None
    if reference_panel_index < len(all_panel_results) and all_panel_results[reference_panel_index]:
        panel_res = all_panel_results[reference_panel_index]
        ref = panel_res[len(panel_res) // 2]
    return all_panel_results, all_panel_diag, ref, vlasov


# ---------------------------------------------------------------------------
# MITC4 entry point
# ---------------------------------------------------------------------------

def run_section_with_mitc4_shell(
    airfoil: np.ndarray,
    spars: list[float],
    *,
    skin_lam: Any | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    B: float = 0.0,
    dB_dx: float = 0.0,
    reference_panel_index: int = 0,
    n_elements_per_panel: int = 10,
    use_global_coupled: bool = True,
    global_bc_mode: str = "legacy",
    interface_constraint_mode: str = "shared",
) -> SectionShellRecoveryBundle:
    """
    Run thin-wall recovery + Vlasov warping + MITC4 panel solve.

    Steps
    -----
    1. ``run_section`` (Bredt thin-wall) → sigma_xx, q per panel station.
    2. ``compute_section_vlasov`` → E-weighted ω̂, I_ω_E, warping-corrected sigma_xx.
       Web panels receive zero warping correction (their sigma_omega is not defined
       on the open skin outline).
    3. For each panel: ``solve_panel_mitc4`` → all 6 shell resultants (no placeholders).
    4. The reference panel resultants carry ``ProvenanceKind.MITC4`` on all fields.

    Parameters mirror ``run_section_with_shell_mapping``; add ``n_elements_per_panel``
    to control MITC4 mesh density along each panel contour.
    """
    _ensure_stress_imports()
    from multi_cell_blade_section import run_section  # type: ignore[import-untyped]

    out = run_section(
        airfoil,
        spars,
        skin_lam=skin_lam,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        B=B,
        dB_dx=dB_dx,
    )
    (
        panels, booms, webs_geom, q_tot, sig_p, sig_b, q0, props,
        y_sc, z_sc, areas, I_omega, gamma_y, gamma_z, GA_y, GA_z,
        q_primary, q_warp,
    ) = out

    if use_global_coupled:
        all_panel_results, all_panel_diag, ref, vlasov = _mitc4_global_coupled_solve_from_thin_wall(
            airfoil, panels, webs_geom, q_tot, sig_p, B, dB_dx, n_elements_per_panel, reference_panel_index,
            global_bc_mode, interface_constraint_mode
        )
    else:
        all_panel_results, all_panel_diag, ref, vlasov = _mitc4_solve_from_thin_wall(
            airfoil, panels, webs_geom, q_tot, sig_p, B, dB_dx, n_elements_per_panel, reference_panel_index
        )

    return SectionShellRecoveryBundle(
        panels=panels,
        booms=booms,
        webs_geom=webs_geom,
        q_tot=q_tot,
        sig_p=sig_p,
        sig_b=sig_b,
        q0=q0,
        props=props,
        y_sc=vlasov.y_sc,
        z_sc=vlasov.z_sc,
        areas=list(areas),
        I_omega=vlasov.I_omega_E,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
        GA_y=GA_y,
        GA_z=GA_z,
        q_primary=q_primary,
        q_warp=q_warp,
        reference_resultants=ref,
        all_panel_mitc4_results=all_panel_results,
        all_panel_mitc4_diagnostics=all_panel_diag,
        vlasov_result=vlasov,
    )


# ---------------------------------------------------------------------------
# Combined entry point (avoids duplicate run_section() call)
# ---------------------------------------------------------------------------

def run_section_both(
    airfoil: np.ndarray,
    spars: list[float],
    *,
    skin_lam: Any | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    B: float = 0.0,
    dB_dx: float = 0.0,
    reference_panel_index: int = 0,
    reference_station_index: int | None = None,
    n_elements_per_panel: int = 10,
    use_global_coupled: bool = True,
    global_bc_mode: str = "legacy",
    interface_constraint_mode: str = "shared",
) -> tuple[SectionShellRecoveryBundle, SectionShellRecoveryBundle]:
    """
    Call ``run_section()`` once and return both MVP and MITC4 bundles.

    Returns
    -------
    (mvp_bundle, mitc4_bundle)
    """
    _ensure_stress_imports()
    from multi_cell_blade_section import run_section  # type: ignore[import-untyped]

    out = run_section(
        airfoil,
        spars,
        skin_lam=skin_lam,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        B=B,
        dB_dx=dB_dx,
    )
    (
        panels, booms, webs_geom, q_tot, sig_p, sig_b, q0, props,
        y_sc, z_sc, areas, I_omega, gamma_y, gamma_z, GA_y, GA_z,
        q_primary, q_warp,
    ) = out

    # MVP bundle
    ref_mvp = panel_station_shell_resultants(
        panels, q_tot, sig_p,
        panel_index=reference_panel_index,
        station_index=reference_station_index,
    )
    mvp_bundle = SectionShellRecoveryBundle(
        panels=panels,
        booms=booms,
        webs_geom=webs_geom,
        q_tot=q_tot,
        sig_p=sig_p,
        sig_b=sig_b,
        q0=q0,
        props=props,
        y_sc=y_sc,
        z_sc=z_sc,
        areas=list(areas),
        I_omega=I_omega,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
        GA_y=GA_y,
        GA_z=GA_z,
        q_primary=q_primary,
        q_warp=q_warp,
        reference_resultants=ref_mvp,
    )

    # MITC4 bundle
    if use_global_coupled:
        all_panel_results, all_panel_diag, ref_mitc4, vlasov = _mitc4_global_coupled_solve_from_thin_wall(
            airfoil, panels, webs_geom, q_tot, sig_p, B, dB_dx, n_elements_per_panel, reference_panel_index,
            global_bc_mode, interface_constraint_mode
        )
    else:
        all_panel_results, all_panel_diag, ref_mitc4, vlasov = _mitc4_solve_from_thin_wall(
            airfoil, panels, webs_geom, q_tot, sig_p, B, dB_dx, n_elements_per_panel, reference_panel_index
        )
    mitc4_bundle = SectionShellRecoveryBundle(
        panels=panels,
        booms=booms,
        webs_geom=webs_geom,
        q_tot=q_tot,
        sig_p=sig_p,
        sig_b=sig_b,
        q0=q0,
        props=props,
        y_sc=vlasov.y_sc,
        z_sc=vlasov.z_sc,
        areas=list(areas),
        I_omega=vlasov.I_omega_E,
        gamma_y=gamma_y,
        gamma_z=gamma_z,
        GA_y=GA_y,
        GA_z=GA_z,
        q_primary=q_primary,
        q_warp=q_warp,
        reference_resultants=ref_mitc4,
        all_panel_mitc4_results=all_panel_results,
        all_panel_mitc4_diagnostics=all_panel_diag,
        vlasov_result=vlasov,
    )

    return mvp_bundle, mitc4_bundle


# ---------------------------------------------------------------------------
# Panel-to-panel equilibrium check
# ---------------------------------------------------------------------------

def _panel_endpoint(panel: Any, which: str) -> np.ndarray:
    nodes = np.asarray(getattr(panel, "nodes", []), dtype=float)
    if len(nodes) == 0:
        return np.array([np.nan, np.nan], dtype=float)
    return nodes[0] if which == "start" else nodes[-1]


def _panel_end_tangent(panel: Any, which: str) -> np.ndarray:
    nodes = np.asarray(getattr(panel, "nodes", []), dtype=float)
    if len(nodes) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if which == "start":
        t = nodes[1] - nodes[0]
    else:
        t = nodes[-1] - nodes[-2]
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return t / n


def _classify_boundary(label_i: str, label_j: str) -> str:
    is_web_i = "web" in label_i.lower()
    is_web_j = "web" in label_j.lower()
    if is_web_i and is_web_j:
        return "web-web"
    if is_web_i or is_web_j:
        return "skin-web"
    return "skin-skin"


def _thresholds_for_boundary(boundary_type: str) -> tuple[float, float]:
    if boundary_type == "skin-skin":
        return 0.05, 0.05
    if boundary_type == "skin-web":
        return 0.10, 0.10
    return 0.10, 0.10


def _secondary_thresholds_for_boundary(boundary_type: str) -> tuple[float, float]:
    if boundary_type == "skin-skin":
        return 0.10, 0.10
    if boundary_type == "skin-web":
        return 0.15, 0.15
    return 0.15, 0.15


def _diag_boundary_force(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    br = diag.get("boundary_reaction_set", diag.get("boundary_reaction", {}))
    side = br.get(which, {})
    return float(side.get("Fx", 0.0)), float(side.get("Fs", 0.0))


def _diag_boundary_field(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    fs = diag.get("interface_field_set", {})
    side = fs.get(which, {})
    return float(side.get("Nx", 0.0)), float(side.get("Nxy", 0.0))


def _diag_boundary_edge_traction(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    es = diag.get("interface_edge_set", {})
    side = es.get(which, {})
    return float(side.get("Tx_int", 0.0)), float(side.get("Ts_int", 0.0))


def _is_web_label(label: str) -> bool:
    return "web" in label.lower()


def _interface_traction_residuals(
    tx_i: float,
    ts_i: float,
    tx_j: float,
    ts_j: float,
    t_i: np.ndarray,
    t_j: np.ndarray,
) -> dict[str, float]:
    """
    Compute physically correct interface traction residuals.

    Each panel's local frame has X = span (common to all panels) and ŝ = contour
    tangent (panel-specific, lives in the (Y,Z) cross-section plane).

    ``mitc4_edge_shear_traction_integrated`` already applies the outward-normal sign
    (normal_sign = ±1 for start/end edges), so:

      tx_* = Nxy * normal_sign  — spanwise (X) traction component
      ts_* = Ny  * normal_sign  — contour (ŝ) traction component

    Interface equilibrium (Newton's 3rd law, sum = 0 over all panels at junction):
      X-channel : Tx_i + Tx_j = 0   (X is shared — no rotation needed)
      YZ-channel: Ts_i * ŝ_i + Ts_j * ŝ_j = 0  (vector sum in (Y,Z))

    Parameters
    ----------
    tx_i, tx_j : spanwise tractions (already outward-normal signed)
    ts_i, ts_j : contour tractions (already outward-normal signed)
    t_i, t_j   : 2-D (Y,Z) unit contour tangent vectors at the junction endpoint
    """
    dTx = float(tx_i) + float(tx_j)
    t_i_arr = np.asarray(t_i, dtype=float)
    t_j_arr = np.asarray(t_j, dtype=float)
    dT_yz = float(ts_i) * t_i_arr + float(ts_j) * t_j_arr
    dT_yz_mag = float(np.linalg.norm(dT_yz))
    return {
        "Tx_i": float(tx_i),
        "Tx_j": float(tx_j),
        "Ts_i": float(ts_i),
        "Ts_j": float(ts_j),
        "dTx": dTx,
        "dT_yz_y": float(dT_yz[0]),
        "dT_yz_z": float(dT_yz[1]) if len(dT_yz) > 1 else 0.0,
        "dT_yz_mag": dT_yz_mag,
    }


def _secondary_traction_thresholds(boundary_type: str) -> tuple[float, float]:
    """Return (tol_dTx, tol_dT_yz) acceptance thresholds for the traction-vector check."""
    if boundary_type == "skin-skin":
        return 0.05, 0.10
    if boundary_type == "skin-web":
        return 0.10, 0.15
    return 0.10, 0.15


def _build_geometric_interfaces(panels: list, tol: float = 1e-6) -> list[dict]:
    """
    Build interface pairs from geometric endpoint proximity.

    Returns list of dicts: {pi, pj, end_i, end_j, point}.
    """
    endpoints: list[dict] = []
    for pi, p in enumerate(panels):
        for which in ("start", "end"):
            pt = _panel_endpoint(p, which)
            if np.any(~np.isfinite(pt)):
                continue
            endpoints.append({"pi": pi, "end": which, "pt": pt})

    # Cluster coincident endpoints, then generate deterministic pair graph
    # (supports >2-way junctions).
    clusters: list[list[dict]] = []
    for ep in endpoints:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(ep["pt"] - c[0]["pt"])) <= tol:
                c.append(ep)
                placed = True
                break
        if not placed:
            clusters.append([ep])

    interfaces: list[dict] = []
    seen_pairs: set[tuple[int, str, int, str]] = set()
    for c in clusters:
        if len(c) < 2:
            continue
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                ei, ej = c[i], c[j]
                if ei["pi"] == ej["pi"]:
                    continue
                key = (int(ei["pi"]), str(ei["end"]), int(ej["pi"]), str(ej["end"]))
                key_rev = (key[2], key[3], key[0], key[1])
                if key in seen_pairs or key_rev in seen_pairs:
                    continue
                seen_pairs.add(key)
                interfaces.append(
                    {
                        "pi": int(ei["pi"]),
                        "pj": int(ej["pi"]),
                        "end_i": str(ei["end"]),
                        "end_j": str(ej["end"]),
                        "point": 0.5 * (ei["pt"] + ej["pt"]),
                    }
                )
    interfaces.sort(key=lambda d: (d["pi"], d["pj"], d["end_i"], d["end_j"]))
    return interfaces


def check_panel_equilibrium(
    all_panel_mitc4_results: list[list[ShellPanelResultants]],
    panels: list,
    *,
    all_panel_mitc4_diagnostics: list[dict] | None = None,
    endpoint_tol: float = 1e-6,
) -> list[dict]:
    """
    Compare Nx and Nxy at geometrically shared panel boundaries.

    Interface pairs are detected from endpoint proximity rather than index adjacency.
    For each pair, compares resultants at corresponding interface ends.

    Returns
    -------
    list of dicts with dual metrics:
    - reaction-based interface mismatch (authoritative)
    - resultant-based mismatch (secondary diagnostic)
    """
    results: list[dict] = []
    interfaces = _build_geometric_interfaces(panels, tol=endpoint_tol)
    # Build a common traction-frame tangent per geometric junction cluster.
    cluster_tangent: dict[tuple[int, ...], np.ndarray] = {}
    endpoint_sets: list[set[tuple[int, str]]] = []
    for iface in interfaces:
        a = (int(iface["pi"]), str(iface["end_i"]))
        b = (int(iface["pj"]), str(iface["end_j"]))
        merged = False
        for s in endpoint_sets:
            if a in s or b in s:
                s.add(a)
                s.add(b)
                merged = True
                break
        if not merged:
            endpoint_sets.append({a, b})
    for s in endpoint_sets:
        key = tuple(sorted(hash(x) for x in s))
        t_mats: list[np.ndarray] = []
        for pi, end in s:
            t = _panel_end_tangent(panels[pi], end)
            t_mats.append(np.outer(t, t))
        if not t_mats:
            cluster_tangent[key] = np.array([1.0, 0.0], dtype=float)
            continue
        M = np.sum(np.stack(t_mats, axis=0), axis=0)
        evals, evecs = np.linalg.eigh(M)
        t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
        if float(np.linalg.norm(t_ref)) < 1e-12:
            t_ref = np.array([1.0, 0.0], dtype=float)
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        cluster_tangent[key] = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)

    def _cluster_key(a: tuple[int, str], b: tuple[int, str]) -> tuple[int, ...]:
        for s in endpoint_sets:
            if a in s and b in s:
                return tuple(sorted(hash(x) for x in s))
        return tuple(sorted((hash(a), hash(b))))

    for iface in interfaces:
        pi = int(iface["pi"])
        pj = int(iface["pj"])
        end_i = str(iface["end_i"])
        end_j = str(iface["end_j"])

        if pi >= len(all_panel_mitc4_results) or pj >= len(all_panel_mitc4_results):
            continue
        res_i = all_panel_mitc4_results[pi]
        res_j = all_panel_mitc4_results[pj]
        if not res_i or not res_j:
            continue

        r_i = res_i[0] if end_i == "start" else res_i[-1]
        r_j = res_j[0] if end_j == "start" else res_j[-1]

        label_i = getattr(panels[pi], "label", None) or f"panel_{pi}"
        label_j = getattr(panels[pj], "label", None) or f"panel_{pj}"

        t_i = _panel_end_tangent(panels[pi], end_i)
        t_j = _panel_end_tangent(panels[pj], end_j)
        key = _cluster_key((pi, end_i), (pj, end_j))
        t_ref = cluster_tangent.get(key, t_i)
        sign_i = 1.0 if float(np.dot(t_i, t_ref)) >= 0.0 else -1.0
        sign_j = 1.0 if float(np.dot(t_j, t_ref)) >= 0.0 else -1.0
        orient = "same" if sign_i == sign_j else "opposite"
        nxy_sign = sign_i / sign_j

        Nx_a = float(r_i.Nx)
        Nx_b = float(r_j.Nx)
        Nxy_a = float(r_i.Nxy)
        Nxy_b = float(r_j.Nxy)

        Nx_a_norm = Nx_a
        Nx_b_norm = Nx_b
        Nxy_a_norm = Nxy_a
        Nxy_b_norm = nxy_sign * Nxy_b

        dNx_abs = abs(Nx_a_norm - Nx_b_norm)
        dNxy_abs = abs(Nxy_a_norm - Nxy_b_norm)
        scale_Nx = max(abs(r_i.Nx), abs(r_j.Nx), 1.0)
        scale_Nxy = max(abs(r_i.Nxy), abs(r_j.Nxy), 1.0)

        dNx_rel_resultant = dNx_abs / scale_Nx
        dNxy_rel_resultant = dNxy_abs / scale_Nxy
        boundary_type = _classify_boundary(str(label_i), str(label_j))
        tol_react_nx, tol_react_nxy = _thresholds_for_boundary(boundary_type)
        tol_res_nx, tol_res_nxy = _secondary_thresholds_for_boundary(boundary_type)

        diag_i = None
        diag_j = None
        if all_panel_mitc4_diagnostics is not None:
            if pi < len(all_panel_mitc4_diagnostics):
                diag_i = all_panel_mitc4_diagnostics[pi]
            if pj < len(all_panel_mitc4_diagnostics):
                diag_j = all_panel_mitc4_diagnostics[pj]
        Fx_i, Fs_i = _diag_boundary_force(diag_i, end_i)
        Fx_j, Fs_j = _diag_boundary_force(diag_j, end_j)
        Fx_j_norm = Fx_j
        Fs_j_norm = nxy_sign * Fs_j
        dFx_rel = abs(Fx_i - Fx_j_norm) / max(abs(Fx_i), abs(Fx_j_norm), 1.0)
        dFs_rel = abs(Fs_i - Fs_j_norm) / max(abs(Fs_i), abs(Fs_j_norm), 1.0)
        pass_react_nx = dFx_rel <= tol_react_nx
        pass_react_nxy = dFs_rel <= tol_react_nxy

        # Secondary field continuity.
        # Nx (spanwise): directly comparable across panels — X is common to all.
        # Nxy/traction: use geometry-based traction-vector residuals when edge data
        # available; fall back to field-set Nxy (orientation-normalised) or centroid.
        fNx_i, fNxy_i = _diag_boundary_field(diag_i, end_i)
        fNx_j, fNxy_j = _diag_boundary_field(diag_j, end_j)
        have_field = (diag_i is not None and diag_j is not None and
                      "interface_field_set" in diag_i and "interface_field_set" in diag_j)
        have_edge = (diag_i is not None and diag_j is not None and
                     "interface_edge_set" in diag_i and "interface_edge_set" in diag_j)

        # Nx: spanwise component, no rotation needed.
        if have_field or have_edge:
            dNx_rel_field = abs(fNx_i - fNx_j) / max(abs(fNx_i), abs(fNx_j), 1.0)
        else:
            dNx_rel_field = dNx_rel_resultant

        trac_res: dict[str, float]
        if have_edge:
            # Geometry-only traction vector residuals (see _interface_traction_residuals).
            tx_i_val, ts_i_val = _diag_boundary_edge_traction(diag_i, end_i)
            tx_j_val, ts_j_val = _diag_boundary_edge_traction(diag_j, end_j)
            trac_res = _interface_traction_residuals(tx_i_val, ts_i_val, tx_j_val, ts_j_val, t_i, t_j)
            scale_Tx = max(abs(trac_res["Tx_i"]), abs(trac_res["Tx_j"]), 1.0)
            scale_Ts = max(abs(trac_res["Ts_i"]), abs(trac_res["Ts_j"]), 1.0)
            dTx_rel = abs(trac_res["dTx"]) / scale_Tx
            dT_yz_rel = trac_res["dT_yz_mag"] / scale_Ts
            nxy_source = "traction-vector-strict"
        elif have_field:
            # Field-set Nxy: orientation-normalised comparison.
            fNxy_j_norm = nxy_sign * fNxy_j
            scale_fNxy = max(abs(fNxy_i), abs(fNxy_j_norm), 1.0)
            dTx_rel = abs(fNxy_i - fNxy_j_norm) / scale_fNxy
            dT_yz_rel = 0.0
            nxy_source = "field-fallback"
            trac_res = {
                "Tx_i": float(fNxy_i), "Tx_j": float(fNxy_j),
                "Ts_i": 0.0, "Ts_j": 0.0,
                "dTx": float(fNxy_i - fNxy_j_norm),
                "dT_yz_y": 0.0, "dT_yz_z": 0.0, "dT_yz_mag": 0.0,
            }
        else:
            dTx_rel = dNxy_rel_resultant
            dT_yz_rel = 0.0
            nxy_source = "centroid-fallback"
            trac_res = {
                "Tx_i": float(Nxy_a), "Tx_j": float(Nxy_b),
                "Ts_i": 0.0, "Ts_j": 0.0,
                "dTx": float(Nxy_a - Nxy_b_norm),
                "dT_yz_y": 0.0, "dT_yz_z": 0.0, "dT_yz_mag": 0.0,
            }

        dNxy_rel_field = dTx_rel
        tol_res_tx, tol_res_tyz = _secondary_traction_thresholds(boundary_type)
        pass_res_nx = dNx_rel_field <= tol_res_nx
        pass_res_nxy = dTx_rel <= tol_res_tx
        pass_dT_yz = dT_yz_rel <= tol_res_tyz

        results.append({
            "pi": pi,
            "pj": pj,
            "end_i": end_i,
            "end_j": end_j,
            "label_i": str(label_i),
            "label_j": str(label_j),
            "orientation": orient,
            "boundary_type": boundary_type,
            "Nx_a": Nx_a,
            "Nx_b": Nx_b,
            "Nxy_a": Nxy_a,
            "Nxy_b": Nxy_b,
            "Nx_a_norm": Nx_a_norm,
            "Nx_b_norm": Nx_b_norm,
            "Nxy_a_norm": Nxy_a_norm,
            "Nxy_b_norm": Nxy_b_norm,
            "reaction_Nx_i": Fx_i,
            "reaction_Nx_j": Fx_j,
            "reaction_Nxy_i": Fs_i,
            "reaction_Nxy_j": Fs_j,
            "reaction_dNx_rel": dFx_rel,
            "reaction_dNxy_rel": dFs_rel,
            "reaction_tol_nx": tol_react_nx,
            "reaction_tol_nxy": tol_react_nxy,
            "reaction_pass_nx": pass_react_nx,
            "reaction_pass_nxy": pass_react_nxy,
            "reaction_pass": bool(pass_react_nx and pass_react_nxy),
            "field_Nx_i": fNx_i,
            "field_Nx_j": fNx_j,
            "field_Nxy_i": fNxy_i,
            "field_Nxy_j": fNxy_j,
            "resultant_dNx_rel": dNx_rel_field,
            "resultant_dNxy_rel": dNxy_rel_field,
            "resultant_nxy_source": nxy_source,
            "nxy_compare_mode": nxy_source,
            # Physics-correct traction-vector continuity (geometry-based).
            "dTx_rel": dTx_rel,
            "dT_yz_rel": dT_yz_rel,
            "dT_yz_mag": trac_res["dT_yz_mag"],
            # Backward-compat aliases for run_example display (Phase D will rename columns).
            "t_t_i": trac_res["Tx_i"],
            "t_t_j": trac_res["Tx_j"],
            "t_n_i": trac_res["Ts_i"],
            "t_n_j": trac_res["Ts_j"],
            "resultant_dTn_rel": dT_yz_rel,
            "resultant_dNx_rel_centroid": dNx_rel_resultant,
            "resultant_dNxy_rel_centroid": dNxy_rel_resultant,
            "resultant_tol_nx": tol_res_nx,
            "resultant_tol_nxy": tol_res_nxy,
            "resultant_tol_dTx": tol_res_tx,
            "resultant_tol_dT_yz": tol_res_tyz,
            "resultant_pass_nx": pass_res_nx,
            "resultant_pass_nxy": pass_res_nxy,
            "resultant_pass_dT_yz": pass_dT_yz,
            "resultant_pass": bool(pass_res_nx and pass_res_nxy),
        })
    return results


def build_load_reaction_audit(all_panel_mitc4_diagnostics: list[dict] | None) -> dict[str, float]:
    """
    Summarize panel/global load-reaction consistency from diagnostics.
    """
    if not all_panel_mitc4_diagnostics:
        return {"n_panels": 0.0, "max_rel_mismatch": 0.0, "mean_rel_mismatch": 0.0}
    rel_vals: list[float] = []
    for di in all_panel_mitc4_diagnostics:
        if not di:
            continue
        lt = di.get("load_totals", {})
        br = di.get("boundary_reaction_set", {})
        if "start" not in br or "end" not in br:
            continue
        fx = float(lt.get("Fx_total", 0.0))
        fs = float(lt.get("Fs_total", 0.0))
        rx = float(br["start"].get("Fx", 0.0) + br["end"].get("Fx", 0.0))
        rs = float(br["start"].get("Fs", 0.0) + br["end"].get("Fs", 0.0))
        ex = abs(rx + fx) / max(abs(rx), abs(fx), 1.0)
        es = abs(rs + fs) / max(abs(rs), abs(fs), 1.0)
        rel_vals.append(max(ex, es))
    if not rel_vals:
        return {"n_panels": 0.0, "max_rel_mismatch": 0.0, "mean_rel_mismatch": 0.0}
    return {
        "n_panels": float(len(rel_vals)),
        "max_rel_mismatch": float(max(rel_vals)),
        "mean_rel_mismatch": float(np.mean(rel_vals)),
    }
