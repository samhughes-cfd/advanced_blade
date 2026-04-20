"""
Adapter from :mod:`examples.section_stress_model.multi_cell_blade_section` to shell DTOs.

Adds ``examples/section_stress_model`` to ``sys.path`` when importing the recovery module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

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
