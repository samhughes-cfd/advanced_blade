"""
Runnable MVP: one NACA section, unit resultants, shell handoff + CLPT Tsai–Wu FI.

Writes PNG diagnostics under ``outputs/`` next to this script (mesh, thin-wall
stress ribbons, CLPT ply figure, along-panel curves).

Run from repo root::

    python examples/section_shell_model/run_example.py

Or from ``examples``::

    python section_shell_model/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    examples = here.parent
    stress = examples / "section_stress_model"
    # Stress model first so ``from lib.*`` resolves to section_stress_model/lib.
    for p in (stress, examples):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main() -> None:
    _bootstrap_path()

    import numpy as np

    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.example_plots import (
        save_clpt_ply_figure,
        save_panel_along_contour_figure,
        save_shell_mesh_figure,
        save_thin_wall_stress_figures,
    )
    from section_shell_model.lib.local_clpt_shell import (
        default_skin_strengths_pa,
        solve_station_clpt_shell,
    )
    from section_shell_model.lib.recovery_adapter import run_section_with_shell_mapping

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "section_shell_demo"

    airfoil = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=120)
    spars = [0.35]

    bundle = run_section_with_shell_mapping(
        airfoil,
        spars,
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        B=0.0,
        dB_dx=0.0,
        reference_panel_index=0,
        reference_station_index=None,
    )

    panels = bundle.panels
    booms = bundle.booms
    webs_geom = bundle.webs_geom
    q_tot = bundle.q_tot
    sig_p = bundle.sig_p
    sig_b = bundle.sig_b

    ref = bundle.reference_resultants
    assert ref is not None
    p0 = panels[0]
    plies = p0.lam.build_plies()
    st = default_skin_strengths_pa()

    result = solve_station_clpt_shell(
        ref,
        plies,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
    )

    fi_max = float(np.max(result.fi_tsai_wu)) if len(result.fi_tsai_wu) else 0.0

    saved: list[Path] = []
    dpi = 150

    saved.append(
        save_shell_mesh_figure(
            out_dir / "mesh_shell_strips.png",
            panels,
            webs_geom,
            airfoil,
            spars,
            dpi=dpi,
        )
    )
    p_shear, p_axial = save_thin_wall_stress_figures(
        prefix,
        panels,
        booms,
        webs_geom,
        airfoil,
        spars,
        q_tot,
        sig_p,
        sig_b,
        dpi=dpi,
    )
    saved.extend([p_shear, p_axial])
    saved.append(
        save_clpt_ply_figure(
            out_dir / "clpt_ply_tsai_wu.png",
            panels,
            q_tot,
            sig_p,
            panel_index=0,
            station_index=None,
            strengths=st,
        )
    )
    saved.append(
        save_panel_along_contour_figure(
            out_dir / "reference_panel_q_sigma_vs_s.png",
            panels,
            q_tot,
            sig_p,
            panel_index=0,
            dpi=dpi,
        )
    )

    print("section_shell_model MVP — example run")
    print(f"  panel: {ref.panel_label}  station_index={ref.station_index}")
    print(f"  N_vec [Nx,Ny,Nxy] = {result.N_vec}")
    print(f"  M_vec [Mx,My,Mxy] = {result.M_vec}")
    print(f"  provenance Nx: {ref.provenance['Nx'].kind.value}")
    print(f"  max Tsai-Wu FI = {fi_max:.6e}")
    print(f"  shear centre (y_sc, z_sc) = ({bundle.y_sc:.5f}, {bundle.z_sc:.5f}) m")
    print(f"  I_omega = {bundle.I_omega:.4e}")
    print("  PNG outputs (dpi=%d):" % dpi)
    for p in saved:
        print(f"    {p}")


if __name__ == "__main__":
    main()
