"""
Precompute job diagnostics: PNG bundle matching ``run_example`` MVP+MITC4 plots (no mesh sweeps).

Writes station-tagged filenames under ``out_dir`` (typically ``<job>/section_shell_model/``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_section_shell_sys_path() -> None:
    """Match ``run_example._bootstrap_path`` so ``multi_cell_blade_section`` resolves."""
    repo = _repo_root()
    examples = repo / "examples"
    stress = examples / "section_stress_model"
    for p in (repo, stress, examples):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def naca_digits_to_params(naca_m: float, naca_p: float, naca_xx: float) -> tuple[float, float, float]:
    """
    Map spanwise NACA digits (same convention as ``orchestration.precompute.stages.naca4``)
    to ``naca_four_digit`` fractions: m, p, t_c.
    """
    mi = int(np.clip(int(round(float(naca_m))), 0, 9))
    pi = int(np.clip(int(round(float(naca_p))), 0, 9))
    xxi = int(np.clip(int(round(float(naca_xx))), 0, 99))
    m = float(mi) / 100.0
    p = float(pi) / 10.0
    t_c = float(xxi) / 100.0
    return m, p, t_c


def build_airfoil_for_station(
    naca_m: float,
    naca_p: float,
    naca_xx: float,
    chord_m: float,
    *,
    n_points: int = 120,
) -> NDArray[np.float64]:
    """Unit-chord NACA polyline scaled to ``chord_m`` (metres)."""
    ensure_section_shell_sys_path()
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]

    m, p, t_c = naca_digits_to_params(naca_m, naca_p, naca_xx)
    airfoil = np.asarray(naca_four_digit(m=m, p=p, t_c=t_c, n=n_points), dtype=np.float64)
    c = float(max(chord_m, 1e-9))
    return airfoil * c


def write_section_shell_model_station_outputs(
    out_dir: Path,
    *,
    airfoil: NDArray[np.float64],
    spars: list[float],
    station_tag: str,
    n_elements_per_panel: int = 12,
    dpi: int = 150,
    merge_nose: bool = False,
) -> list[Path]:
    """
    Single-pass shell diagnostics (unit section resultants). No mesh / airfoil refinement sweeps.

    Filenames include ``station_tag`` (e.g. ``i000_rz0.000``) to avoid collisions across stations.
    """
    ensure_section_shell_sys_path()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_shell_model.lib.example_plots import (
        save_clpt_fi_on_section_geometry,
        save_clpt_ply_figure,
        save_mitc4_fi_figure,
        save_mitc4_resultants_figure,
        save_panel_along_contour_figure,
        save_shell_mesh_figure,
        save_thin_wall_stress_figures,
    )
    from blade_precompute.section_shell_model.lib.local_clpt_shell import (
        default_skin_strengths_pa,
        solve_station_clpt_shell,
        sweep_panel_clpt_fi,
    )
    from blade_precompute.section_shell_model.lib.recovery_adapter import run_section_both

    bundle, mitc4_bundle = run_section_both(
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
        n_elements_per_panel=int(n_elements_per_panel),
        merge_nose=merge_nose,
    )

    panels = bundle.panels
    booms = bundle.booms
    webs_geom = bundle.webs_geom
    q_tot = bundle.q_tot
    sig_p = bundle.sig_p
    sig_b = bundle.sig_b

    ref = bundle.reference_resultants
    if ref is None:
        raise RuntimeError("section_shell_model: missing reference_resultants from thin-wall bundle.")
    p0 = panels[0]
    plies = p0.lam.build_plies()
    st = default_skin_strengths_pa()

    solve_station_clpt_shell(
        ref,
        plies,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
    )

    saved: list[Path] = []
    suf = f"_{station_tag}" if station_tag else ""

    saved.append(
        save_shell_mesh_figure(
            out_dir / f"mesh_shell_strips{suf}.png",
            panels,
            webs_geom,
            airfoil,
            spars,
            dpi=dpi,
        )
    )
    prefix = out_dir / f"section_shell_demo{suf}"
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
            out_dir / f"clpt_ply_hashin{suf}.png",
            panels,
            q_tot,
            sig_p,
            panel_index=0,
            station_index=None,
            strengths=st,
        )
    )

    vlasov = mitc4_bundle.vlasov_result
    sig_omega_p0 = vlasov.sigma_omega[0] if vlasov and vlasov.sigma_omega else None

    saved.append(
        save_panel_along_contour_figure(
            out_dir / f"reference_panel_q_sigma_vs_s{suf}.png",
            panels,
            q_tot,
            sig_p,
            panel_index=0,
            sigma_omega_mids=sig_omega_p0,
            dpi=dpi,
        )
    )

    all_panel_mitc4 = mitc4_bundle.all_panel_mitc4_results or []
    if all_panel_mitc4 and all_panel_mitc4[0]:
        p0_m = mitc4_bundle.panels[0]
        label_m = getattr(p0_m, "label", "") or "panel_0"
        saved.append(
            save_mitc4_resultants_figure(
                out_dir / f"mitc4_resultants_along_panel{suf}.png",
                all_panel_mitc4[0],
                panel_label=str(label_m),
                dpi=dpi,
            )
        )

    if all_panel_mitc4:
        all_panel_fi: list[np.ndarray] = []
        panel_labels_m: list[str] = []
        for pi, (p_m, panel_res) in enumerate(zip(mitc4_bundle.panels, all_panel_mitc4)):
            if not panel_res:
                all_panel_fi.append(np.array([]))
                panel_labels_m.append(f"panel_{pi}")
                continue
            plies_m = p_m.lam.build_plies()
            clpt_results = sweep_panel_clpt_fi(
                panel_res,
                plies_m,
                Xt=st["Xt"],
                Xc=st["Xc"],
                Yt=st["Yt"],
                Yc=st["Yc"],
                S12=st["S12"],
            )
            fi_elem = np.array(
                [float(np.max(r.fi)) if len(r.fi) else 0.0 for r in clpt_results]
            )
            all_panel_fi.append(fi_elem)
            panel_labels_m.append(getattr(p_m, "label", None) or f"panel_{pi}")

        saved.append(
            save_mitc4_fi_figure(
                out_dir / f"mitc4_hashin_fi_heatmap{suf}.png",
                all_panel_fi,
                panel_labels_m,
                dpi=dpi,
            )
        )
        saved.append(
            save_clpt_fi_on_section_geometry(
                out_dir / f"clpt_fi_on_section_geometry{suf}.png",
                airfoil,
                webs_geom,
                spars,
                panels,
                all_panel_fi,
                dpi=dpi,
            )
        )

    return saved
