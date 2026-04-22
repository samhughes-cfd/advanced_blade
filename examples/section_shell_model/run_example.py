"""
Runnable MVP: one NACA section, unit resultants, shell handoff + CLPT Tsai–Wu FI.

Writes PNG diagnostics under ``outputs/`` next to this script (mesh, thin-wall
stress ribbons, CLPT ply figure, along-panel curves, MITC4 resultants, FI heatmap).

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
        save_mitc4_fi_figure,
        save_mitc4_resultants_figure,
        save_panel_along_contour_figure,
        save_shell_mesh_figure,
        save_thin_wall_stress_figures,
    )
    from section_shell_model.lib.local_clpt_shell import (
        default_skin_strengths_pa,
        solve_station_clpt_shell,
        sweep_panel_clpt_fi,
    )
    from section_shell_model.lib.recovery_adapter import (
        build_load_reaction_audit,
        check_panel_equilibrium,
        run_section_with_mitc4_shell,
        run_section_both,
    )

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "section_shell_demo"

    airfoil = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=120)
    spars = [0.35]

    # Single thin-wall solve → both MVP and MITC4 bundles
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
        n_elements_per_panel=12,
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

    # Vlasov sigma_omega for panel 0 (from MITC4 bundle's stored vlasov result)
    vlasov = mitc4_bundle.vlasov_result
    sig_omega_p0 = vlasov.sigma_omega[0] if vlasov and vlasov.sigma_omega else None

    saved.append(
        save_panel_along_contour_figure(
            out_dir / "reference_panel_q_sigma_vs_s.png",
            panels,
            q_tot,
            sig_p,
            panel_index=0,
            sigma_omega_mids=sig_omega_p0,
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

    # ------------------------------------------------------------------
    # MITC4 path — all 6 resultants, E-weighted Vlasov warping
    # ------------------------------------------------------------------
    print()
    print("section_shell_model MITC4 — improved shell solve")

    mitc4_ref = mitc4_bundle.reference_resultants
    all_panel_mitc4 = mitc4_bundle.all_panel_mitc4_results or []

    if mitc4_ref is not None:
        print(f"  panel: {mitc4_ref.panel_label}  station_index={mitc4_ref.station_index}")
        print(f"  Nx  = {mitc4_ref.Nx:.4e} N/m   (prov: {mitc4_ref.provenance['Nx'].kind.value})")
        print(f"  Ny  = {mitc4_ref.Ny:.4e} N/m   (prov: {mitc4_ref.provenance['Ny'].kind.value})")
        print(f"  Nxy = {mitc4_ref.Nxy:.4e} N/m   (prov: {mitc4_ref.provenance['Nxy'].kind.value})")
        print(f"  Mx  = {mitc4_ref.Mx:.4e} N·m/m (prov: {mitc4_ref.provenance['Mx'].kind.value})")
        print(f"  My  = {mitc4_ref.My:.4e} N·m/m (prov: {mitc4_ref.provenance['My'].kind.value})")
        print(f"  Mxy = {mitc4_ref.Mxy:.4e} N·m/m (prov: {mitc4_ref.provenance['Mxy'].kind.value})")
        print(f"  shear centre (Vlasov) = ({mitc4_bundle.y_sc:.5f}, {mitc4_bundle.z_sc:.5f}) m")
        print(f"  I_omega_E (E-weighted) = {mitc4_bundle.I_omega:.4e}")
        if vlasov:
            print(f"  n_cells detected = {vlasov.n_cells}")
            print(f"  web panel indices = {vlasov.web_panel_indices}")

    # Separate quick summary for skin vs web panels from MITC4 element resultants.
    skin_results: list[tuple[str, int, float, float]] = []
    web_results: list[tuple[str, int, float, float]] = []
    for pi, panel_res in enumerate(all_panel_mitc4):
        if not panel_res:
            continue
        label = str(getattr(mitc4_bundle.panels[pi], "label", None) or f"panel_{pi}")
        nx_mid = float(panel_res[len(panel_res) // 2].Nx)
        nxy_mean = float(np.mean([r.Nxy for r in panel_res]))
        item = (label, len(panel_res), nx_mid, nxy_mean)
        if "web" in label.lower():
            web_results.append(item)
        else:
            skin_results.append(item)

    def _print_panel_group(title: str, rows: list[tuple[str, int, float, float]]) -> None:
        if not rows:
            return
        print()
        print(f"  {title}:")
        print(f"  {'Panel':<22} {'n_elem':>7}  {'Nx(mid) [N/m]':>15}  {'mean Nxy [N/m]':>16}")
        for lbl, n_elem, nx_mid, nxy_mean in rows:
            print(f"  {lbl:<22} {n_elem:>7d}  {nx_mid:>15.4e}  {nxy_mean:>16.4e}")

    _print_panel_group("Skin panel quick summary", skin_results)
    _print_panel_group("Web panel quick summary", web_results)

    # Resultants figure for reference panel
    if all_panel_mitc4 and all_panel_mitc4[0]:
        p0_m = mitc4_bundle.panels[0]
        label_m = getattr(p0_m, "label", "") or "panel_0"
        mitc4_fig = save_mitc4_resultants_figure(
            out_dir / "mitc4_resultants_along_panel.png",
            all_panel_mitc4[0],
            panel_label=str(label_m),
            dpi=dpi,
        )
        saved.append(mitc4_fig)
        print(f"  PNG: {mitc4_fig}")

    # CLPT Tsai–Wu failure sweep across all panels
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
            fi_elem = np.array([float(np.max(r.fi_tsai_wu)) if len(r.fi_tsai_wu) else 0.0
                                for r in clpt_results])
            all_panel_fi.append(fi_elem)
            panel_labels_m.append(getattr(p_m, "label", None) or f"panel_{pi}")

        fi_fig = save_mitc4_fi_figure(
            out_dir / "mitc4_tsai_wu_fi_heatmap.png",
            all_panel_fi,
            panel_labels_m,
            dpi=dpi,
        )
        saved.append(fi_fig)
        print(f"  PNG: {fi_fig}")

        # Failure margin summary table
        print()
        print("  Tsai-Wu FI summary (all MITC4 panels):")
        print(f"  {'Panel':<22} {'max FI':>12}  {'elem idx':>8}")
        critical_label, critical_fi = "", 0.0
        for lbl, fi_arr in zip(panel_labels_m, all_panel_fi):
            if len(fi_arr) == 0:
                continue
            idx = int(np.argmax(fi_arr))
            fi_val = float(fi_arr[idx])
            is_web = "web" in lbl.lower()
            tag = " [web]" if is_web else ""
            print(f"  {lbl + tag:<22} {fi_val:>12.4e}  {idx:>8d}")
            if fi_val > critical_fi:
                critical_fi, critical_label = fi_val, lbl
        if critical_label:
            print(f"  Critical: {critical_label} (FI = {critical_fi:.4e})")

    # Panel-to-panel equilibrium check
    if all_panel_mitc4:
        eq_checks = check_panel_equilibrium(
            all_panel_mitc4,
            mitc4_bundle.panels,
            all_panel_mitc4_diagnostics=mitc4_bundle.all_panel_mitc4_diagnostics,
        )
        if eq_checks:
            print()
            print("  Panel boundary equilibrium check:")
            print(
                f"  {'Boundary':<30} {'type':<9} {'ori':<8} "
                f"{'r-dNx':>10} {'r-dNxy':>10} {'f-dNx':>10} "
                f"{'dTx_rel':>10} {'dT_yz_rel':>10} {'mode':>20}  flag"
            )
            n_skin_skin = n_skin_skin_react_pass = 0
            n_skin_web = n_skin_web_react_pass = 0
            n_skin_skin_dTx_pass = n_skin_skin_dTyz_pass = 0
            n_skin_web_dTx_pass = n_skin_web_dTyz_pass = 0
            all_load_mismatch = []
            for di in (mitc4_bundle.all_panel_mitc4_diagnostics or []):
                if not di:
                    continue
                if "load_totals" in di and "boundary_reaction_set" in di:
                    fx = float(di["load_totals"].get("Fx_total", 0.0))
                    fs = float(di["load_totals"].get("Fs_total", 0.0))
                    rx = float(di["boundary_reaction_set"]["start"].get("Fx", 0.0) +
                               di["boundary_reaction_set"]["end"].get("Fx", 0.0))
                    rs = float(di["boundary_reaction_set"]["start"].get("Fs", 0.0) +
                               di["boundary_reaction_set"]["end"].get("Fs", 0.0))
                    mx = abs(rx + fx) / max(abs(fx), abs(rx), 1.0)
                    ms = abs(rs + fs) / max(abs(fs), abs(rs), 1.0)
                    all_load_mismatch.append(max(mx, ms))
            for chk in eq_checks:
                fail_nx = not chk["reaction_pass_nx"]
                fail_nxy = not chk["reaction_pass_nxy"]
                dTx_rel = float(chk.get("dTx_rel", chk.get("resultant_dNxy_rel", 0.0)))
                dT_yz_rel = float(chk.get("dT_yz_rel", 0.0))
                tol_tx = float(chk.get("resultant_tol_dTx", 0.10))
                tol_tyz = float(chk.get("resultant_tol_dT_yz", 0.15))
                if chk["boundary_type"] == "skin-skin":
                    n_skin_skin += 1
                    if chk["reaction_pass"]:
                        n_skin_skin_react_pass += 1
                    if dTx_rel <= tol_tx:
                        n_skin_skin_dTx_pass += 1
                    if dT_yz_rel <= tol_tyz:
                        n_skin_skin_dTyz_pass += 1
                elif chk["boundary_type"] == "skin-web":
                    n_skin_web += 1
                    if chk["reaction_pass"]:
                        n_skin_web_react_pass += 1
                    if dTx_rel <= tol_tx:
                        n_skin_web_dTx_pass += 1
                    if dT_yz_rel <= tol_tyz:
                        n_skin_web_dTyz_pass += 1
                flag = ""
                if fail_nx and fail_nxy:
                    flag = " <-- ! reaction Nx,Nxy"
                elif fail_nx:
                    flag = " <-- ! reaction Nx"
                elif fail_nxy:
                    flag = " <-- ! reaction Nxy"
                boundary = f"{chk['label_i']} | {chk['label_j']}"
                print(
                    f"  {boundary:<30} {chk['boundary_type']:<9} {chk['orientation']:<8} "
                    f"{chk['reaction_dNx_rel']:>10.4f} {chk['reaction_dNxy_rel']:>10.4f} "
                    f"{chk['resultant_dNx_rel']:>10.4f} "
                    f"{dTx_rel:>10.4f} {dT_yz_rel:>10.4f} "
                    f"{str(chk.get('nxy_compare_mode', 'n/a')):>20}{flag}"
                )
            print(
                f"  Primary reaction-balance: skin-skin {n_skin_skin_react_pass}/{n_skin_skin} pass "
                f"(tol 5%), skin-web {n_skin_web_react_pass}/{n_skin_web} pass (tol 10%)."
            )
            print(
                f"  Secondary dTx continuity: skin-skin {n_skin_skin_dTx_pass}/{n_skin_skin} pass "
                f"(tol 5%), skin-web {n_skin_web_dTx_pass}/{n_skin_web} pass (tol 10%)."
            )
            print(
                f"  Secondary dT_yz continuity: skin-skin {n_skin_skin_dTyz_pass}/{n_skin_skin} pass "
                f"(tol 10%), skin-web {n_skin_web_dTyz_pass}/{n_skin_web} pass (tol 15%)."
            )
            audit = build_load_reaction_audit(mitc4_bundle.all_panel_mitc4_diagnostics)
            if all_load_mismatch:
                print(f"  Load-reaction audit: worst panel mismatch = {max(all_load_mismatch):.4f} (rel)")
            print(
                f"  Load-reaction audit summary: n={int(audit['n_panels'])}, "
                f"max={audit['max_rel_mismatch']:.4f}, mean={audit['mean_rel_mismatch']:.4f}"
            )

            # Optional mesh-convergence sweep for secondary traction continuity.
            print()
            print("  Secondary traction continuity mesh sweep:")
            print(
                f"  {'n_elem':>6} {'ss f-dNx':>10} {'sw f-dNx':>10} "
                f"{'ss dTx':>10} {'sw dTx':>10} {'ss dT_yz':>10} {'sw dT_yz':>10}"
            )
            for n_elem in (8, 12, 16, 24):
                sweep = run_section_with_mitc4_shell(
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
                    n_elements_per_panel=n_elem,
                    use_global_coupled=True,
                )
                checks_sw = check_panel_equilibrium(
                    sweep.all_panel_mitc4_results or [],
                    sweep.panels,
                    all_panel_mitc4_diagnostics=sweep.all_panel_mitc4_diagnostics,
                )
                ss_nx = [c["resultant_dNx_rel"] for c in checks_sw if c["boundary_type"] == "skin-skin"]
                sw_nx = [c["resultant_dNx_rel"] for c in checks_sw if c["boundary_type"] == "skin-web"]
                ss_dtx = [c.get("dTx_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-skin"]
                sw_dtx = [c.get("dTx_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-web"]
                ss_dtyz = [c.get("dT_yz_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-skin"]
                sw_dtyz = [c.get("dT_yz_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-web"]
                print(
                    f"  {n_elem:>6d} "
                    f"{(max(ss_nx) if ss_nx else 0.0):>10.4f} {(max(sw_nx) if sw_nx else 0.0):>10.4f} "
                    f"{(max(ss_dtx) if ss_dtx else 0.0):>10.4f} {(max(sw_dtx) if sw_dtx else 0.0):>10.4f} "
                    f"{(max(ss_dtyz) if ss_dtyz else 0.0):>10.4f} {(max(sw_dtyz) if sw_dtyz else 0.0):>10.4f}"
                )

            # Tiered acceptance summary.
            print()
            print("  === Tiered acceptance (interface equilibrium) ===")
            _THRESHOLDS = {
                "primary_nx":   ("skin-skin r-dNx ≤5%",  "skin-web r-dNx ≤10%"),
                "primary_nxy":  ("skin-skin r-dNxy≤5%",  "skin-web r-dNxy≤10%"),
                "secondary_tx": ("skin-skin |dTx|≤5%",   "skin-web |dTx|≤10%"),
                "secondary_yz": ("skin-skin |dT_yz|≤10%","skin-web |dT_yz|≤15%"),
            }
            results_by_type = {"skin-skin": [c for c in eq_checks if c["boundary_type"] == "skin-skin"],
                               "skin-web": [c for c in eq_checks if c["boundary_type"] == "skin-web"]}
            def _pass_frac(checks, key, tol):
                if not checks:
                    return "—/—"
                n_pass = sum(1 for c in checks if float(c.get(key, 1.0)) <= tol)
                return f"{n_pass}/{len(checks)}"
            print(f"  {'Metric':<30} {'skin-skin':>12} {'skin-web':>12}  status")
            rows_e = [
                ("Primary r-dNx",  "reaction_dNx_rel",  0.05,  0.10),
                ("Primary r-dNxy", "reaction_dNxy_rel", 0.05,  0.10),
                ("Secondary |dTx|_rel", "dTx_rel",      0.05,  0.10),
                ("Secondary |dT_yz|_rel", "dT_yz_rel",  0.10,  0.15),
            ]
            all_pass = True
            for label, key, tol_ss, tol_sw in rows_e:
                ss_r = _pass_frac(results_by_type["skin-skin"], key, tol_ss)
                sw_r = _pass_frac(results_by_type["skin-web"], key, tol_sw)
                ss_ok = all(float(c.get(key, 1.0)) <= tol_ss for c in results_by_type["skin-skin"])
                sw_ok = all(float(c.get(key, 1.0)) <= tol_sw for c in results_by_type["skin-web"])
                ok_str = "PASS" if (ss_ok and sw_ok) else "FAIL"
                if not (ss_ok and sw_ok):
                    all_pass = False
                print(f"  {label:<30} {ss_r:>12} {sw_r:>12}  {ok_str}")
            print(f"  {'Overall':30} {'':>12} {'':>12}  {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
