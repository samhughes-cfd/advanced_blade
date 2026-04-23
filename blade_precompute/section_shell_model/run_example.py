"""
Runnable MVP: one NACA section, unit resultants, shell handoff + CLPT Hashin-envelope FI.

Writes PNG diagnostics under ``outputs/`` next to this script (mesh, thin-wall
stress ribbons, CLPT ply figure, along-panel curves, MITC4 resultants, FI heatmap).

Run from repo root::

    python blade_precompute/section_shell_model/run_example.py

Or use the thin wrapper::

    python examples/section_shell_model/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    examples = repo / "examples"
    stress = examples / "section_stress_model"
    # Order: repo (blade_precompute), stress first for ``lib.*``, then examples.
    for p in (repo, stress, examples):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def main() -> None:
    _bootstrap_path()

    import numpy as np

    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
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
    from blade_precompute.section_shell_model.lib.recovery_adapter import (
        build_load_reaction_audit,
        check_cluster_equilibrium,
        check_panel_equilibrium,
        run_section_with_mitc4_shell,
        run_section_both,
        _diag_boundary_edge_traction_stats,
    )

    repo = Path(__file__).resolve().parent.parent.parent
    out_dir = repo / "examples" / "section_shell_model" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "section_shell_demo"

    airfoil = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=120)
    spars = [0.15, 0.50]
    # Default: two half-skin panels at the LE (USkin C1 + LSkin C1). Set merge_nose=True
    # in run_section_both to splice them into a single "Nose" panel for the shell handoff.
    merge_nose = False

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
        merge_nose=merge_nose,
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

    fi_max = float(np.max(result.fi)) if len(result.fi) else 0.0

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
            out_dir / "clpt_ply_hashin.png",
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
    print(f"  max Hashin FI = {fi_max:.6e}")
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

    # CLPT Hashin failure sweep across all panels
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

        fi_fig = save_mitc4_fi_figure(
            out_dir / "mitc4_hashin_fi_heatmap.png",
            all_panel_fi,
            panel_labels_m,
            dpi=dpi,
        )
        saved.append(fi_fig)
        print(f"  PNG: {fi_fig}")

        section_fi_fig = save_clpt_fi_on_section_geometry(
            out_dir / "clpt_fi_on_section_geometry.png",
            airfoil,
            webs_geom,
            spars,
            panels,
            all_panel_fi,
            dpi=dpi,
        )
        saved.append(section_fi_fig)
        print(f"  PNG: {section_fi_fig}")

        # Failure margin summary table
        print()
        print("  Hashin FI summary (all MITC4 panels):")
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
            # Tolerances: tight when Nose panel removes LE 2-way (merge_nose);
            # else calibrated MPC floor (35% / 40% ss) from airfoil-n sweeps.
            if merge_nose:
                _TOL_SS_REACT, _TOL_SW_REACT = 0.15, 0.10
                _TOL_SS_DTX,   _TOL_SW_DTX   = 0.15, 0.10
                _TOL_SS_DTYZ,  _TOL_SW_DTYZ  = 0.25, 0.15
                _p_ss, _p_sw, _dss, _dsw = "15%", "10%", "25%", "15%"
            else:
                _TOL_SS_REACT, _TOL_SW_REACT = 0.35, 0.10
                _TOL_SS_DTX,   _TOL_SW_DTX   = 0.35, 0.10
                _TOL_SS_DTYZ,  _TOL_SW_DTYZ  = 0.40, 0.15
                _p_ss, _p_sw, _dss, _dsw = "35%", "10%", "40%", "15%"
            for chk in eq_checks:
                fail_nx = not chk["reaction_pass_nx"]
                fail_nxy = not chk["reaction_pass_nxy"]
                dTx_rel = float(chk.get("dTx_rel", chk.get("resultant_dNxy_rel", 0.0)))
                dT_yz_rel = float(chk.get("dT_yz_rel", 0.0))
                r_nx = float(chk["reaction_dNx_rel"])
                r_nxy = float(chk["reaction_dNxy_rel"])
                if chk["boundary_type"] == "skin-skin":
                    n_skin_skin += 1
                    if r_nx <= _TOL_SS_REACT and r_nxy <= _TOL_SS_REACT:
                        n_skin_skin_react_pass += 1
                    if dTx_rel <= _TOL_SS_DTX:
                        n_skin_skin_dTx_pass += 1
                    if dT_yz_rel <= _TOL_SS_DTYZ:
                        n_skin_skin_dTyz_pass += 1
                elif chk["boundary_type"] == "skin-web":
                    n_skin_web += 1
                    if r_nx <= _TOL_SW_REACT and r_nxy <= _TOL_SW_REACT:
                        n_skin_web_react_pass += 1
                    if dTx_rel <= _TOL_SW_DTX:
                        n_skin_web_dTx_pass += 1
                    if dT_yz_rel <= _TOL_SW_DTYZ:
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
                f"(tol {_p_ss}), skin-web {n_skin_web_react_pass}/{n_skin_web} pass (tol {_p_sw})."
            )
            print(
                f"  Secondary dTx continuity: skin-skin {n_skin_skin_dTx_pass}/{n_skin_skin} pass "
                f"(tol {_p_ss}), skin-web {n_skin_web_dTx_pass}/{n_skin_web} pass (tol {_p_sw})."
            )
            print(
                f"  Secondary dT_yz continuity: skin-skin {n_skin_skin_dTyz_pass}/{n_skin_skin} pass "
                f"(tol {_dss}), skin-web {n_skin_web_dTyz_pass}/{n_skin_web} pass (tol {_dsw})."
            )
            audit = build_load_reaction_audit(mitc4_bundle.all_panel_mitc4_diagnostics)
            if all_load_mismatch:
                print(f"  Load-reaction audit: worst panel mismatch = {max(all_load_mismatch):.4f} (rel)")
            print(
                f"  Load-reaction endpoint-only: n={int(audit['n_panels'])}, "
                f"max={audit['max_rel_mismatch_endpoint']:.4f}, mean={audit['mean_rel_mismatch_endpoint']:.4f}"
            )
            _gbal_at_fixed = audit.get("global_force_balance_rel_at_fixed", float("nan"))
            _gbal_str = f"{_gbal_at_fixed:.2e}" if not (
                _gbal_at_fixed != _gbal_at_fixed  # nan check
            ) else "n/a"
            print(
                f"  Load-reaction target-aware: max={audit['max_rel_mismatch_target']:.4f}, "
                f"mean={audit['mean_rel_mismatch_target']:.4f}"
            )
            print(
                f"  Load-reaction global-balance (at fixed DOFs): {_gbal_str}  "
                f"[panel-sum (informational): {audit['global_force_balance_rel']:.4f}]"
            )

            cluster_checks = check_cluster_equilibrium(
                mitc4_bundle.panels,
                all_panel_mitc4_diagnostics=mitc4_bundle.all_panel_mitc4_diagnostics,
            )
            # Compute cluster tangent-jump angles from panel nodes directly.
            # Uses the same signed abs-angle as _cluster_is_collinear to be consistent.
            def _end_tangent_from_panel(panel: object, which: str) -> tuple[float, float]:
                nd = np.asarray(getattr(panel, "nodes", []), dtype=float)
                if len(nd) < 2:
                    return (1.0, 0.0)
                t = nd[1] - nd[0] if which == "start" else nd[-1] - nd[-2]
                n = float(np.linalg.norm(t))
                if n < 1e-12:
                    return (1.0, 0.0)
                return (float(t[0] / n), float(t[1] / n))

            _cluster_tangent_angles: dict[int, float] = {}
            for cc in cluster_checks:
                cid = cc["cluster_id"]
                _mbr_angles: list[tuple[float, float]] = []
                for mbr in cc.get("members", []):
                    parts = mbr.rsplit(":", 1)
                    if len(parts) != 2:
                        continue
                    lbl_m, end_m = parts[0], parts[1]
                    pi_m = next(
                        (i for i, p in enumerate(mitc4_bundle.panels)
                         if str(getattr(p, "label", None) or f"panel_{i}") == lbl_m),
                        None,
                    )
                    if pi_m is None:
                        continue
                    _mbr_angles.append(_end_tangent_from_panel(mitc4_bundle.panels[pi_m], end_m))
                max_jump = 0.0
                for _ai in range(len(_mbr_angles)):
                    for _aj in range(_ai + 1, len(_mbr_angles)):
                        ta, tb = _mbr_angles[_ai], _mbr_angles[_aj]
                        cos_ab = float(np.clip(abs(ta[0] * tb[0] + ta[1] * tb[1]), 0.0, 1.0))
                        max_jump = max(max_jump, float(np.degrees(np.arccos(cos_ab))))
                _cluster_tangent_angles[cid] = max_jump

            if cluster_checks:
                print()
                print("  Cluster traction equilibrium check:")
                print(
                    f"  {'cluster':>7} {'n':>4} {'Tx_rel':>10} {'T_yz_rel':>10} "
                    f"{'jump_deg':>9} {'kind':>6} {'status':>16} {'members':>40}"
                )
                for cc in cluster_checks:
                    status = cc.get("status", "ok")
                    tx_r = cc["Tx_rel_cluster"]
                    tyz_r = cc["T_yz_rel_cluster"]
                    tx_str = f"{tx_r:>10.4f}" if tx_r == tx_r else f"{'n/a':>10}"
                    tyz_str = f"{tyz_r:>10.4f}" if tyz_r == tyz_r else f"{'n/a':>10}"
                    jump = _cluster_tangent_angles.get(cc["cluster_id"], 0.0)
                    kind = "smooth" if cc.get("cluster_collinear", True) else "cusp"
                    print(
                        f"  {cc['cluster_id']:>7d} {cc['n_panels']:>4d} "
                        f"{tx_str} {tyz_str} {jump:>9.1f} {kind:>6} {status:>16} "
                        f"{', '.join(cc.get('members', []))[:40]:>40}"
                    )

            # Mesh-convergence sweep for secondary traction continuity.
            # When SHELL_MODE_SWEEP=1 both shared and transformed_basis are compared
            # side-by-side and the results are saved to mesh_sweep_secondary.csv.
            import os
            do_mode_sweep = os.environ.get("SHELL_MODE_SWEEP", "0").strip() == "1"
            sweep_modes = (["shared", "transformed_basis"] if do_mode_sweep
                           else ["transformed_basis"])
            sweep_n_elems = (8, 12, 16, 24)

            print()
            print("  Secondary traction continuity mesh sweep:")
            _hdr_mode = f"  {'mode':<20}" if do_mode_sweep else "  "
            print(
                f"{_hdr_mode}{'n_elem':>6} {'ss f-dNx':>10} {'sw f-dNx':>10} "
                f"{'ss dTx':>10} {'sw dTx':>10} {'ss dT_yz':>10} {'sw dT_yz':>10} "
                f"{'cl_Tx':>8} {'cl_Tyz':>8}"
            )

            sweep_csv_rows: list[dict] = []
            for sweep_mode in sweep_modes:
                for n_elem in sweep_n_elems:
                    sw_bundle = run_section_with_mitc4_shell(
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
                        interface_constraint_mode=sweep_mode,
                    )
                    checks_sw = check_panel_equilibrium(
                        sw_bundle.all_panel_mitc4_results or [],
                        sw_bundle.panels,
                        all_panel_mitc4_diagnostics=sw_bundle.all_panel_mitc4_diagnostics,
                    )
                    cl_sw = check_cluster_equilibrium(
                        sw_bundle.panels,
                        all_panel_mitc4_diagnostics=sw_bundle.all_panel_mitc4_diagnostics,
                    )
                    ss_nx = [c["resultant_dNx_rel"] for c in checks_sw if c["boundary_type"] == "skin-skin"]
                    sw_nx = [c["resultant_dNx_rel"] for c in checks_sw if c["boundary_type"] == "skin-web"]
                    ss_dtx = [c.get("dTx_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-skin"]
                    sw_dtx = [c.get("dTx_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-web"]
                    ss_dtyz = [c.get("dT_yz_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-skin"]
                    sw_dtyz = [c.get("dT_yz_rel", 0.0) for c in checks_sw if c["boundary_type"] == "skin-web"]
                    valid_cl = [cc for cc in cl_sw if cc.get("status") != "insufficient_data"]
                    cl_tx_r = max((cc.get("Tx_rel_cluster", 0.0) for cc in valid_cl), default=0.0)
                    cl_tyz_r = max((cc.get("T_yz_rel_cluster", 0.0) for cc in valid_cl), default=0.0)
                    _mode_col = f"  {sweep_mode:<20}" if do_mode_sweep else "  "
                    print(
                        f"{_mode_col}{n_elem:>6d} "
                        f"{(max(ss_nx) if ss_nx else 0.0):>10.4f} {(max(sw_nx) if sw_nx else 0.0):>10.4f} "
                        f"{(max(ss_dtx) if ss_dtx else 0.0):>10.4f} {(max(sw_dtx) if sw_dtx else 0.0):>10.4f} "
                        f"{(max(ss_dtyz) if ss_dtyz else 0.0):>10.4f} {(max(sw_dtyz) if sw_dtyz else 0.0):>10.4f} "
                        f"{cl_tx_r:>8.4f} {cl_tyz_r:>8.4f}"
                    )
                    sweep_csv_rows.append({
                        "mode": sweep_mode,
                        "n_elem": n_elem,
                        "ss_f_dNx": max(ss_nx) if ss_nx else 0.0,
                        "sw_f_dNx": max(sw_nx) if sw_nx else 0.0,
                        "ss_dTx": max(ss_dtx) if ss_dtx else 0.0,
                        "sw_dTx": max(sw_dtx) if sw_dtx else 0.0,
                        "ss_dT_yz": max(ss_dtyz) if ss_dtyz else 0.0,
                        "sw_dT_yz": max(sw_dtyz) if sw_dtyz else 0.0,
                        "cl_Tx_rel": cl_tx_r,
                        "cl_Tyz_rel": cl_tyz_r,
                    })

            # Persist sweep to CSV for B4 acceptance-gate evaluation.
            if sweep_csv_rows:
                import csv as _csv
                csv_path = out_dir / "mesh_sweep_secondary.csv"
                _fieldnames = [
                    "mode", "n_elem",
                    "ss_f_dNx", "sw_f_dNx",
                    "ss_dTx", "sw_dTx",
                    "ss_dT_yz", "sw_dT_yz",
                    "cl_Tx_rel", "cl_Tyz_rel",
                ]
                with open(csv_path, "w", newline="") as _fh:
                    _wr = _csv.DictWriter(_fh, fieldnames=_fieldnames)
                    _wr.writeheader()
                    _wr.writerows(sweep_csv_rows)
                print(f"  [sweep saved → {csv_path}]")

            # Airfoil-polyline refinement sweep: measures how LE traction residual
            # changes as the cusp angle shrinks with increasing airfoil discretisation.
            from multi_cell_blade_section import naca_four_digit as _nf  # type: ignore[import-untyped]
            sweep_airfoil_n = (120, 240, 480)
            print()
            print("  Airfoil polyline refinement sweep (LE traction residual vs. cusp angle):")
            print(
                f"  {'naca_n':>7} {'LE_jump_deg':>12} {'LE_cl_Tx':>10} {'TE_cl_Tx':>10}"
                f" {'LE_r-dNx':>10} {'TE_r-dNx':>10}"
            )
            airfoil_n_csv_rows: list[dict] = []
            for _naca_n in sweep_airfoil_n:
                _air_n = _nf(m=0.02, p=0.4, t_c=0.12, n=_naca_n)
                _sw_n = run_section_with_mitc4_shell(
                    _air_n, spars,
                    N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=1.0,
                    B=0.0, dB_dx=0.0,
                    n_elements_per_panel=12,
                    use_global_coupled=True,
                    interface_constraint_mode="transformed_basis",
                )
                _cl_n = check_cluster_equilibrium(
                    _sw_n.panels,
                    all_panel_mitc4_diagnostics=_sw_n.all_panel_mitc4_diagnostics,
                )
                _chk_n = check_panel_equilibrium(
                    _sw_n.all_panel_mitc4_results or [],
                    _sw_n.panels,
                    all_panel_mitc4_diagnostics=_sw_n.all_panel_mitc4_diagnostics,
                )
                # Find 2-way clusters and match to LE (largest jump) / TE (smallest jump).
                _2way_cl = [cc for cc in _cl_n if cc.get("n_panels", 0) == 2 and cc.get("status") != "insufficient_data"]
                _cl_jumps: list[tuple[float, dict]] = []
                for _cc in _2way_cl:
                    _mbr_tvecs: list[tuple[float, float]] = []
                    for _mbr in _cc.get("members", []):
                        _pts = _mbr.rsplit(":", 1)
                        if len(_pts) != 2:
                            continue
                        _pi2 = next(
                            (i for i, p in enumerate(_sw_n.panels)
                             if str(getattr(p, "label", None) or f"panel_{i}") == _pts[0]),
                            None,
                        )
                        if _pi2 is not None:
                            _mbr_tvecs.append(_end_tangent_from_panel(_sw_n.panels[_pi2], _pts[1]))
                    _jmp = 0.0
                    for _ai2 in range(len(_mbr_tvecs)):
                        for _aj2 in range(_ai2 + 1, len(_mbr_tvecs)):
                            _ta, _tb = _mbr_tvecs[_ai2], _mbr_tvecs[_aj2]
                            _cos = float(np.clip(abs(_ta[0] * _tb[0] + _ta[1] * _tb[1]), 0.0, 1.0))
                            _jmp = max(_jmp, float(np.degrees(np.arccos(_cos))))
                    _cl_jumps.append((_jmp, _cc))
                _cl_jumps.sort(key=lambda x: -x[0])  # largest jump = LE
                _le_jump = _cl_jumps[0][0] if _cl_jumps else float("nan")
                _le_tx = _cl_jumps[0][1].get("Tx_rel_cluster", float("nan")) if _cl_jumps else float("nan")
                _te_tx = _cl_jumps[-1][1].get("Tx_rel_cluster", float("nan")) if len(_cl_jumps) > 1 else float("nan")
                _2way_chk = [c for c in _chk_n if c.get("cluster_size", 0) == 2]
                _chk_by_jump: list[tuple[float, dict]] = []
                for _ch in _2way_chk:
                    _cid_ch = next(
                        (cc["cluster_id"] for cc in _2way_cl
                         if cc.get("cluster_id") == _ch.get("cluster_id_i") or
                         cc.get("cluster_id") == _ch.get("cluster_id_j")),
                        None,
                    )
                    _jmp_ch = next((j for j, cc in _cl_jumps if cc.get("cluster_id") == _cid_ch), 0.0)
                    _chk_by_jump.append((_jmp_ch, _ch))
                _chk_by_jump.sort(key=lambda x: -x[0])
                _le_rdnx = _chk_by_jump[0][1].get("reaction_dNx_rel", float("nan")) if _chk_by_jump else float("nan")
                _te_rdnx = _chk_by_jump[-1][1].get("reaction_dNx_rel", float("nan")) if len(_chk_by_jump) > 1 else float("nan")
                print(
                    f"  {_naca_n:>7d} {_le_jump:>12.2f} {_le_tx:>10.4f} {_te_tx:>10.4f}"
                    f" {_le_rdnx:>10.4f} {_te_rdnx:>10.4f}"
                )
                airfoil_n_csv_rows.append({
                    "naca_n": _naca_n,
                    "LE_jump_deg": _le_jump,
                    "LE_cl_Tx": _le_tx,
                    "TE_cl_Tx": _te_tx,
                    "LE_r_dNx": _le_rdnx,
                    "TE_r_dNx": _te_rdnx,
                })
            if airfoil_n_csv_rows:
                import csv as _csv2
                _an_csv = out_dir / "airfoil_n_sweep.csv"
                _an_fields = ["naca_n", "LE_jump_deg", "LE_cl_Tx", "TE_cl_Tx", "LE_r_dNx", "TE_r_dNx"]
                with open(_an_csv, "w", newline="") as _fh2:
                    _wr2 = _csv2.DictWriter(_fh2, fieldnames=_an_fields)
                    _wr2.writeheader()
                    _wr2.writerows(airfoil_n_csv_rows)
                print(f"  [airfoil-n sweep saved → {_an_csv}]")

            # Tiered acceptance summary.
            print()
            print("  === Tiered acceptance (interface equilibrium) ===")
            results_by_type = {
                "skin-skin": [c for c in eq_checks if c["boundary_type"] == "skin-skin"],
                "skin-web": [c for c in eq_checks if c["boundary_type"] == "skin-web"],
            }
            # Secondary traction checks are only meaningful for true 2-way junctions
            # (Newton III for a pair: tx_i + tx_j = 0). At N-way junctions the
            # pair subset is not expected to vanish — use cluster sums instead.
            results_2way_by_type = {
                "skin-skin": [c for c in results_by_type["skin-skin"] if c.get("cluster_size", 2) == 2],
                "skin-web":  [c for c in results_by_type["skin-web"]  if c.get("cluster_size", 2) == 2],
            }

            def _pass_frac(checks, key, tol):
                if not checks:
                    return "—/—"
                n_pass = sum(1 for c in checks if float(c.get(key, 1.0)) <= tol)
                return f"{n_pass}/{len(checks)}"

            if merge_nose:
                # LE merged: 2-way skin-skin is TE only (~12%). T-junctions attach to the
                # longer Nose; cusp cluster |Tx| can be ~0.35 (still Newton-III on 3 members).
                _PPR, _PSR = 0.15, 0.10
                _DTX_S, _DTX_W = 0.15, 0.10
                _DTY_S, _DTY_W = 0.25, 0.15
                _PL1, _PL2, _Ldx1, _Ldx2, _Ldy1, _Ldy2 = (
                    "15%", "10%", "15%", "10%", "25%", "15%")
            else:
                # MPC-inherent floor at 2-way LE/TE: see airfoil-n sweep in run_example.
                _PPR, _PSR = 0.35, 0.10
                _DTX_S, _DTX_W = 0.35, 0.10
                _DTY_S, _DTY_W = 0.40, 0.15
                _PL1, _PL2, _Ldx1, _Ldx2, _Ldy1, _Ldy2 = (
                    "35%", "10%", "35%", "10%", "40%", "15%")

            print(f"  {'Metric':<36} {'skin-skin':>12} {'skin-web':>12}  status")
            primary_rows = [
                (f"Primary r-dNx  (2-way; ss≤{_PL1} sw≤{_PL2})",
                 "reaction_dNx_rel",  _PPR, _PSR, results_2way_by_type),
                (f"Primary r-dNxy (2-way; ss≤{_PL1} sw≤{_PL2})",
                 "reaction_dNxy_rel", _PPR, _PSR, results_2way_by_type),
            ]
            secondary_rows = [
                (f"Secondary |dTx|   (2-way; ss≤{_Ldx1} sw≤{_Ldx2})",
                 "dTx_rel",  _DTX_S,  _DTX_W,  results_2way_by_type),
                (f"Secondary |dT_yz| (2-way; ss≤{_Ldy1} sw≤{_Ldy2})",
                 "dT_yz_rel",  _DTY_S,  _DTY_W,  results_2way_by_type),
            ]
            all_pass = True
            for label, key, tol_ss, tol_sw, pool in primary_rows + secondary_rows:
                ss_r = _pass_frac(pool["skin-skin"], key, tol_ss)
                sw_r = _pass_frac(pool["skin-web"],  key, tol_sw)
                ss_ok = all(float(c.get(key, 1.0)) <= tol_ss for c in pool["skin-skin"])
                sw_ok = all(float(c.get(key, 1.0)) <= tol_sw for c in pool["skin-web"])
                ok_str = "PASS" if (ss_ok and sw_ok) else "FAIL"
                if not (ss_ok and sw_ok):
                    all_pass = False
                print(f"  {label:<36} {ss_r:>12} {sw_r:>12}  {ok_str}")

            # Cluster-sum traction check split by collinearity:
            #   smooth clusters (LE/TE): MPC-inherent floor, tol 35%/40%
            #   cusp clusters (T-junctions): geometric resolution, tol 15%/20%
            valid_cluster_checks = [c for c in cluster_checks if c.get("status") != "insufficient_data"]
            nway_clusters = [c for c in valid_cluster_checks if c.get("n_panels", 0) >= 3]
            if cluster_checks:
                smooth_cl = [c for c in valid_cluster_checks if c.get("cluster_collinear", True)]
                cusp_cl = [c for c in valid_cluster_checks if not c.get("cluster_collinear", True)]
                if merge_nose:
                    TOL_SM_TX, TOL_SM_TYZ = 0.15, 0.25
                    TOL_CU_TX, TOL_CU_TYZ = 0.40, 0.45
                    _clb_tx, _clb_tyz = "sm≤15% cu≤40%", "sm≤25% cu≤45%"
                else:
                    TOL_SM_TX, TOL_SM_TYZ = 0.35, 0.40
                    TOL_CU_TX, TOL_CU_TYZ = 0.15, 0.20
                    _clb_tx, _clb_tyz = "sm≤35% cu≤15%", "sm≤40% cu≤20%"
                sm_tx_ok = all(c.get("Tx_rel_cluster", 1.0) <= TOL_SM_TX for c in smooth_cl) if smooth_cl else True
                cu_tx_ok = all(c.get("Tx_rel_cluster", 1.0) <= TOL_CU_TX for c in cusp_cl) if cusp_cl else True
                sm_tyz_ok = all(c.get("T_yz_rel_cluster", 1.0) <= TOL_SM_TYZ for c in smooth_cl) if smooth_cl else True
                cu_tyz_ok = all(c.get("T_yz_rel_cluster", 1.0) <= TOL_CU_TYZ for c in cusp_cl) if cusp_cl else True
                cl_tx_ok = sm_tx_ok and cu_tx_ok
                cl_tyz_ok = sm_tyz_ok and cu_tyz_ok
                n_cl = len(valid_cluster_checks)
                n_cl_tx_pass = (
                    sum(1 for c in smooth_cl if c.get("Tx_rel_cluster", 1.0) <= TOL_SM_TX)
                    + sum(1 for c in cusp_cl if c.get("Tx_rel_cluster", 1.0) <= TOL_CU_TX)
                )
                n_cl_tyz_pass = (
                    sum(1 for c in smooth_cl if c.get("T_yz_rel_cluster", 1.0) <= TOL_SM_TYZ)
                    + sum(1 for c in cusp_cl if c.get("T_yz_rel_cluster", 1.0) <= TOL_CU_TYZ)
                )
                _cl_tx_frac  = f"{n_cl_tx_pass}/{n_cl}"
                _cl_tyz_frac = f"{n_cl_tyz_pass}/{n_cl}"
                print(
                    f"  {('Cluster |dTx|   ' + _clb_tx):<36} "
                    f"{_cl_tx_frac:>12} {'':>12}  {'PASS' if cl_tx_ok else 'FAIL'}"
                )
                print(
                    f"  {('Cluster |dT_yz| ' + _clb_tyz):<36} "
                    f"{_cl_tyz_frac:>12} {'':>12}  {'PASS' if cl_tyz_ok else 'FAIL'}"
                )
                if not (cl_tx_ok and cl_tyz_ok):
                    all_pass = False
            _gbal_pass = _gbal_at_fixed == _gbal_at_fixed and _gbal_at_fixed < 1e-6
            print(
                f"  {'Global UX balance (at fixed DOFs)':<36} {'n/a':>12} {'n/a':>12}  "
                f"{'PASS' if _gbal_pass else ('n/a' if _gbal_at_fixed != _gbal_at_fixed else 'FAIL')}"
            )
            print(f"  {'Overall':36} {'':>12} {'':>12}  {'PASS' if all_pass else 'FAIL'}")

            # R4 physics analysis: per-N-way-cluster shear-gradient breakdown.
            if nway_clusters:
                diags = mitc4_bundle.all_panel_mitc4_diagnostics or []
                print()
                print("  === Physics analysis: N-way junction shear-gradient breakdown ===")
                print(
                    f"  {'cluster':>7} {'panel:end':<22} {'Tx_int':>10} "
                    f"{'Tx_mean':>10} {'Tx_std':>10} {'Tx_std_rel':>12}"
                )
                for cc in nway_clusters:
                    cid = cc["cluster_id"]
                    members = cc.get("members", [])
                    for mbr in members:
                        parts = mbr.rsplit(":", 1)
                        if len(parts) != 2:
                            continue
                        lbl, end = parts[0], parts[1]
                        pi = next(
                            (i for i, p in enumerate(mitc4_bundle.panels)
                             if str(getattr(p, "label", None) or f"panel_{i}") == lbl),
                            None,
                        )
                        if pi is None or pi >= len(diags):
                            continue
                        di = diags[pi] or {}
                        tx_int = float(di.get("interface_edge_set", {}).get(end, {}).get("Tx_int", 0.0))
                        stats = _diag_boundary_edge_traction_stats(di, end)
                        print(
                            f"  {cid:>7d} {mbr:<22} {tx_int:>10.4f} "
                            f"{stats['Tx_mean']:>10.4f} {stats['Tx_std']:>10.4f} "
                            f"{stats['Tx_std_rel']:>12.4f}"
                        )


if __name__ == "__main__":
    main()
