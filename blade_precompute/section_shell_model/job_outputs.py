"""
Precompute job diagnostics: PNG bundle matching ``run_example`` MVP+MITC4 plots (no mesh sweeps).

Writes station-tagged filenames under ``out_dir`` (typically ``<job>/section_shell_model/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute._utils.jsonutil import to_jsonable, write_json


def airfoil_polygon_to_multicell_airfoil(
    xy: NDArray[np.float64],
    *,
    n_each: int,
    chord_spacing: str = "nested_cosine",
) -> NDArray[np.float64]:
    """
    Convert ``airfoil_vertices_from_spanwise`` output to the thin-wall contract.

    Parametric generators stack an **outer polygon** (upper TE→LE, then lower
    LE→TE, unequal row counts). ``multi_cell_blade_section`` and
    ``open_outline_from_airfoil`` expect ``[upper_LE→TE; lower_LE→TE]`` with
    **equal** row counts on identical chordwise stations.
    """
    v = np.asarray(xy, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 4:
        raise ValueError(f"airfoil_polygon_to_multicell_airfoil: need (N,2) with N>=4; got {v.shape}.")
    ne = int(max(4, int(n_each)))
    # Ensure examples/section_stress_model is on sys.path so its internal
    # relative imports (e.g. `from lib.laminate_clpt import ...`) resolve.
    # job_outputs.py lives at <repo>/blade_precompute/section_shell_model/job_outputs.py
    # so parents[2] == <repo root>.
    import sys
    _stress_root = str(Path(__file__).resolve().parents[2] / "examples" / "section_stress_model")
    if _stress_root not in sys.path:
        sys.path.insert(0, _stress_root)
    from examples.section_stress_model.multi_cell_blade_section import (
        chordwise_stations,  # type: ignore[import-untyped]
    )

    le_i = int(np.argmin(v[:, 0]))
    upper_te_le = v[: le_i + 1]
    lower_le_te = v[le_i:]
    if len(upper_te_le) < 2 or len(lower_le_te) < 2:
        raise ValueError("airfoil_polygon_to_multicell_airfoil: degenerate upper/lower split at LE.")
    upper = upper_te_le[::-1].copy()
    lower = lower_le_te.copy()

    def _mean_z_on_unique_x(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x.astype(np.float64))
        xs = x[order].astype(np.float64)
        zs = z[order].astype(np.float64)
        ux, inv = np.unique(xs, return_inverse=True)
        cnt = np.bincount(inv)
        uz = np.bincount(inv, weights=zs) / np.maximum(cnt, 1)
        return ux, uz

    xu, zu = _mean_z_on_unique_x(upper[:, 0], upper[:, 1])
    xl, zl = _mean_z_on_unique_x(lower[:, 0], lower[:, 1])
    x_le = float(max(xu.min(), xl.min()))
    x_te = float(min(xu.max(), xl.max()))
    span = max(x_te - x_le, 1e-18)
    xc = x_le + chordwise_stations(ne, spacing=chord_spacing) * span

    zu_i = np.interp(xc, xu, zu)
    zl_i = np.interp(xc, xl, zl)
    z_mid_le = float(0.5 * (zu_i[0] + zl_i[0]))
    z_mid_te = float(0.5 * (zu_i[-1] + zl_i[-1]))
    zu_i[0] = z_mid_le
    zl_i[0] = z_mid_le
    zu_i[-1] = z_mid_te
    zl_i[-1] = z_mid_te
    return np.vstack(
        [
            np.column_stack([xc.astype(np.float64), zu_i.astype(np.float64)]),
            np.column_stack([xc.astype(np.float64), zl_i.astype(np.float64)]),
        ]
    )


def naca_digits_to_params(naca_m: float, naca_p: float, naca_xx: float) -> tuple[float, float, float]:
    """
    Map spanwise NACA digits (same convention as spanwise NACA label generation)
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
    naca_series: int = 4,
    n_points: int = 120,
) -> NDArray[np.float64]:
    """NACA polyline (metres chord) for shell recovery; respects ``naca_series`` (4/5/6)."""
    from blade_precompute.section_geometry.geometry.naca_parametric import airfoil_vertices_from_spanwise

    c = float(max(chord_m, 1e-9))
    if int(naca_series) == 4:
        import sys
        _stress_root = str(Path(__file__).resolve().parents[2] / "examples" / "section_stress_model")
        if _stress_root not in sys.path:
            sys.path.insert(0, _stress_root)
        from examples.section_stress_model.multi_cell_blade_section import (
            naca_four_digit,  # type: ignore[import-untyped]
        )

        m, p, t_c = naca_digits_to_params(naca_m, naca_p, naca_xx)
        return np.asarray(naca_four_digit(m=m, p=p, t_c=t_c, n=n_points), dtype=np.float64) * c
    raw = np.asarray(
        airfoil_vertices_from_spanwise(
            int(naca_series), float(naca_m), float(naca_p), float(naca_xx), int(n_points), c, closed_te=True
        ),
        dtype=np.float64,
    )
    return airfoil_polygon_to_multicell_airfoil(raw, n_each=n_points)


def _serialize_webs_geom(webs_geom: Any) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for item in webs_geom or []:
        if len(item) != 2:
            continue
        u, w = item
        out.append([np.asarray(u, dtype=float).ravel().tolist(), np.asarray(w, dtype=float).ravel().tolist()])
    return out


def _serialize_booms(booms: Any) -> list[Any]:
    rows: list[Any] = []
    for b in booms or []:
        try:
            a = np.asarray(b, dtype=float)
            if a.size and a.ndim <= 2:
                rows.append(a.tolist())
                continue
        except (TypeError, ValueError):
            pass
        if hasattr(b, "__dict__"):
            d: dict[str, Any] = {}
            for k, v in vars(b).items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (float, int, np.floating, np.integer)):
                    d[k] = float(v)
                elif isinstance(v, np.ndarray):
                    d[k] = v.astype(float).tolist()
                else:
                    d[k] = str(v)
            rows.append(d)
        else:
            rows.append(str(b))
    return rows


def _serialize_panel_polylines(panels: Any, q_tot: Any, sig_p: Any, sig_b: Any) -> list[dict[str, Any]]:
    n = len(panels)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p = panels[i]
        label = getattr(p, "label", None) or f"panel_{i}"
        s = np.asarray(getattr(p, "s", []), dtype=float)
        nodes = np.asarray(getattr(p, "nodes", []), dtype=float)
        q_i = np.asarray(q_tot[i], dtype=float).ravel() if i < len(q_tot) else np.array([])
        sp_i = np.asarray(sig_p[i], dtype=float).ravel() if i < len(sig_p) else np.array([])
        sb_i = np.asarray(sig_b[i], dtype=float).ravel() if i < len(sig_b) else np.array([])
        rows.append(
            {
                "label": str(label),
                "s_m": s.tolist(),
                "nodes_m": nodes.tolist(),
                "q_N_per_m": q_i.tolist(),
                "sigma_panel_Pa": sp_i.tolist(),
                "sigma_boom_Pa": sb_i.tolist(),
            }
        )
    return rows


def _serialize_mitc4_station_resultants(panel_resultants: list[Any] | None) -> list[dict[str, Any]] | None:
    if not panel_resultants:
        return None
    out: list[dict[str, Any]] = []
    for r in panel_resultants:
        out.append(
            {
                "station_index": int(getattr(r, "station_index", 0)),
                "Nx": float(getattr(r, "Nx", 0.0)),
                "Ny": float(getattr(r, "Ny", 0.0)),
                "Nxy": float(getattr(r, "Nxy", 0.0)),
                "Mx": float(getattr(r, "Mx", 0.0)),
                "My": float(getattr(r, "My", 0.0)),
                "Mxy": float(getattr(r, "Mxy", 0.0)),
                "sigma_xx_Pa": float(getattr(r, "sigma_xx_pa", 0.0)),
                "tau_xy_Pa": float(getattr(r, "tau_xy_pa", 0.0)),
                "q_n_per_m": float(getattr(r, "q_n_per_m", 0.0)),
                "thickness_m": float(getattr(r, "thickness_m", 0.0)),
                "panel_label": str(getattr(r, "panel_label", "")),
                "panel_index": int(getattr(r, "panel_index", 0)),
            }
        )
    return out


def build_section_shell_station_json_payload(
    *,
    station_tag: str,
    airfoil: NDArray[np.float64],
    spars: list[float],
    n_elements_per_panel: int,
    bundle: Any,
    strengths: dict[str, float],
    clpt_reference_skin: Any,
    all_panel_mitc4: list[Any],
    all_panel_fi: list[np.ndarray],
    panel_labels_m: list[str],
    sig_omega_reference_panel: np.ndarray | None,
) -> Any:
    """Structured data backing the section_shell_model PNG bundle (SI units unless noted)."""
    p0 = bundle.panels[0]
    s_ref = np.asarray(p0.s, dtype=float)
    q_ref = np.asarray(bundle.q_tot[0], dtype=float).ravel()
    sig_ref = np.asarray(bundle.sig_p[0], dtype=float).ravel()
    n = min(len(s_ref), len(q_ref), len(sig_ref))
    s_ref = s_ref[:n]
    q_ref = q_ref[:n]
    sig_ref = sig_ref[:n]
    s_mids: list[float] | None = None
    sig_omega_mpa: list[float] | None = None
    if sig_omega_reference_panel is not None and len(sig_omega_reference_panel) > 0 and n >= 2:
        s_mids_arr = 0.5 * (s_ref[:-1] + s_ref[1:])
        n_om = min(len(s_mids_arr), len(sig_omega_reference_panel))
        s_mids = [float(s_mids_arr[i]) for i in range(n_om)]
        sig_omega_mpa = [float(sig_omega_reference_panel[i]) / 1e6 for i in range(n_om)]

    cr = clpt_reference_skin
    clpt_json: dict[str, Any] = {
        "fi_per_ply": np.asarray(cr.fi, dtype=float).tolist(),
        "eps0": np.asarray(cr.eps0, dtype=float).tolist(),
        "kappa": np.asarray(cr.kappa, dtype=float).tolist(),
        "N_vec_N_per_m": np.asarray(cr.N_vec, dtype=float).tolist(),
        "M_vec_Nmm_per_m": np.asarray(cr.M_vec, dtype=float).tolist(),
        "sig_lam_mid_Pa": [np.asarray(x, dtype=float).tolist() for x in cr.sig_lam_mid],
        "eps_lam_mid": [np.asarray(x, dtype=float).tolist() for x in cr.eps_lam_mid],
    }

    fi_per_panel = [np.asarray(fi, dtype=float).tolist() for fi in all_panel_fi]

    payload: dict[str, Any] = {
        "schema": "section_shell_station_v1",
        "station_tag": station_tag,
        "unit_section_resultants": {
            "N_N": 1.0,
            "Vy_N": 1.0,
            "Vz_N": 1.0,
            "My_Nm": 1.0,
            "Mz_Nm": 1.0,
            "T_Nm": 1.0,
            "B_Nm2": 0.0,
            "dB_dx_Nm": 0.0,
        },
        "n_elements_per_panel": int(n_elements_per_panel),
        "reference_panel_index": 0,
        "airfoil_m": np.asarray(airfoil, dtype=float).tolist(),
        "spar_chord_fracs": [float(x) for x in spars],
        "hashin_strengths_Pa": {k: float(v) for k, v in strengths.items()},
        "thin_wall": {
            "y_sc_m": float(bundle.y_sc),
            "z_sc_m": float(bundle.z_sc),
            "webs_geom": _serialize_webs_geom(bundle.webs_geom),
            "booms": _serialize_booms(bundle.booms),
            "panels": _serialize_panel_polylines(bundle.panels, bundle.q_tot, bundle.sig_p, bundle.sig_b),
        },
        "reference_panel_along_contour": {
            "s_m": s_ref.tolist(),
            "q_N_per_m": q_ref.tolist(),
            "sigma_xx_panel_Pa": sig_ref.tolist(),
            "sigma_omega_Vlasov_mids_m": s_mids,
            "sigma_omega_Vlasov_mids_MPa": sig_omega_mpa,
        },
        "clpt_reference_skin_station": clpt_json,
        "mitc4_panel0_resultants_by_element": _serialize_mitc4_station_resultants(
            all_panel_mitc4[0] if all_panel_mitc4 else None
        ),
        "mitc4_clpt_hashin_fi_max_over_plies_by_panel": {
            "panel_labels": list(panel_labels_m),
            "fi_by_panel": fi_per_panel,
        },
    }
    return to_jsonable(payload)


def write_section_shell_model_station_outputs(
    out_dir: Path,
    *,
    airfoil: NDArray[np.float64],
    spars: list[float],
    station_tag: str,
    n_elements_per_panel: int = 12,
    dpi: int = 150,
    merge_nose: bool = False,
    N: float = 1.0,
    Vy: float = 1.0,
    Vz: float = 1.0,
    My: float = 1.0,
    Mz: float = 1.0,
    T: float = 1.0,
    persist_pngs: bool = True,
    rz_m: float | None = None,
) -> tuple[list[Path], Path]:
    """
    Single-pass shell diagnostics per station.

    ``N``, ``Vy``, ``Vz``, ``My``, ``Mz``, ``T`` are the section resultants for this station.
    Pass real extreme-load values from a beam solve for physically meaningful stress/FI plots.
    Defaults to unit resultants for standalone/test use only.

    ``persist_pngs=False`` writes only the station JSON (no figure files).

    Filenames include ``station_tag`` (e.g. ``i000_rz0.000``) to avoid collisions across stations.

    When ``persist_pngs=True`` (legacy MITC4 path), PNGs written in order include:

    * ``mesh_shell_strips_<tag>.png`` — thin-wall strip layout
    * ``section_shell_demo_<tag>_shear_flow.png``, ``_axial_stress.png`` — thin-wall ribbons
    * ``clpt_ply_hashin_<tag>.png`` — CLPT Hashin on reference panel
    * ``reference_panel_q_sigma_vs_s_<tag>.png`` — streamwise ``q`` / σ
    * ``mitc4_resultants_along_panel_<tag>.png`` — MITC4 resultants (reference panel)
    * ``mitc4_hashin_fi_heatmap_<tag>.png`` — full-section Hashin FI heatmap
    * ``clpt_fi_on_section_geometry_<tag>.png`` — FI on section outline
    * ``multi_panel_hashin_montage_<tag>.png`` — per-panel FI mini heatmaps
    * ``loads_provenance_<tag>.png`` — text block of resultants (optional ``rz_m``)

    Precompute mirrors the same station layout under ``section_shell_model/final/<station>/``
    when optimisation runs (beam-derived resultants).

    Returns
    -------
    png_paths
        Written figure paths in job plot order (empty list when ``persist_pngs=False``).
    station_json
        Path to ``section_shell_station_<tag>.json`` with numeric data backing the PNGs.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_shell_model.lib.local_clpt_shell import (
        default_skin_strengths_pa,
        solve_station_clpt_shell,
        sweep_panel_clpt_fi,
    )
    from blade_precompute.section_shell_model.lib.recovery_adapter import run_section_both

    bundle, mitc4_bundle = run_section_both(
        airfoil,
        spars,
        N=float(N),
        Vy=float(Vy),
        Vz=float(Vz),
        My=float(My),
        Mz=float(Mz),
        T=float(T),
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

    clpt_reference_skin = solve_station_clpt_shell(
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

    vlasov = mitc4_bundle.vlasov_result
    sig_omega_p0 = vlasov.sigma_omega[0] if vlasov and vlasov.sigma_omega else None

    all_panel_mitc4 = mitc4_bundle.all_panel_mitc4_results or []
    all_panel_fi: list[np.ndarray] = []
    panel_labels_m: list[str] = []
    if all_panel_mitc4:
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

    if persist_pngs:
        from blade_precompute.section_shell_model.vis import (
            save_clpt_fi_on_section_geometry,
            save_clpt_ply_figure,
            save_loads_provenance_png,
            save_mitc4_fi_figure,
            save_mitc4_resultants_figure,
            save_multi_panel_hashin_montage,
            save_panel_along_contour_figure,
            save_shell_mesh_figure,
            save_thin_wall_stress_figures,
        )

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
        if all_panel_fi:
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
            saved.append(
                save_multi_panel_hashin_montage(
                    out_dir / f"multi_panel_hashin_montage{suf}.png",
                    all_panel_fi,
                    panel_labels_m,
                    dpi=dpi,
                )
            )
        saved.append(
            save_loads_provenance_png(
                out_dir / f"loads_provenance{suf}.png",
                N=float(N),
                Vy=float(Vy),
                Vz=float(Vz),
                My=float(My),
                Mz=float(Mz),
                T=float(T),
                station_tag=station_tag,
                rz_m=rz_m,
                n_elements_per_panel=int(n_elements_per_panel),
                dpi=dpi,
            )
        )

    sig_omega_arr = np.asarray(sig_omega_p0, dtype=float) if sig_omega_p0 is not None else None
    json_payload = build_section_shell_station_json_payload(
        station_tag=station_tag or "station",
        airfoil=airfoil,
        spars=spars,
        n_elements_per_panel=int(n_elements_per_panel),
        bundle=bundle,
        strengths=st,
        clpt_reference_skin=clpt_reference_skin,
        all_panel_mitc4=all_panel_mitc4,
        all_panel_fi=all_panel_fi,
        panel_labels_m=panel_labels_m,
        sig_omega_reference_panel=sig_omega_arr,
    )
    json_name = f"section_shell_station_{station_tag}.json" if station_tag else "section_shell_station.json"
    station_json = write_json(out_dir / json_name, json_payload)
    return saved, station_json
