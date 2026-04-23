"""
Post-solve thin-wall + MITC4 shell recovery driven by beam section resultants.

Phase 1 couples beam Gauss-sample resultants ``[N, Vy, Vz, My, Mz, T, B]`` (same order as
``BeamSolveResult.resultants``) into ``run_section_both`` at each ``station_z`` from
section properties. Output is a compact JSON-serializable dict for ``beam_result.json``
under ``shell_recovery``.

**Phase 2** (optional): compare shell vs strip-based FI in ``section_properties`` summaries
without changing ``K6``/``K7``.

**Phase 3** (research): homogenise shell panels to equivalent ``K6``/``K7`` for the beam.

Warping: ``dB_dx`` is set to ``0.0`` in this MVP. The seventh beam resultant is passed as
bimoment ``B`` into the thin-wall/Vlasov shell path; a full beam–shell bimoment derivative map
is not implemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute.containers import PrecomputeInputs
from blade_precompute.orchestration.precompute.grid import interp_series


def shell_recovery_payload(
    res: Any,
    inp: PrecomputeInputs,
    station_z: NDArray[np.float64],
    spars: list[float],
    *,
    n_elements_per_panel: int = 4,
) -> dict[str, Any]:
    """
    For each ``station_z``, interpolate beam resultants, build NACA airfoil from ``inp``,
    run ``run_section_both``, and aggregate max CLPT Hashin FI over all panel elements.
    """
    from blade_precompute.global_beam_model.engine.postprocess import sample_resultants_at_z
    from blade_precompute.section_shell_model.job_outputs import build_airfoil_for_station
    from blade_precompute.section_shell_model.lib.local_clpt_shell import (
        default_skin_strengths_pa,
        sweep_panel_clpt_fi,
    )
    from blade_precompute.section_shell_model.lib.recovery_adapter import run_section_both

    if not spars:
        return {"skipped": True, "reason": "no_shear_webs_for_thin_wall_shell", "stations": []}

    z_out = getattr(res, "z_stations_out", None)
    if z_out is None or np.size(z_out) < 1:
        return {"skipped": True, "reason": "beam_solve_missing_z_stations_out", "stations": []}

    zq = np.asarray(station_z, dtype=np.float64).ravel()
    if zq.size < 1:
        return {"skipped": True, "reason": "empty_station_z", "stations": []}

    R = sample_resultants_at_z(
        zq,
        np.asarray(z_out, dtype=np.float64),
        np.asarray(res.resultants, dtype=np.float64),
    )
    strengths = default_skin_strengths_pa()
    z_src = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()

    rows: list[dict[str, Any]] = []
    for i in range(zq.shape[0]):
        z = float(zq[i])
        N, Vy, Vz, My, Mz, T, B = (float(R[i, j]) for j in range(7))
        dB_dx = 0.0
        zm = np.array([z], dtype=np.float64)
        m = float(interp_series(z_src, np.asarray(inp.naca_m, dtype=np.float64), zm)[0])
        p = float(interp_series(z_src, np.asarray(inp.naca_p, dtype=np.float64), zm)[0])
        xx = float(interp_series(z_src, np.asarray(inp.naca_xx, dtype=np.float64), zm)[0])
        chord = float(interp_series(z_src, np.asarray(inp.chord_m, dtype=np.float64), zm)[0])
        airfoil = build_airfoil_for_station(m, p, xx, chord_m=chord)
        _mvp, mitc4 = run_section_both(
            airfoil,
            spars,
            N=N,
            Vy=Vy,
            Vz=Vz,
            My=My,
            Mz=Mz,
            T=T,
            B=B,
            dB_dx=dB_dx,
            n_elements_per_panel=int(n_elements_per_panel),
        )
        max_fi = 0.0
        panels = mitc4.panels
        apr = mitc4.all_panel_mitc4_results or []
        for pi, p in enumerate(panels):
            if pi >= len(apr):
                continue
            prs = apr[pi]
            if not prs:
                continue
            plies = p.lam.build_plies()
            st_list = sweep_panel_clpt_fi(prs, plies, **strengths)
            for st in st_list:
                max_fi = max(max_fi, float(np.max(np.asarray(st.fi, dtype=float))))
        v = mitc4.vlasov_result
        y_sc = float(v.y_sc) if v is not None else float("nan")
        z_sc = float(v.z_sc) if v is not None else float("nan")
        rows.append(
            {
                "z_m": z,
                "beam_resultants_N_Vy_Vz_My_Mz_T_B": [N, Vy, Vz, My, Mz, T, B],
                "max_clpt_hashin_fi": max_fi,
                "vlasov_y_sc_m": y_sc,
                "vlasov_z_sc_m": z_sc,
                "chord_m": chord,
            }
        )
    return {
        "skipped": False,
        "n_stations": int(zq.shape[0]),
        "spars": list(spars),
        "n_elements_per_panel": int(n_elements_per_panel),
        "note_dB_dx": "MVP uses dB_dx=0; beam–shell bimoment span derivative not mapped.",
        "stations": rows,
    }


__all__ = ["shell_recovery_payload"]
