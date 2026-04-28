"""
Post-solve thin-wall + MITC4 shell recovery driven by beam section resultants.

Phase 1 couples beam Gauss-sample resultants ``[N, Vy, Vz, My, Mz, T, B]`` (same order as
``BeamSolveResult.resultants``) into ``run_section_both`` at each ``station_z`` from
section properties. Output is a compact JSON-serializable dict for ``beam_result.json``
under ``shell_recovery``.

**Phase 2** (optional): compare shell vs strip-based FI in ``section_properties`` summaries
without changing ``K6``/``K7``.

**Phase 3** (research): homogenise shell panels to equivalent ``K6``/``K7`` for the beam.

Warping: ``dB_dx`` is computed via central finite differences of the bimoment column
(index 6) in ``BeamSolveResult.resultants`` w.r.t. ``z_stations_out``.  One-sided
differences are used at the endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Process-local cache: airfoil polygon for fixed (NACA) geometry is invariant across design iterations
# when the same station is revisited; only ``run_section_both`` resultants/ABD depend on beam loads.
_airfoil_polygon_cache: dict[tuple, NDArray[np.float64]] = {}


def _airfoil_for_station_keyed(
    m: float,
    p: float,
    xx: float,
    chord_m: float,
    naca_series: int,
) -> NDArray[np.float64]:
    """Return cached 2D airfoil coordinates for a rounded geometry key."""
    from blade_precompute.section_shell_model.job_outputs import build_airfoil_for_station

    akey = (round(m, 12), round(p, 12), round(xx, 12), round(chord_m, 12), int(naca_series))
    if akey not in _airfoil_polygon_cache:
        _airfoil_polygon_cache[akey] = build_airfoil_for_station(
            m, p, xx, chord_m=chord_m, naca_series=naca_series
        )
    return _airfoil_polygon_cache[akey]


@dataclass(frozen=True)
class BladeShellEnrichmentInputs:
    """Minimal spanwise inputs required for shell recovery enrichment."""

    span_r_z_m: NDArray[np.float64]
    naca_m: NDArray[np.float64]
    naca_p: NDArray[np.float64]
    naca_xx: NDArray[np.float64]
    naca_series: NDArray[np.int64]
    chord_m: NDArray[np.float64]


def _compute_dBdz_at_zq(
    z_gp: NDArray[np.float64],
    resultants: NDArray[np.float64],
    zq: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Central FD of bimoment column (index 6) over Gauss stations, interpolated to ``zq``."""
    B_gp = resultants[:, 6]
    if z_gp.size >= 2:
        dBdz_gp = np.gradient(B_gp, z_gp)
    else:
        dBdz_gp = np.zeros_like(B_gp)
    return np.interp(zq, z_gp, dBdz_gp)


def shell_recovery_payload(
    res: Any,
    inp: BladeShellEnrichmentInputs,
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

    z_gp = np.asarray(z_out, dtype=np.float64).ravel()
    resultants_arr = np.asarray(res.resultants, dtype=np.float64)
    R = sample_resultants_at_z(zq, z_gp, resultants_arr)
    dBdz_at_zq = _compute_dBdz_at_zq(z_gp, resultants_arr, zq)

    strengths = default_skin_strengths_pa()
    z_src = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()

    rows: list[dict[str, Any]] = []
    for i in range(zq.shape[0]):
        z = float(zq[i])
        N, Vy, Vz, My, Mz, T, B = (float(R[i, j]) for j in range(7))
        dB_dx = float(dBdz_at_zq[i])
        zm = np.array([z], dtype=np.float64)
        m = float(np.interp(zm, z_src, np.asarray(inp.naca_m, dtype=np.float64))[0])
        p = float(np.interp(zm, z_src, np.asarray(inp.naca_p, dtype=np.float64))[0])
        xx = float(np.interp(zm, z_src, np.asarray(inp.naca_xx, dtype=np.float64))[0])
        chord = float(np.interp(zm, z_src, np.asarray(inp.chord_m, dtype=np.float64))[0])
        ns = int(
            np.clip(
                round(float(np.interp(zm, z_src, np.asarray(inp.naca_series, dtype=np.float64))[0])),
                4.0,
                6.0,
            )
        )
        airfoil = _airfoil_for_station_keyed(m, p, xx, chord, ns)
        # Spars come in as chord fractions [0..1]; airfoil is chord-scaled (metres),
        # and run_section_both → multi_cell_blade_section.build_section uses spars as
        # physical x-positions on the supplied airfoil. Convert to metres to match.
        spars_m = [float(s) * chord for s in spars]
        _mvp, mitc4 = run_section_both(
            airfoil,
            spars_m,
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
                "dB_dz_Nm_per_m": dB_dx,
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
        "stations": rows,
    }


__all__ = ["BladeShellEnrichmentInputs", "shell_recovery_payload", "_compute_dBdz_at_zq"]
