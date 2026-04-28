"""MITC4 shell → beam K7 homogenisation for the precompute pipeline (Group F / I coupling).

Uses :func:`~blade_precompute.section_shell_model.lib.homogenisation.compute_section_K7_from_shell`
with the unconstrained stiffness from :func:`~blade_precompute.section_shell_model.lib.global_mitc4_assembly.solve_global_coupled_mitc4`
(..., ``return_assembly_data=True``). Membrane-unit loads ``Nx``, ``Nxy`` are zero; only ``K``
matters for the bilinear stiffness extraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_shell_homogenized_K7_stack(
    inp: Any,
    orchestration: Any,
    station_z_m: NDArray[np.float64],
    *,
    n_elements_per_panel: int = 12,
    run_log: Any | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Return ``(n_stations, 7, 7)`` K7 from MITC4 shell stiffness at each section station.

    Parameters
    ----------
    inp
        :class:`~blade_precompute.orchestration.precompute.containers.PrecomputeInputs`
        (spanwise airfoil + chord columns aligned with ``station_z_m`` rows).
    orchestration
        :class:`~blade_precompute.orchestration.PrecomputeOrchestrationContext`
    station_z_m
        ``(n_stations,)`` spanwise coordinates [m] — typically ``SectionPropertiesOutputs.station_z``.
    """
    from blade_precompute.orchestration import build_section_view
    from blade_precompute.section_shell_model.lib.global_mitc4_assembly import (
        NElements,
        solve_global_coupled_mitc4,
    )
    from blade_precompute.section_shell_model.lib.homogenisation import (
        compute_elastic_centroid_from_panels,
        compute_section_K7_from_shell,
    )
    from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh
    from blade_precompute.section_shell_model.lib.shell_inputs_from_section import build_shell_mesh_inputs

    span = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
    z_sta = np.asarray(station_z_m, dtype=np.float64).ravel()
    n_s = int(z_sta.size)
    out = np.zeros((n_s, 7, 7), dtype=np.float64)
    per_station: list[dict[str, Any]] = []

    for si in range(n_s):
        z_i = float(z_sta[si])
        ii = int(np.argmin(np.abs(span - z_i)))
        chord = float(np.asarray(inp.chord_m, dtype=np.float64).ravel()[ii])
        twist_deg_ii = float(np.asarray(inp.twist_deg, dtype=np.float64).ravel()[ii])
        twist_rad = float(np.deg2rad(twist_deg_ii))
        airfoil_sdf_i, _ = _airfoil_sdf_at_index(inp, ii, chord=chord)
        section_i = build_section_view(airfoil_sdf_i, orchestration.layout, twist_angle_rad=twist_rad)
        shell_inputs_i = build_shell_mesh_inputs(
            section_i,
            twist_rad=twist_rad,
            layout_key=orchestration.system_type_key,
        )
        mesh_i = build_mitc4_mesh(shell_inputs_i, n_elements_per_panel=int(n_elements_per_panel))
        panels = list(mesh_i.panels)
        n_ep: NElements = [int(x) for x in mesh_i.n_elements_per_panel]
        Nx_panels = [np.zeros_like(np.asarray(p.s, dtype=float)) for p in panels]
        Nxy_panels = [np.zeros_like(np.asarray(p.s, dtype=float)) for p in panels]

        try:
            _res, _diag, K_global, _T, node_meta = solve_global_coupled_mitc4(
                panels,
                Nx_panels,
                Nxy_panels,
                n_elements_per_panel=n_ep,
                bc_mode="full_clamp",
                interface_constraint_mode="transformed_basis",
                return_assembly_data=True,
            )
        except Exception as exc:
            per_station.append(
                {
                    "station_index": si,
                    "z_m": z_i,
                    "ok": False,
                    "error": str(exc),
                }
            )
            if run_log is not None:
                try:
                    run_log.warn_event(
                        "shell_k7_homogenize.station_failed",
                        station_index=si,
                        z_m=z_i,
                        error=str(exc),
                    )
                except Exception:
                    pass
            continue

        y_e, z_e = compute_elastic_centroid_from_panels(panels)
        K7_i = compute_section_K7_from_shell(
            K_global,
            node_meta,
            y_e,
            z_e,
            omega_hat=None,
            run_log=run_log,
        )
        out[si] = K7_i
        per_station.append(
            {
                "station_index": si,
                "z_m": z_i,
                "ok": True,
                "K7_cond": float(np.linalg.cond(K7_i)),
            }
        )

    return out, per_station


def _airfoil_sdf_at_index(inp: Any, i: int, *, chord: float) -> tuple[Any, str]:
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.section_geometry.geometry.naca_parametric import spanwise_airfoil_label

    series = int(np.clip(int(round(float(inp.naca_series[i]))), 4, 6))
    af = AirfoilSDF.from_naca_series(
        series,
        float(inp.naca_m[i]),
        float(inp.naca_p[i]),
        float(inp.naca_xx[i]),
        n_points=200,
        chord=float(chord),
        closed_te=True,
    )
    label = spanwise_airfoil_label(series, float(inp.naca_m[i]), float(inp.naca_p[i]), float(inp.naca_xx[i]))
    return af, label
