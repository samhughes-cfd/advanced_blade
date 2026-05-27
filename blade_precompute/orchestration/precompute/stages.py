"""Precompute stage implementations (geometry, properties, global beam, optimisation)."""

from __future__ import annotations

from dataclasses import replace
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import traceback
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration import (
    MIDLINE_CONTRACT_VERSION,
    PrecomputeOrchestrationContext,
    assert_grid_phi_finite,
    build_section_view,
    midline_series_contract_doc,
    section_boundary_stub_from_labels,
)

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    PrecomputeInputs,
    SectionGeometryOutputs,
    SectionOptimisationOutputs,
    SectionPropertiesOutputs,
    SectionShellModelOutputs,
)
from blade_precompute.orchestration.precompute.grid import station_indices, station_subdir_name
from blade_precompute.orchestration.precompute.shell_spars import section_shell_spars_from_layout
from blade_precompute._utils.jsonutil import write_json
from blade_precompute._utils.job_progress import emit_progress_event, throttled_emit_station_index
from blade_precompute._utils.run_logging import RunLogger, get_run_logger
from blade_precompute.orchestration.precompute.vis import (
    plot_section_properties_station,
    write_beam_model_pngs,
    write_section_optimisation_pngs,
)
from blade_precompute.section_optimisation.core.types import OptimisationObjective, apply_dv_to_bg
from blade_precompute.section_properties.core.types import SectionSolveResult


def _airfoil_from_spanwise(
    inp: PrecomputeInputs, i: int, *, chord: float, n_points: int = 200
) -> tuple[Any, str]:
    """Return ``(AirfoilSDF, label)`` for station ``i`` using ``naca_series`` + digit columns."""
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.section_geometry.geometry.naca_parametric import spanwise_airfoil_label

    series = int(np.clip(int(round(float(inp.naca_series[i]))), 4, 6))
    af = AirfoilSDF.from_naca_series(
        series,
        float(inp.naca_m[i]),
        float(inp.naca_p[i]),
        float(inp.naca_xx[i]),
        n_points=int(n_points),
        chord=float(chord),
        closed_te=True,
    )
    label = spanwise_airfoil_label(series, float(inp.naca_m[i]), float(inp.naca_p[i]), float(inp.naca_xx[i]))
    return af, label


def default_dv0(n_station: int):
    from blade_precompute.section_optimisation.core.types import DesignVector

    n = int(n_station)
    return DesignVector(
        t_skin=np.full(n, 0.012, dtype=np.float64),
        t_cap=np.full(n, 0.050, dtype=np.float64),
        t_web=np.full(n, 0.015, dtype=np.float64),
    )


def section_shell_model_skipped_outputs(
    out_dir: Path,
    *,
    orchestration: PrecomputeOrchestrationContext,
    reason: str,
    grid_meta: Mapping[str, Any] | None = None,
    run_log: RunLogger | None = None,
) -> SectionShellModelOutputs:
    """Stage disabled: write ``section_shell_model/summary.json`` and return a typed result (no null in job summary)."""
    out_stage = (out_dir / "section_shell_model").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    log = run_log or get_run_logger(package="section_shell_model", job_dir=out_dir)
    log.info_event("stage.skipped", reason=reason)
    sj = write_json(
        out_stage / "summary.json",
        {
            "skipped": True,
            "reason": reason,
            "stations": [],
            "png_paths": [],
            "station_result_json_paths": [],
            "spars": [],
            "n_elements_per_panel": None,
            "loads_note": None,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )
    return SectionShellModelOutputs(
        station_indices=[],
        station_r_z_m=[],
        png_paths=[],
        summary_json=sj,
        station_result_json_paths=[],
        skipped=True,
    )


def design_eval_payload(ev: Any, dv: Any) -> dict[str, Any]:
    return {
        "mass_kg": float(ev.mass),
        "stiffness_metric_int_trace_k7": float(ev.stiffness_metric),
        "stiffness_metric_over_mass": float(ev.stiffness_metric / max(ev.mass, 1e-300)),
        "max_fi_hashin": float(ev.max_fi_hashin),
        "max_fi_vm": float(ev.max_fi_vm),
        "dv": dv,
    }


def _section_geometry_station_task(
    *,
    i: int,
    rz: float,
    chord: float,
    twist_deg: float,
    naca_series: float,
    naca_m: float,
    naca_p: float,
    naca_xx: float,
    out_stage: str,
    station_subdir: str,
    layout: Any,
    orchestration_meta: Mapping[str, Any],
    n_cells_hint: int,
    props_sdf_nx: int,
    props_sdf_ny: int,
    plot_sdf_nx: int,
    plot_sdf_ny: int,
    do_plot: bool,
) -> dict[str, Any]:
    """Worker-safe station task for section_geometry stage."""
    from blade_precompute.orchestration import (
        MIDLINE_CONTRACT_VERSION,
        build_section_view,
        midline_series_contract_doc,
        section_boundary_stub_from_labels,
    )
    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.section_geometry.geometry.grid import SDFGrid
    from blade_precompute.section_geometry.geometry.naca_parametric import spanwise_airfoil_label
    from blade_precompute.section_geometry.interface.export import SectionPropertiesReport

    plot_section = None
    if do_plot:
        try:
            from blade_precompute.section_geometry.interface.plot import plot_section as _plot_section

            plot_section = _plot_section
        except Exception:  # pragma: no cover
            plot_section = None

    series = int(np.clip(int(round(float(naca_series))), 4, 6))
    airfoil = AirfoilSDF.from_naca_series(
        series,
        float(naca_m),
        float(naca_p),
        float(naca_xx),
        n_points=200,
        chord=float(chord),
        closed_te=True,
    )
    naca_label = spanwise_airfoil_label(series, float(naca_m), float(naca_p), float(naca_xx))

    station_dir = (Path(out_stage) / station_subdir).resolve()
    station_dir.mkdir(parents=True, exist_ok=True)
    twist_rad = float(np.deg2rad(float(twist_deg)))
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil
    props_grid = SDFGrid.from_airfoil(airfoil_b, nx=int(props_sdf_nx), ny=int(props_sdf_ny))

    tag = f"i{i:03d}_rz{rz:.3f}"
    props_json = (station_dir / f"geometry_report_{tag}.json").resolve()
    labels = list(getattr(section, "labels", list(section)))
    job_meta = {
        **dict(orchestration_meta),
        "midline_contract_version": MIDLINE_CONTRACT_VERSION,
        "midline_contract_summary": (midline_series_contract_doc() or "").strip().split("\n\n", 1)[0],
        "section_boundary_export": section_boundary_stub_from_labels(
            labels,
            n_cells_hint=int(n_cells_hint),
        ).to_jsonable(),
    }
    first_label = next(iter(section))
    phi0 = props_grid.eval(section[first_label])
    assert_grid_phi_finite(phi0)
    SectionPropertiesReport(section, props_grid).to_json(props_json, job_meta=job_meta)

    png = None
    if plot_section is not None:
        plot_grid = SDFGrid.from_airfoil(airfoil_b, nx=int(plot_sdf_nx), ny=int(plot_sdf_ny))
        fig, _ = plot_section(
            section,
            plot_grid,
            title=f"section_geometry: NACA{naca_label}, chord={chord:.3g} @ r_z={rz:.3g} m",
        )
        png = (station_dir / f"section_{tag}.png").resolve()
        fig.savefig(png, dpi=170, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    return {
        "i": int(i),
        "r_z_m": float(rz),
        "station_subdir": station_subdir,
        "props_json": props_json,
        "png": png,
    }


def section_geometry_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    section_plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    grid_meta: Mapping[str, Any] | None = None,
    section_solve_n_workers: int = 1,
    persist_pngs: bool = True,
    subdir_override: Path | None = None,
    run_log: RunLogger | None = None,
    progress: Any | None = None,
) -> SectionGeometryOutputs:
    stage_dir = subdir_override if subdir_override is not None else Path("section_geometry")
    out_stage = (out_dir / stage_dir).resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    if run_log is None:
        run_log = get_run_logger(
            package="section_geometry",
            job_dir=out_dir,
            dump_level=inp.log_dump_level,
        )

    try:
        from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
        from blade_precompute.section_geometry.geometry.grid import SDFGrid
        from blade_precompute.section_geometry.interface.export import SectionPropertiesReport

        try:
            from blade_precompute.section_geometry.interface.plot import plot_section
        except Exception:  # pragma: no cover
            plot_section = None
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import section_geometry. Ensure dependencies are installed "
            "(matplotlib is required for plots)."
        ) from e

    idx = station_indices(int(inp.span_r_z_m.shape[0]), section_plot_station_spec)
    run_log.info_event("stations.selected", n_stations=len(idx), station_indices=idx)
    use_csg_ir = os.getenv("SECTION_GEOMETRY_USE_CSG_IR", "0").lower() in {"1", "true", "yes"}
    run_log.info_event("section_geometry.eval_mode", use_csg_ir=bool(use_csg_ir))
    png_paths: list[Path] = []
    geometry_report_json_paths: list[Path] = []
    rz_used: list[float] = []
    props_sdf_nx, props_sdf_ny = 384, 180
    plot_sdf_nx, plot_sdf_ny = 768, 330

    n_workers = max(1, int(section_solve_n_workers))
    do_parallel = n_workers > 1 and len(idx) > 1
    n_idx = len(idx)
    emit_progress_event(
        progress,
        "geometry_stations_begin",
        n_stations=n_idx,
        parallel=bool(do_parallel),
        n_workers=min(n_workers, n_idx) if do_parallel else 1,
    )
    if do_parallel:
        run_log.info_event("stations.parallel", n_workers=n_workers, n_stations=len(idx))
        task_kwargs: list[dict[str, Any]] = []
        for i in idx:
            rz = float(inp.span_r_z_m[i])
            task_kwargs.append(
                {
                    "i": int(i),
                    "rz": rz,
                    "chord": float(inp.chord_m[i]),
                    "twist_deg": float(inp.twist_deg[i]),
                    "naca_series": float(inp.naca_series[i]),
                    "naca_m": float(inp.naca_m[i]),
                    "naca_p": float(inp.naca_p[i]),
                    "naca_xx": float(inp.naca_xx[i]),
                    "out_stage": str(out_stage),
                    "station_subdir": station_subdir_name(i, rz),
                    "layout": orchestration.layout,
                    "orchestration_meta": orchestration.job_meta(),
                    "n_cells_hint": orchestration.layout.n_cells,
                    "props_sdf_nx": props_sdf_nx,
                    "props_sdf_ny": props_sdf_ny,
                    "plot_sdf_nx": plot_sdf_nx,
                    "plot_sdf_ny": plot_sdf_ny,
                    "do_plot": persist_pngs and plot_section is not None,
                }
            )
        nbatch = len(task_kwargs)
        with ProcessPoolExecutor(max_workers=min(n_workers, nbatch)) as ex:
            fut_to_i: dict[Any, int] = {}
            for kw in task_kwargs:
                fut = ex.submit(_section_geometry_station_task, **kw)
                fut_to_i[fut] = int(kw["i"])
            results: list[dict[str, Any]] = []
            done_n = 0
            for fut in as_completed(fut_to_i):
                rec = fut.result()
                results.append(rec)
                done_n += 1
                if throttled_emit_station_index(done_n - 1, nbatch):
                    emit_progress_event(
                        progress,
                        "geometry_station_done",
                        station_index=int(rec["i"]),
                        n=nbatch,
                        r_z_m=float(rec["r_z_m"]),
                        completed=done_n,
                    )
        for rec in sorted(results, key=lambda r: int(r["i"])):
            i = int(rec["i"])
            rz = float(rec["r_z_m"])
            pjson = Path(str(rec["props_json"])).resolve()
            geometry_report_json_paths.append(pjson)
            run_log.log_artefact(pjson, "geometry_report_json", station_index=i, r_z_m=rz)
            p = rec.get("png")
            if p is not None:
                png = Path(str(p)).resolve()
                png_paths.append(png)
                run_log.log_artefact(png, "png", station_index=i, r_z_m=rz)
            rz_used.append(rz)
    else:
        for j, i in enumerate(idx):
            rz = float(inp.span_r_z_m[i])
            chord = float(inp.chord_m[i])
            station_dir = (out_stage / station_subdir_name(i, rz)).resolve()
            station_dir.mkdir(parents=True, exist_ok=True)
            airfoil, naca_label = _airfoil_from_spanwise(inp, i, chord=chord, n_points=200)
            twist_rad = float(np.deg2rad(inp.twist_deg[i]))
            section = build_section_view(airfoil, orchestration.layout, twist_angle_rad=twist_rad)
            airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil
            props_grid = SDFGrid.from_airfoil(airfoil_b, nx=props_sdf_nx, ny=props_sdf_ny)
            _airfoil_vertices = np.asarray(
                getattr(airfoil_b, "vertices", getattr(airfoil_b, "_vertices", np.empty((0, 2)))),
                dtype=np.float64,
            )
            run_log.log_tensor("section_geometry.airfoil_vertices", _airfoil_vertices, step=f"i{i:03d}")

            tag = f"i{i:03d}_rz{rz:.3f}"
            props_json = (station_dir / f"geometry_report_{tag}.json").resolve()
            labels = list(getattr(section, "labels", list(section)))
            job_meta = {
                **orchestration.job_meta(),
                "midline_contract_version": MIDLINE_CONTRACT_VERSION,
                "midline_contract_summary": (midline_series_contract_doc() or "").strip().split("\n\n", 1)[0],
                "section_boundary_export": section_boundary_stub_from_labels(
                    labels,
                    n_cells_hint=orchestration.layout.n_cells,
                ).to_jsonable(),
            }
            first_label = next(iter(section))
            phi0 = props_grid.eval(section[first_label])
            assert_grid_phi_finite(phi0)
            run_log.log_tensor("section_geometry.phi0", np.asarray(phi0), step=f"i{i:03d}")
            SectionPropertiesReport(section, props_grid).to_json(props_json, job_meta=job_meta)
            run_log.log_artefact(props_json, "geometry_report_json", station_index=int(i), r_z_m=rz)
            geometry_report_json_paths.append(props_json)

            if persist_pngs and plot_section is not None:
                plot_grid = SDFGrid.from_airfoil(airfoil_b, nx=plot_sdf_nx, ny=plot_sdf_ny)
                fig, _ = plot_section(
                    section,
                    plot_grid,
                    title=f"section_geometry: NACA{naca_label}, chord={chord:.3g} @ r_z={rz:.3g} m",
                )
                png = (station_dir / f"section_{tag}.png").resolve()
                fig.savefig(png, dpi=170, bbox_inches="tight")
                try:
                    import matplotlib.pyplot as plt

                    plt.close(fig)
                except Exception:
                    pass
                png_paths.append(png)
                run_log.log_artefact(png, "png", station_index=int(i), r_z_m=rz)

            rz_used.append(rz)
            if throttled_emit_station_index(j, n_idx):
                emit_progress_event(
                    progress,
                    "geometry_station",
                    station_index=int(i),
                    n=n_idx,
                    r_z_m=rz,
                )

    write_json(
        out_stage / "summary.json",
        {
            "stations": [
                {
                    "i": int(i),
                    "r_z_m": float(inp.span_r_z_m[i]),
                    "station_subdir": station_subdir_name(i, float(inp.span_r_z_m[i])),
                }
                for i in idx
            ],
            "png_paths": png_paths,
            "geometry_report_json_paths": geometry_report_json_paths,
            "sdf_grid_properties": {"nx": props_sdf_nx, "ny": props_sdf_ny},
            "sdf_grid_plot": {"nx": plot_sdf_nx, "ny": plot_sdf_ny},
            "section_geometry_eval": {"use_csg_ir": bool(use_csg_ir)},
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )
    run_log.log_artefact(out_stage / "summary.json", "summary_json")

    return SectionGeometryOutputs(
        station_indices=idx,
        station_r_z_m=rz_used,
        png_paths=png_paths,
        geometry_report_json_paths=geometry_report_json_paths,
    )


def _v2_use_flag(use_mitc4_v2_path: bool) -> bool:
    """Return True when the v2 midline→MITC4 path is activated via param or env var."""
    return use_mitc4_v2_path or os.environ.get("ADVANCED_BLADE_SHELL_MITC4_V2", "0") == "1"


def _write_v2_station_json(
    out_dir: Path,
    *,
    station_tag: str,
    mesh: Any,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    n_elements_per_panel: int,
) -> Path:
    """Write per-station JSON for the v2 midline→MITC4 path.

    Produces schema ``"section_shell_station_v2"`` with mesh topology under ``thin_wall``
    (panel count, arc lengths, cluster count) so downstream consumers that only
    check ``"thin_wall" in payload`` continue to pass without modification.
    """
    panel_rows = [
        {
            "label": pm.panel_label,
            "kind": pm.kind,
            "arc_length_m": float(pm.arc_length_m),
            "thickness_m": float(pm.thickness_m),
            "n_elements": int(pm.n_elements),
        }
        for pm in mesh.panel_meshes
    ]
    payload: dict[str, Any] = {
        "schema": "section_shell_station_v2",
        "station_tag": station_tag,
        "unit_section_resultants": {
            "N_N": float(N),
            "Vy_N": float(Vy),
            "Vz_N": float(Vz),
            "My_Nm": float(My),
            "Mz_Nm": float(Mz),
            "T_Nm": float(T),
        },
        "n_elements_per_panel": int(n_elements_per_panel),
        "layout_key": mesh.layout_key,
        "chord_m": float(mesh.chord_m),
        "twist_rad": float(mesh.twist_rad),
        "thin_wall": {
            "n_panels": int(len(mesh.panels)),
            "n_total_nodes": int(mesh.n_total_nodes),
            "n_total_elements": int(mesh.n_total_elements),
            "n_clusters": int(len(mesh.clusters)),
            "panels": panel_rows,
        },
        "mesh_summary": mesh.summary(),
    }
    from blade_precompute._utils.jsonutil import to_jsonable
    json_path = out_dir / f"section_shell_station_{station_tag}.json"
    write_json(json_path, to_jsonable(payload))
    return json_path


def section_shell_model_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    section_plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    n_elements_per_panel: int = 12,
    dpi: int = 150,
    grid_meta: Mapping[str, Any] | None = None,
    station_resultants: Mapping[int, tuple[float, float, float, float, float, float]] | None = None,
    persist_pngs: bool = True,
    subdir_override: Path | None = None,
    loads_provenance: str = "unit_resultants",
    use_mitc4_v2_path: bool = False,
    run_log: RunLogger | None = None,
    progress: Any | None = None,
) -> SectionShellModelOutputs:
    """MITC4/CLPT shell bundle per plot station.

    ``station_resultants`` maps station index → ``(N, Vy, Vz, My, Mz, T)`` from a beam solve.
    When None, defaults to unit resultants (useful for standalone tests only).
    ``persist_pngs=False`` writes JSON only (pre-optimisation JSON-only pass).
    ``use_mitc4_v2_path=True`` (or env ``ADVANCED_BLADE_SHELL_MITC4_V2=1``) routes
    per-station solving through build_section_view → build_shell_mesh_inputs →
    build_mitc4_mesh instead of the legacy run_section_both path.
    """
    stage_dir = subdir_override if subdir_override is not None else Path("section_shell_model")
    out_stage = (out_dir / stage_dir).resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    if run_log is None:
        run_log = get_run_logger(
            package="section_shell_model",
            job_dir=out_dir,
            dump_level=inp.log_dump_level,
        )
    spars = section_shell_spars_from_layout(orchestration.layout)
    idx = station_indices(int(inp.span_r_z_m.shape[0]), section_plot_station_spec)
    n_shell = len(idx)
    png_paths: list[Path] = []
    station_result_json_paths: list[Path] = []
    rz_used: list[float] = []
    try:
        from blade_precompute.section_shell_model.job_outputs import (
            build_airfoil_for_station,
            write_section_shell_model_station_outputs,
        )

        for j, i in enumerate(idx):
            rz = float(inp.span_r_z_m[i])
            chord = float(inp.chord_m[i])
            station_dir = (out_stage / station_subdir_name(i, rz)).resolve()
            station_dir.mkdir(parents=True, exist_ok=True)
            airfoil = build_airfoil_for_station(
                float(inp.naca_m[i]),
                float(inp.naca_p[i]),
                float(inp.naca_xx[i]),
                chord_m=chord,
                naca_series=int(inp.naca_series[i]),
            )
            # Spars from system_layout are chord fractions [0..1]; the airfoil from
            # build_airfoil_for_station is already chord-scaled to metres, and
            # multi_cell_blade_section.build_section uses spars as physical x-positions
            # on the supplied airfoil. Convert fractions to metres to match the airfoil.
            spars_m = [float(s) * chord for s in spars]
            tag = f"i{i:03d}_rz{rz:.3f}"
            # Resolve per-station load 6-vector; fall back to unit resultants only if no real loads provided.
            if station_resultants is not None and i in station_resultants:
                sr = station_resultants[i]
                N, Vy, Vz, My, Mz, T = float(sr[0]), float(sr[1]), float(sr[2]), float(sr[3]), float(sr[4]), float(sr[5])
            else:
                N, Vy, Vz, My, Mz, T = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

            if _v2_use_flag(use_mitc4_v2_path):
                # v2 path: build_section_view → build_shell_mesh_inputs → build_mitc4_mesh
                from blade_precompute.section_shell_model.lib.shell_inputs_from_section import (
                    build_shell_mesh_inputs,
                )
                from blade_precompute.section_shell_model.lib.mitc4_mesh import build_mitc4_mesh

                twist_rad_i = float(np.deg2rad(float(inp.twist_deg[i])))
                airfoil_sdf_i, _ = _airfoil_from_spanwise(inp, i, chord=chord)
                section_i = build_section_view(
                    airfoil_sdf_i, orchestration.layout, twist_angle_rad=twist_rad_i
                )
                shell_inputs_i = build_shell_mesh_inputs(
                    section_i,
                    twist_rad=twist_rad_i,
                    layout_key=orchestration.system_type_key,
                )
                mesh_i = build_mitc4_mesh(
                    shell_inputs_i, n_elements_per_panel=int(n_elements_per_panel)
                )
                station_json = _write_v2_station_json(
                    station_dir,
                    station_tag=tag,
                    mesh=mesh_i,
                    N=N,
                    Vy=Vy,
                    Vz=Vz,
                    My=My,
                    Mz=Mz,
                    T=T,
                    n_elements_per_panel=int(n_elements_per_panel),
                )
                station_result_json_paths.append(station_json)
                run_log.log_artefact(station_json, "station_json", station_index=int(i), r_z_m=rz)
                if persist_pngs:
                    from blade_precompute.section_shell_model.vis import (
                        save_loads_provenance_png,
                        save_mitc4_v2_dashboard_figure,
                        save_mitc4_v2_section_mesh_figure,
                    )

                    suf = f"_{tag}"
                    cap = [
                        f"N={N:.4g} Vy={Vy:.4g} Vz={Vz:.4g} My={My:.4g} Mz={Mz:.4g} T={T:.4g}",
                        f"r_z_m={rz:.4f}  provenance={loads_provenance}",
                    ]
                    p_mesh = (station_dir / f"mesh_mitc4_v2{suf}.png").resolve()
                    save_mitc4_v2_section_mesh_figure(p_mesh, mesh_i, dpi=int(dpi), caption_lines=cap)
                    png_paths.append(p_mesh)
                    run_log.log_artefact(p_mesh, "png", station_index=int(i), r_z_m=rz)
                    p_dash = (station_dir / f"station_dashboard_v2{suf}.png").resolve()
                    save_mitc4_v2_dashboard_figure(
                        p_dash,
                        mesh_i,
                        N=N,
                        Vy=Vy,
                        Vz=Vz,
                        My=My,
                        Mz=Mz,
                        T=T,
                        station_tag=tag,
                        rz_m=float(rz),
                        n_elements_per_panel=int(n_elements_per_panel),
                        dpi=int(dpi),
                    )
                    png_paths.append(p_dash)
                    run_log.log_artefact(p_dash, "png", station_index=int(i), r_z_m=rz)
                    p_prov = (station_dir / f"loads_provenance{suf}.png").resolve()
                    save_loads_provenance_png(
                        p_prov,
                        N=N,
                        Vy=Vy,
                        Vz=Vz,
                        My=My,
                        Mz=Mz,
                        T=T,
                        station_tag=tag,
                        rz_m=float(rz),
                        n_elements_per_panel=int(n_elements_per_panel),
                        dpi=int(dpi),
                    )
                    png_paths.append(p_prov)
                    run_log.log_artefact(p_prov, "png", station_index=int(i), r_z_m=rz)
                rz_used.append(rz)
                if throttled_emit_station_index(j, n_shell):
                    emit_progress_event(
                        progress,
                        "shell_station",
                        station_index=int(i),
                        n=n_shell,
                        r_z_m=rz,
                        path="v2",
                    )
                continue  # skip legacy path for this station

            # #region agent log
            try:
                import json as _agent_json
                import time as _agent_time
                _af_x = np.asarray(airfoil, dtype=float)[:, 0]
                _all_x = [0.0] + sorted(float(s) for s in spars_m) + [float(_af_x.max())]
                _entry = {
                    "sessionId": "55cddb",
                    "runId": "post-fix",
                    "hypothesisId": "H1",
                    "location": "stages.py:section_shell_model_impl/per_station",
                    "message": "shell stage units check (post-fix)",
                    "data": {
                        "station_i": int(i),
                        "rz_m": float(rz),
                        "chord_m": float(chord),
                        "spars_chord_frac": [float(s) for s in spars],
                        "spars_m": [float(s) for s in spars_m],
                        "airfoil_x_min": float(_af_x.min()),
                        "airfoil_x_max": float(_af_x.max()),
                        "all_x_post_fix": _all_x,
                        "trailing_cell_x0_m": float(sorted(spars_m)[-1]),
                        "expected_TE_x_m": float(_af_x.max()),
                    },
                    "timestamp": int(_agent_time.time() * 1000),
                }
                with open("debug-55cddb.log", "a", encoding="utf-8") as _f:
                    _f.write(_agent_json.dumps(_entry) + "\n")
            except Exception:
                pass
            # #endregion
            paths, station_json = write_section_shell_model_station_outputs(
                station_dir,
                airfoil=airfoil,
                spars=spars_m,
                station_tag=tag,
                n_elements_per_panel=int(n_elements_per_panel),
                dpi=int(dpi),
                N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
                persist_pngs=persist_pngs,
                rz_m=float(rz),
            )
            if persist_pngs:
                png_paths.extend(paths)
            station_result_json_paths.append(station_json)
            run_log.log_artefact(station_json, "station_json", station_index=int(i), r_z_m=rz)
            for p in paths:
                run_log.log_artefact(p, "png", station_index=int(i), r_z_m=rz)
            rz_used.append(rz)
            if throttled_emit_station_index(j, n_shell):
                emit_progress_event(
                    progress,
                    "shell_station",
                    station_index=int(i),
                    n=n_shell,
                    r_z_m=rz,
                    path="legacy",
                )
        sj = write_json(
            out_stage / "summary.json",
            {
                "skipped": False,
                "stations": [
                    {
                        "i": int(i),
                        "r_z_m": float(inp.span_r_z_m[i]),
                        "station_subdir": station_subdir_name(i, float(inp.span_r_z_m[i])),
                    }
                    for i in idx
                ],
                "png_paths": png_paths,
                "station_result_json_paths": station_result_json_paths,
                "loads_provenance": loads_provenance,
                "n_elements_per_panel": int(n_elements_per_panel),
                "spars": spars,
                "grid": dict(grid_meta) if grid_meta is not None else None,
                "orchestration": orchestration.job_meta(),
            },
        )
        return SectionShellModelOutputs(
            station_indices=idx,
            station_r_z_m=rz_used,
            png_paths=png_paths,
            summary_json=sj,
            station_result_json_paths=station_result_json_paths,
            skipped=False,
        )
    except Exception as e:
        run_log.info_event("stage.error", reason=str(e), traceback=traceback.format_exc())
        if isinstance(e, ModuleNotFoundError):
            raise
        run_log.info_event("stage.skipped", reason=str(e))
        sj = write_json(
            out_stage / "summary.json",
            {
                "skipped": True,
                "reason": str(e),
                "stations": [
                    {
                        "i": int(i),
                        "r_z_m": float(inp.span_r_z_m[i]),
                        "station_subdir": station_subdir_name(i, float(inp.span_r_z_m[i])),
                    }
                    for i in idx
                ],
                "png_paths": [],
                "station_result_json_paths": [],
                "loads_provenance": loads_provenance,
                "spars": spars,
                "n_elements_per_panel": int(n_elements_per_panel),
                "grid": dict(grid_meta) if grid_meta is not None else None,
                "orchestration": orchestration.job_meta(),
            },
        )
        return SectionShellModelOutputs(
            station_indices=idx,
            station_r_z_m=[float(inp.span_r_z_m[i]) for i in idx],
            png_paths=[],
            summary_json=sj,
            station_result_json_paths=[],
            skipped=True,
        )


def section_properties_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    section_plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
    section_solve_n_workers: int = 1,
    persist_pngs: bool = True,
    subdir_override: Path | None = None,
    run_log: RunLogger | None = None,
    progress: Any | None = None,
) -> SectionPropertiesOutputs:
    stage_dir = subdir_override if subdir_override is not None else Path("section_properties")
    out_stage = (out_dir / stage_dir).resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    if run_log is None:
        run_log = get_run_logger(
            package="section_properties",
            job_dir=out_dir,
            dump_level=inp.log_dump_level,
        )

    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.engine.parallel import solve_dirty_stations
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
    from blade_precompute.section_properties.io.section_solve_bundle import save_section_solve_stations_bundle

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    dv0 = getattr(bg, "resolved_dv", None) or default_dv0(int(bg.z_stations.shape[0]))
    section_defs = SectionBuilder.build(dv0, bg)

    z = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)

    n_st = len(section_defs)
    n_wrk = int(section_solve_n_workers)
    emit_progress_event(
        progress,
        "section_solve_begin",
        n_stations=n_st,
        n_workers=max(1, n_wrk),
    )

    def _on_sec_done(si: int, _r: SectionSolveResult) -> None:
        if throttled_emit_station_index(si, n_st):
            emit_progress_event(
                progress,
                "section_solve_station_done",
                station_index=int(si),
                n=n_st,
            )

    res_map = solve_dirty_stations(
        section_defs,
        list(range(n_st)),
        n_workers=n_wrk,
        on_done=_on_sec_done,
    )
    results: list[SectionSolveResult] = [res_map[i] for i in range(n_st)]
    emit_progress_event(progress, "section_solve_done", n_stations=n_st)

    n = len(results)
    K6 = np.stack([np.asarray(r.K6, dtype=np.float64) for r in results], axis=0).reshape(n, 6, 6)
    K7 = np.stack([np.asarray(r.K7, dtype=np.float64) for r in results], axis=0).reshape(n, 7, 7)
    from blade_precompute.global_beam_model.engine.stiffness_validation import (
        validate_k6_k7_stacks,
    )

    for msg in validate_k6_k7_stacks(K6, K7, strict=False):
        warnings.warn(f"stiffness_validation: {msg}", UserWarning, stacklevel=1)
    run_log.log_tensor("section_properties.K6", K6)
    run_log.log_tensor("section_properties.K7", K7)

    summary_rows = []
    for sd, r in zip(section_defs, results):
        summary_rows.append(
            {
                "station_z": float(sd.station_z),
                "area": float(r.area),
                "mass_per_length": float(r.mass_per_length),
                "elastic_center": np.asarray(r.elastic_center, dtype=np.float64),
                "mass_center": np.asarray(r.mass_center, dtype=np.float64),
                "shear_center": np.asarray(r.shear_center, dtype=np.float64),
            }
        )

    summary_json = write_json(
        out_stage / "section_solve_summary.json",
        {
            "blade_yaml": blade_yaml.resolve() if blade_yaml.is_file() else str(blade_yaml),
            "n_station": int(n),
            "stations": summary_rows,
            "K6": K6,
            "K7": K7,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )
    run_log.log_artefact(summary_json, "summary_json")

    png_paths: list[Path] = []
    _png_idx = list(station_indices(int(z.shape[0]), section_plot_station_spec))
    n_png = len(_png_idx)
    for j, i in enumerate(_png_idx):
        zi = float(z[i])
        station_dir = (out_stage / station_subdir_name(i, zi)).resolve()
        station_dir.mkdir(parents=True, exist_ok=True)
        if persist_pngs:
            out_png = (station_dir / "section_station.png").resolve()
            plot_section_properties_station(section_defs[i], results[i], out_png)
            png_paths.append(out_png)
            run_log.log_artefact(out_png, "png", station_index=int(i), r_z_m=zi)
            if throttled_emit_station_index(j, max(1, n_png)):
                emit_progress_event(
                    progress,
                    "section_png_station",
                    station_index=int(i),
                    n=n_png,
                    r_z_m=zi,
                )

    emit_progress_event(progress, "section_bundle_write")
    bundle_meta = save_section_solve_stations_bundle(out_stage, z, results)
    run_log.log_artefact(out_stage / "section_solve_stations.npz", "npz")
    write_json(
        out_stage / "summary.json",
        {
            "results_summary_json": summary_json,
            "png_paths": png_paths,
            "section_solve_bundle": bundle_meta,
        },
    )

    return SectionPropertiesOutputs(
        station_z=z,
        K6=K6,
        K7=K7,
        results_summary_json=summary_json,
        png_paths=png_paths,
        section_results=tuple(results),
        section_definitions=tuple(section_defs),
    )


def beam_model_impl(
    inp: PrecomputeInputs,
    sec: SectionPropertiesOutputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    n_beam_nodes: int,
    orchestration: PrecomputeOrchestrationContext,
    save_section_recovery_cache_npz: bool = False,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
    enable_shell_recovery_enrichment: bool = False,
    shell_recovery_n_elements_per_panel: int = 4,
    beam_png_span_samples: int = 400,
    persist_pngs: bool = True,
    subdir_override: Path | None = None,
    enable_global_buckling: bool = False,
    n_global_buckling_modes: int = 5,
    run_log: RunLogger | None = None,
    progress: Any | None = None,
    enable_shell_k7_homogenisation: bool = False,
    shell_k7_relax: float = 1.0,
    shell_k7_outer_max_iter: int = 1,
    shell_k7_tol_rel: float = 1e-3,
    shell_k7_n_elements_per_panel: int | None = None,
) -> BeamModelOutputs:
    stage_dir = subdir_override if subdir_override is not None else Path("global_beam_model")
    out_stage = (out_dir / stage_dir).resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    if run_log is None:
        run_log = get_run_logger(
            package="global_beam_model",
            job_dir=out_dir,
            dump_level=inp.log_dump_level,
        )

    from blade_precompute.global_beam_model.api import BeamAnalysis
    from blade_precompute.global_beam_model.core.types import BeamLoads, BoundaryCondition, SolverOptions
    from blade_precompute.global_beam_model.engine.blade_geometry import BladeGeometry
    from blade_precompute.global_beam_model.engine.interp import stations_from_arrays
    from blade_precompute.section_optimisation.api import BladeDesignProblem

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    if not getattr(bg, "run_global_beam", True):
        stub = write_json(
            out_stage / "beam_result.json",
            {
                "skipped": True,
                "run_global_beam": False,
                "orchestration": orchestration.job_meta(),
                "grid": dict(grid_meta) if grid_meta is not None else None,
            },
        )
        return BeamModelOutputs(
            result_json=stub,
            png_paths=[],
            beam_n_iterations=None,
            beam_converged=None,
        )

    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        chord=np.asarray(bg.chord, dtype=np.float64),
        twist=np.asarray(bg.twist, dtype=np.float64),
        airfoil_profiles=list(bg.airfoil_profiles),
        web_positions=np.asarray(bg.web_positions, dtype=np.float64),
        subcomponent_materials=dict(bg.subcomponent_materials),
        chi0=None,
    )

    stiffness_source = str(getattr(bg, "beam_section_stiffness_source", "section_properties")).lower()
    if stiffness_source == "gbt":
        raise ValueError(
            'beam_section_stiffness_source="gbt" is not supported in precompute. '
            "Use examples/section_beam_model and examples/section_buckling for GBT-based workflows."
        )

    sec_effective = sec
    shell_k7_meta: dict[str, Any] = {"enabled": bool(enable_shell_k7_homogenisation)}
    if enable_shell_k7_homogenisation:
        try:
            from blade_precompute.orchestration.precompute.shell_k7_homogenize import (
                compute_shell_homogenized_K7_stack,
            )

            n_ep = shell_k7_n_elements_per_panel
            if n_ep is None:
                n_ep = 12
            K7_strip = np.asarray(sec.K7, dtype=np.float64)
            w_blend = float(np.clip(shell_k7_relax, 0.0, 1.0))
            K7_work = K7_strip.copy()
            outer_log: list[dict[str, Any]] = []
            for _out_iter in range(max(1, int(shell_k7_outer_max_iter))):
                K7_shell, per_st = compute_shell_homogenized_K7_stack(
                    inp,
                    orchestration,
                    np.asarray(sec.station_z, dtype=np.float64),
                    n_elements_per_panel=int(n_ep),
                    run_log=run_log,
                )
                K7_next = w_blend * K7_shell + (1.0 - w_blend) * K7_strip
                rel_diff = float(
                    np.linalg.norm(K7_next - K7_work)
                    / max(float(np.linalg.norm(K7_work)), 1e-300)
                )
                outer_log.append(
                    {
                        "outer_iter": int(_out_iter),
                        "rel_diff_K7": rel_diff,
                        "n_stations_ok": sum(1 for r in per_st if r.get("ok")),
                    }
                )
                K7_work = K7_next
                if rel_diff < float(shell_k7_tol_rel):
                    break
            sec_effective = replace(sec, K7=K7_work)
            shell_k7_meta = {
                "enabled": True,
                "relax": w_blend,
                "n_elements_per_panel": int(n_ep),
                "outer_iterations": outer_log,
                "per_station": per_st,
            }
            run_log.info_event(
                "beam_shell_k7_homogenisation",
                relax=w_blend,
                n_elements_per_panel=int(n_ep),
                n_outer_iterations=len(outer_log),
                n_stations=len(per_st),
            )
        except Exception as exc:
            warnings.warn(
                f"Shell K7 homogenisation failed; using strip section_properties K7: {exc!r}",
                stacklevel=1,
            )
            shell_k7_meta = {"enabled": True, "error": str(exc), "fallback": "section_properties_K7"}
            sec_effective = sec

    stations = stations_from_arrays(np.asarray(sec_effective.station_z, dtype=np.float64), sec_effective.K6, sec_effective.K7)
    k7_diag = np.diagonal(np.asarray(sec_effective.K7, dtype=np.float64), axis1=1, axis2=2)
    mid_idx = int(k7_diag.shape[0] // 2)
    try:
        mid_eigs = np.linalg.eigvalsh(np.asarray(sec_effective.K7[mid_idx], dtype=np.float64))
    except np.linalg.LinAlgError:
        mid_eigs = np.array([], dtype=np.float64)
    run_log.info_event(
        "beam_input_k7_diagnostics",
        diag_min=np.min(k7_diag, axis=0),
        diag_max=np.max(k7_diag, axis=0),
        diag_median=np.median(k7_diag, axis=0),
        mid_station_z=float(np.asarray(sec_effective.station_z, dtype=np.float64)[mid_idx]),
        mid_station_eigvals=mid_eigs,
    )
    analysis = BeamAnalysis.from_blade_geometry(geom, int(n_beam_nodes), stations, span_axis=2)

    model = analysis.model
    n_nodes = int(model.n_nodes)
    n_elem = int(len(model.elements))
    emit_progress_event(progress, "beam_assembly", n_nodes=n_nodes, n_elements=n_elem)
    z_mid = np.asarray([el.z_mid for el in model.elements], dtype=np.float64)

    qy = np.interp(z_mid, inp.loads_r_z_m, inp.q_y_Npm)
    qz = np.interp(z_mid, inp.loads_r_z_m, inp.q_z_Npm)
    mx = np.interp(z_mid, inp.loads_r_z_m, inp.m_x_Nmpm)

    distributed_q = np.zeros((n_elem, 3), dtype=np.float64)
    distributed_q[:, 1] = qy
    distributed_q[:, 2] = qz

    loads = BeamLoads(
        nodal_F=np.zeros((n_nodes, 3), dtype=np.float64),
        nodal_M=np.zeros((n_nodes, 3), dtype=np.float64),
        distributed_q=distributed_q,
        distributed_mz=np.asarray(mx, dtype=np.float64),
        bcs=[BoundaryCondition(0, tuple(range(7)))],
    )
    # Material-only tangent by default; ``full_fd_hessian=True`` + ``project_fd_hessian_spd`` remain opt-in.
    opts = SolverOptions(
        max_iter=200,
        tol_res=7e-2,
        tol_res_rel=6e-3,
        tol_du=1e-6,
        n_gauss=2,
        n_load_steps=72,
        full_fd_hessian=False,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        relax_factor=0.9,
        verbose=False,
        tol_res_rel_rhs=0.035,
        cap_floor_rel=0.055,
        line_search=False,
        extract_buckling=bool(enable_global_buckling),
        n_buckling_modes=int(n_global_buckling_modes),
    )
    res = analysis.solve_static(loads, options=opts)
    run_log.log_iteration(int(res.n_iterations), residual_norm=float(res.residual_norm), converged=bool(res.converged))
    if not bool(res.converged):
        run_log.info_event(
            "beam_solve.not_converged",
            n_iterations=int(res.n_iterations),
            residual_norm=float(res.residual_norm),
            max_iter=int(opts.max_iter),
            hint="Consider mesh/loads/stiffness or SolverOptions max_iter/tol_res in beam_model_impl.",
        )
    emit_progress_event(
        progress,
        "beam_solve",
        converged=bool(res.converged),
        n_iterations=int(res.n_iterations),
        residual_norm=float(res.residual_norm),
    )

    _recovery_artifacts: tuple[object, object] | None = None
    try:
        emit_progress_event(progress, "beam_section_recovery")
        from blade_precompute.global_beam_model.engine.section_recovery import (
            build_beam_section_recovery_artifacts,
            enrich_beam_result_with_section_stress,
        )

        z_sec_beam = np.asarray(sec.station_z, dtype=np.float64)
        _recovery_artifacts = build_beam_section_recovery_artifacts(
            res,
            station_z=z_sec_beam,
            section_results=sec.section_results,
            section_definitions=sec.section_definitions,
        )
        if _recovery_artifacts is not None:
            _rc, _rb = _recovery_artifacts
            res = enrich_beam_result_with_section_stress(
                res,
                station_z=z_sec_beam,
                section_results=sec.section_results,
                section_definitions=sec.section_definitions,
                recovery_cache=_rc,
                recovery_bundle=_rb,
            )
        else:
            res = enrich_beam_result_with_section_stress(
                res,
                station_z=z_sec_beam,
                section_results=sec.section_results,
                section_definitions=sec.section_definitions,
            )
    except Exception as exc:
        warnings.warn(f"Section stress/strain recovery skipped: {exc}", stacklevel=1)

    if save_section_recovery_cache_npz:
        try:
            from blade_precompute.global_beam_model.engine.section_recovery import (
                save_section_recovery_cache_to_npz,
            )

            _cache_npz = _recovery_artifacts[0] if _recovery_artifacts is not None else None
            save_section_recovery_cache_to_npz(
                res,
                station_z=np.asarray(sec.station_z, dtype=np.float64),
                section_results=sec.section_results,
                section_definitions=sec.section_definitions,
                path=out_stage / "section_recovery_cache.npz",
                cache=_cache_npz,
            )
        except Exception as exc:
            warnings.warn(f"Section recovery cache NPZ not written: {exc}", stacklevel=1)

    shell_recovery: dict[str, Any] | None = None
    if enable_shell_recovery_enrichment:
        emit_progress_event(progress, "beam_shell_enrichment")
        layout = orchestration.layout
        if layout.geometry_mode == "multicell" and layout.n_webs > 0:
            try:
                from blade_precompute.global_beam_model.engine.shell_enrichment import (
                    BladeShellEnrichmentInputs,
                    shell_recovery_payload,
                )

                spars_shell = section_shell_spars_from_layout(layout)
                shell_inputs = BladeShellEnrichmentInputs(
                    span_r_z_m=np.asarray(inp.span_r_z_m, dtype=np.float64),
                    naca_m=np.asarray(inp.naca_m, dtype=np.float64),
                    naca_p=np.asarray(inp.naca_p, dtype=np.float64),
                    naca_xx=np.asarray(inp.naca_xx, dtype=np.float64),
                    naca_series=np.asarray(inp.naca_series, dtype=np.int64),
                    chord_m=np.asarray(inp.chord_m, dtype=np.float64),
                )
                shell_recovery = shell_recovery_payload(
                    res,
                    shell_inputs,
                    np.asarray(sec.station_z, dtype=np.float64),
                    spars_shell,
                    n_elements_per_panel=int(shell_recovery_n_elements_per_panel),
                )
            except Exception as exc:
                warnings.warn(f"Shell recovery enrichment skipped: {exc}", stacklevel=1)
                shell_recovery = {"skipped": True, "reason": str(exc)}
        else:
            shell_recovery = {"skipped": True, "reason": "not_multicell_or_no_webs"}

    tip_disp = np.asarray(res.nodal_positions[-1] - model.X_ref[-1], dtype=np.float64)
    beam_payload: dict[str, Any] = {
            "converged": bool(res.converged),
            "n_iterations": int(res.n_iterations),
            "residual_norm": float(res.residual_norm),
            "tip_displacement_m": tip_disp,
            "z_stations_out": np.asarray(res.z_stations_out, dtype=np.float64) if res.z_stations_out is not None else None,
            "resultants": np.asarray(res.resultants, dtype=np.float64),
            "strains": np.asarray(res.strains, dtype=np.float64),
            "nodal_averages": {
                "z_nodal_out": np.asarray(res.z_nodal_out, dtype=np.float64)
                if res.z_nodal_out is not None
                else None,
                "resultants_nodal": np.asarray(res.resultants_nodal, dtype=np.float64)
                if res.resultants_nodal is not None
                else None,
                "strains_nodal": np.asarray(res.strains_nodal, dtype=np.float64)
                if res.strains_nodal is not None
                else None,
            },
            "section_recovery": {
                "z_section_m": np.asarray(res.z_section_recovery, dtype=np.float64)
                if res.z_section_recovery is not None
                else None,
                "ply_stress_voigt_max_gp_Pa": np.asarray(res.section_stress_voigt_gp, dtype=np.float64)
                if res.section_stress_voigt_gp is not None
                else None,
                "ply_stress_voigt_max_nodal_Pa": np.asarray(res.section_stress_voigt_nodal, dtype=np.float64)
                if res.section_stress_voigt_nodal is not None
                else None,
                "laminate_strain_maxabs_gp": np.asarray(res.section_strain_maxabs_gp, dtype=np.float64)
                if res.section_strain_maxabs_gp is not None
                else None,
                "laminate_strain_maxabs_nodal": np.asarray(res.section_strain_maxabs_nodal, dtype=np.float64)
                if res.section_strain_maxabs_nodal is not None
                else None,
                "hashin_fi_max_gp": np.asarray(res.section_hashin_fi_max_gp, dtype=np.float64)
                if res.section_hashin_fi_max_gp is not None
                else None,
                "hashin_fi_max_nodal": np.asarray(res.section_hashin_fi_max_nodal, dtype=np.float64)
                if res.section_hashin_fi_max_nodal is not None
                else None,
                "von_mises_fi_max_gp": np.asarray(res.section_von_mises_fi_max_gp, dtype=np.float64)
                if res.section_von_mises_fi_max_gp is not None
                else None,
                "von_mises_fi_max_nodal": np.asarray(res.section_von_mises_fi_max_nodal, dtype=np.float64)
                if res.section_von_mises_fi_max_nodal is not None
                else None,
                "ply_stress_secframe_voigt_max_gp_Pa": np.asarray(
                    res.section_stress_voigt_secframe_gp, dtype=np.float64
                )
                if res.section_stress_voigt_secframe_gp is not None
                else None,
                "ply_stress_secframe_voigt_max_nodal_Pa": np.asarray(
                    res.section_stress_voigt_secframe_nodal, dtype=np.float64
                )
                if res.section_stress_voigt_secframe_nodal is not None
                else None,
                "d_hashin_fi_dz_gp": np.asarray(res.section_d_hashin_fi_dz_gp, dtype=np.float64)
                if res.section_d_hashin_fi_dz_gp is not None
                else None,
                "d_hashin_fi_dz_nodal": np.asarray(res.section_d_hashin_fi_dz_nodal, dtype=np.float64)
                if res.section_d_hashin_fi_dz_nodal is not None
                else None,
                "hashin_fi_ply_envelope_gp": np.asarray(res.section_hashin_fi_ply_envelope_gp, dtype=np.float64)
                if res.section_hashin_fi_ply_envelope_gp is not None
                else None,
                "hashin_fi_ply_envelope_nodal": np.asarray(
                    res.section_hashin_fi_ply_envelope_nodal, dtype=np.float64
                )
                if res.section_hashin_fi_ply_envelope_nodal is not None
                else None,
            },
            "gbt_beam_export": None,
            "global_buckling": {
                "enabled": bool(enable_global_buckling),
                "n_modes": int(n_global_buckling_modes),
                "lambdas": np.asarray(res.global_buckling_lambdas, dtype=np.float64).tolist()
                if res.global_buckling_lambdas is not None
                else None,
                "modeshapes": np.asarray(res.global_buckling_modeshapes, dtype=np.float64).tolist()
                if res.global_buckling_modeshapes is not None
                else None,
            },
            "shell_k7_homogenisation": shell_k7_meta,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
    }
    if shell_recovery is not None:
        beam_payload["shell_recovery"] = shell_recovery
    result_json = write_json(out_stage / "beam_result.json", beam_payload)
    run_log.log_artefact(result_json, "result_json")
    run_log.log_tensor("global_beam_model.resultants", np.asarray(res.resultants, dtype=np.float64))
    run_log.log_tensor("global_beam_model.strains", np.asarray(res.strains, dtype=np.float64))

    png_paths: list[Path] = []
    if persist_pngs:
        emit_progress_event(
            progress,
            "beam_png",
            n_png_span_samples=int(beam_png_span_samples),
        )
        png_paths = write_beam_model_pngs(
            out_stage, model, res, loads, span_plot_samples=int(beam_png_span_samples)
        )
        for p in png_paths:
            run_log.log_artefact(p, "png")
    write_json(out_stage / "summary.json", {"result_json": result_json, "png_paths": png_paths})
    return BeamModelOutputs(
        result_json=result_json,
        png_paths=png_paths,
        beam_n_iterations=int(res.n_iterations),
        beam_converged=bool(res.converged),
    )


def _station_resultants_for_shell_from_beam(
    bm_out: BeamModelOutputs | None,
    inp_struct: PrecomputeInputs,
) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Map beam element resultants to structural station indices (same convention as main_precompute)."""
    if bm_out is None:
        return {}
    try:
        with open(bm_out.result_json, "r", encoding="utf-8") as f:
            beam_data = json.load(f)
        resultants = np.asarray(beam_data.get("resultants", []), dtype=np.float64)
        z_out = beam_data.get("z_stations_out", None)
        if resultants.ndim != 2 or resultants.shape[1] < 6:
            return {}
        col_N, col_Vy, col_Vz, col_My, col_Mz, col_T = 0, 1, 2, 3, 4, 5
        n_sh = int(np.asarray(inp_struct.span_r_z_m, dtype=np.float64).ravel().shape[0])
        z_struct = np.asarray(inp_struct.span_r_z_m, dtype=np.float64)
        n_elem = int(resultants.shape[0])
        if z_out is not None:
            z_elem = np.asarray(z_out, dtype=np.float64).ravel()
            if z_elem.size != n_elem:
                z_elem = np.linspace(float(z_struct[0]), float(z_struct[-1]), n_elem)
        else:
            z_elem = np.linspace(float(z_struct[0]), float(z_struct[-1]), n_elem)
        out_map: dict[int, tuple[float, float, float, float, float, float]] = {}
        for si in range(n_sh):
            zi = float(z_struct[si])
            idx = int(np.argmin(np.abs(z_elem - zi)))
            row = resultants[idx]
            out_map[si] = (
                float(row[col_N]),
                float(row[col_Vy]),
                float(row[col_Vz]),
                float(row[col_My]),
                float(row[col_Mz]),
                float(row[col_T]),
            )
        return out_map
    except Exception:
        return {}


def run_pipeline_snapshot_to_dir(
    snapshot_root: Path,
    *,
    inp_struct: PrecomputeInputs,
    inp_geom: PrecomputeInputs,
    bg: Any,
    orchestration: PrecomputeOrchestrationContext,
    blade_yaml: Path,
    bundle: Mapping[str, Any],
    dpi: int,
    persist_pngs: bool,
    loads_provenance: str,
) -> None:
    """Run geometry → properties → beam → shell under ``snapshot_root`` (``iter_NNNN`` layout).

    ``bundle`` must contain grid metas and knobs (see ``SectionOptimisationParams.iteration_snapshot_grid_bundle``).
    ``iter_0000`` = initial evaluate; ``iter_0001``+ = SLSQP callback order (1-based index in dirname).
    Callers typically pass ``snapshot_root = <job>/section_optimisation/iter_NNNN``.
    """
    snapshot_root = Path(snapshot_root).resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    sg_meta = bundle["section_geometry"]
    sp_meta = bundle["section_properties"]
    bm_meta = bundle["beam"]
    sh_meta = bundle["section_shell"]
    section_plot_station_spec = str(bundle["section_plot_station_spec"])
    section_solve_n_workers = int(bundle["section_solve_n_workers"])
    n_beam_nodes = int(bundle["n_beam_nodes"])
    enable_shell_recovery = bool(bundle["enable_shell_recovery_enrichment"])
    shell_recovery_n_elements = int(bundle["shell_recovery_n_elements_per_panel"])
    beam_png_span_samples = int(bundle["beam_png_span_samples"])
    n_elements_per_panel = int(bundle["n_elements_per_panel"])
    use_mitc4_v2 = bool(bundle["use_mitc4_v2_path"])
    save_sec_recovery_npz = bool(bundle.get("save_section_recovery_cache_npz", False))

    section_geometry_impl(
        inp_struct,
        snapshot_root,
        section_plot_station_spec=section_plot_station_spec,
        orchestration=orchestration,
        grid_meta=sg_meta,
        section_solve_n_workers=section_solve_n_workers,
        persist_pngs=persist_pngs,
        run_log=None,
    )
    sp_out = section_properties_impl(
        inp_struct,
        snapshot_root,
        blade_yaml=blade_yaml,
        section_plot_station_spec=section_plot_station_spec,
        orchestration=orchestration,
        bg_override=bg,
        grid_meta=sp_meta,
        section_solve_n_workers=section_solve_n_workers,
        persist_pngs=persist_pngs,
        run_log=None,
    )
    bm_out = beam_model_impl(
        inp_geom,
        sp_out,
        snapshot_root,
        blade_yaml=blade_yaml,
        n_beam_nodes=n_beam_nodes,
        orchestration=orchestration,
        save_section_recovery_cache_npz=save_sec_recovery_npz,
        bg_override=bg,
        grid_meta=bm_meta,
        enable_shell_recovery_enrichment=enable_shell_recovery,
        shell_recovery_n_elements_per_panel=shell_recovery_n_elements,
        beam_png_span_samples=beam_png_span_samples,
        persist_pngs=persist_pngs,
        run_log=None,
        progress=None,
    )
    station_res = _station_resultants_for_shell_from_beam(bm_out, inp_struct)
    section_shell_model_impl(
        inp_struct,
        snapshot_root,
        section_plot_station_spec=section_plot_station_spec,
        orchestration=orchestration,
        n_elements_per_panel=n_elements_per_panel,
        dpi=int(dpi),
        grid_meta=sh_meta,
        station_resultants=station_res if station_res else None,
        persist_pngs=persist_pngs,
        loads_provenance=loads_provenance,
        use_mitc4_v2_path=use_mitc4_v2,
        run_log=None,
    )


def section_optimisation_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    orchestration: PrecomputeOrchestrationContext,
    run_blade_optimizer: bool = False,
    optimization_objective: OptimisationObjective = "min_mass",
    optimizer_max_iter: int = 120,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
    design_n_workers: int = 1,
    section_properties: SectionPropertiesOutputs | None = None,
    seed_section_properties: bool = True,
    ks_rho: float = 35.0,
    # Group J knobs (J.6)
    enable_panel_buckling: bool = False,
    ks_rho_buckling: float = 25.0,
    enable_global_buckling: bool = False,
    global_buckling_lambda_min: float = 1.5,
    n_global_buckling_modes: int = 5,
    # Group L knobs (L.6, L.9)
    orientation_bounds: Any | None = None,
    enforce_spanwise_monotone: bool = True,
    # Fix 1: stress projection diagnostics
    debug_stress_projection: bool = False,
    # Group H: global beam in-loop (distributed loads) + MITC4 stress (optional)
    beam_driver: str = "prescribed",
    distributed_loads_inp: Any | None = None,
    axial_loading: Any | None = None,
    n_beam_nodes: int = 50,
    stress_recovery: str = "mitc4",
    mitc4_n_elements_per_panel: int = 10,
    optimizer_method: str = "SLSQP",
    optimizer_ftol: float = 1e-5,
    optimizer_n_restarts: int = 0,
    optimizer_multistart_seed: int | None = None,
    iteration_dump_npz: bool = False,
    iteration_hotspot_k: int = 10,
    iteration_emit_schema: bool = True,
    run_log: RunLogger | None = None,
    progress: Any | None = None,
    iteration_pipeline_snapshots: bool = False,
    iteration_snapshot_dpi: int = 96,
    iteration_snapshot_pngs: bool = True,
    iteration_snapshot_max: int | None = None,
    iteration_snapshot_stride: int = 1,
    iteration_snapshot_grid_bundle: Mapping[str, Any] | None = None,
    iteration_snapshot_beam_inp: PrecomputeInputs | None = None,
) -> SectionOptimisationOutputs:
    out_stage = (out_dir / "section_optimisation").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)
    if run_log is None:
        run_log = get_run_logger(
            package="section_optimisation",
            job_dir=out_dir,
            dump_level=inp.log_dump_level,
        )

    from blade_precompute.section_optimisation import BladeOptimizer
    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.global_beam_model.engine.axial_loading import (
        AxialLoadingConfig,
        axial_force_distribution,
        q_x_distributed,
    )
    from blade_precompute.section_optimisation.core.types import (
        DesignProblem,
        DistributedLoadCurves,
        ExtremeLoads,
    )

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    extreme_raw = BladeDesignProblem.load_extreme_loads_dat(inp.extreme_loads_path, z_geometry=None)

    z_geom = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_raw = np.asarray(extreme_raw.z_stations, dtype=np.float64).ravel()
    if z_raw.size < 2:
        raise ValueError("ExtremeLoads must contain at least two z stations.")

    def _interp(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return np.interp(z_geom, z_raw, a)

    N_tab = _interp(extreme_raw.N)
    if (
        axial_loading is not None
        and isinstance(axial_loading, AxialLoadingConfig)
        and bool(axial_loading.enabled)
        and section_properties is not None
    ):
        z_st = np.asarray(section_properties.station_z, dtype=np.float64).ravel()
        mu_st = np.array(
            [float(r.mass_per_length) for r in section_properties.section_results],
            dtype=np.float64,
        )
        span_z = np.asarray(inp.span_r_z_m, dtype=np.float64).ravel()
        rad_m = np.asarray(inp.radial_r_m, dtype=np.float64).ravel()
        mu_geom = np.interp(z_geom, z_st, mu_st)
        r_geom = np.interp(z_geom, span_z, rad_m)
        qx_g = q_x_distributed(
            z_geom.astype(np.float64), r_geom, mu_geom, axial_loading
        )
        N_tab = axial_force_distribution(z_geom.astype(np.float64), qx_g)

    extreme = ExtremeLoads(
        z_stations=z_geom,
        N=N_tab,
        Vy=_interp(extreme_raw.Vy),
        Vz=_interp(extreme_raw.Vz),
        My=_interp(extreme_raw.My),
        Mz=_interp(extreme_raw.Mz),
        T=_interp(extreme_raw.T),
        B=_interp(extreme_raw.bimoment()),
    )
    dist_curves: DistributedLoadCurves | None = None
    _dlin = distributed_loads_inp
    if str(beam_driver).lower() in ("global_beam", "coupled_fe") and _dlin is not None:
        qx_f: NDArray[np.float64] | None = None
        if (
            axial_loading is not None
            and isinstance(axial_loading, AxialLoadingConfig)
            and bool(axial_loading.enabled)
            and section_properties is not None
        ):
            z_st = np.asarray(section_properties.station_z, dtype=np.float64).ravel()
            mu_st = np.array(
                [float(r.mass_per_length) for r in section_properties.section_results],
                dtype=np.float64,
            )
            zf = np.asarray(_dlin.loads_r_z_m, dtype=np.float64).ravel()
            span_d = np.asarray(_dlin.span_r_z_m, dtype=np.float64).ravel()
            rad_d = np.asarray(_dlin.radial_r_m, dtype=np.float64).ravel()
            mu_f = np.interp(zf, z_st, mu_st)
            r_f = np.interp(zf, span_d, rad_d)
            qx_f = q_x_distributed(zf, r_f, mu_f, axial_loading)
        dist_curves = DistributedLoadCurves(
            loads_r_z_m=np.asarray(_dlin.loads_r_z_m, dtype=np.float64),
            q_y_Npm=np.asarray(_dlin.q_y_Npm, dtype=np.float64),
            q_z_Npm=np.asarray(_dlin.q_z_Npm, dtype=np.float64),
            m_x_Nmpm=np.asarray(_dlin.m_x_Nmpm, dtype=np.float64),
            q_x_Npm=qx_f,
        )
    problem = DesignProblem(
        blade_geometry=bg,
        extreme_loads=extreme,
        solver=None,
        objective=optimization_objective,
        ks_rho=float(ks_rho),
        n_workers=int(design_n_workers),
        # Group J (J.6)
        enable_panel_buckling=bool(enable_panel_buckling),
        ks_rho_buckling=float(ks_rho_buckling),
        enable_global_buckling=bool(enable_global_buckling),
        global_buckling_lambda_min=float(global_buckling_lambda_min),
        n_global_buckling_modes=int(n_global_buckling_modes),
        # Group L (L.6, L.9)
        orientation_bounds=orientation_bounds,
        enforce_spanwise_monotone=bool(enforce_spanwise_monotone),
        debug_stress_projection=bool(debug_stress_projection),
        beam_driver=str(beam_driver),
        distributed_loads=dist_curves,
        axial_loading=axial_loading if isinstance(axial_loading, AxialLoadingConfig) else None,
        n_beam_nodes=int(n_beam_nodes),
        stress_recovery=str(stress_recovery),  # type: ignore[arg-type]
        mitc4_n_elements_per_panel=int(mitc4_n_elements_per_panel),
        optimizer_method=str(optimizer_method),
        optimizer_ftol=float(optimizer_ftol),
        optimizer_n_restarts=int(optimizer_n_restarts),
        optimizer_multistart_seed=optimizer_multistart_seed,
        iteration_dump_npz=bool(iteration_dump_npz),
        iteration_hotspot_k=int(iteration_hotspot_k),
        iteration_emit_schema=bool(iteration_emit_schema),
    )
    sizing = BladeDesignProblem(problem)

    if bool(iteration_pipeline_snapshots) and iteration_snapshot_grid_bundle is not None:
        run_log.info_event(
            "iteration_snapshots.enabled",
            snapshot_parent=str(out_stage),
            pattern="iter_NNNN",
            note="Written after initial design eval and each optimiser callback; absent if this stage never runs.",
        )

    dv0 = default_dv0(int(bg.z_stations.shape[0]))
    if seed_section_properties and section_properties is not None:
        n_bg = int(np.asarray(bg.z_stations, dtype=np.float64).ravel().shape[0])
        if n_bg != len(section_properties.section_results):
            raise ValueError(
                f"section_properties has {len(section_properties.section_results)} station results, "
                f"blade geometry has {n_bg} stations; cannot seed design evaluator."
            )
        sizing.seed_stations(
            dv0, cast(Sequence[SectionSolveResult], section_properties.section_results)
        )

    def _beam_converged_from_eval(ev: Any) -> bool | None:
        bs = getattr(ev, "beam_state", None)
        if bs is None:
            return None
        if hasattr(bs, "converged"):
            return bool(getattr(bs, "converged"))
        return None

    if progress is not None and getattr(progress, "enabled", True):
        progress.phase_start("design_initial_evaluate")
    ev0 = sizing.evaluate(dv0)
    if progress is not None and getattr(progress, "enabled", True):
        progress.phase_end(
            "design_initial_evaluate",
            mass_kg=float(ev0.mass),
            max_fi_hashin=float(ev0.max_fi_hashin),
            max_fi_vm=float(ev0.max_fi_vm),
            beam_converged=_beam_converged_from_eval(ev0),
        )
    run_log.info_event(
        "evaluation.initial",
        mass_kg=float(ev0.mass),
        stiffness_metric=float(ev0.stiffness_metric),
        max_fi_hashin=float(ev0.max_fi_hashin),
        max_fi_vm=float(ev0.max_fi_vm),
    )

    _beam_snap_inp = (
        iteration_snapshot_beam_inp
        if iteration_snapshot_beam_inp is not None
        else (distributed_loads_inp if distributed_loads_inp is not None else inp)
    )

    def _write_iteration_snapshot(iter_index: int, dv: Any) -> None:
        if not iteration_pipeline_snapshots or iteration_snapshot_grid_bundle is None:
            return
        if iteration_snapshot_max is not None and iter_index >= iteration_snapshot_max:
            return
        if (
            iter_index > 0
            and iteration_snapshot_stride > 1
            and (iter_index % iteration_snapshot_stride) != 0
        ):
            return
        bg_snap = apply_dv_to_bg(bg, dv)
        # Under ``section_optimisation/`` so iteration artefacts stay grouped with the opt stage
        # (not only at the job root, where ``iter_*`` is easy to overlook).
        root = (out_stage / f"iter_{iter_index:04d}").resolve()
        if progress is not None and getattr(progress, "enabled", True):
            progress.phase_start("iteration_pipeline_snapshot", iter_index=int(iter_index))
        try:
            run_pipeline_snapshot_to_dir(
                root,
                inp_struct=inp,
                inp_geom=_beam_snap_inp,
                bg=bg_snap,
                orchestration=orchestration,
                blade_yaml=blade_yaml,
                bundle=iteration_snapshot_grid_bundle,
                dpi=int(iteration_snapshot_dpi),
                persist_pngs=bool(iteration_snapshot_pngs),
                loads_provenance=f"snapshot_iter_{iter_index:04d}",
            )
        finally:
            if progress is not None and getattr(progress, "enabled", True):
                progress.phase_end("iteration_pipeline_snapshot", iter_index=int(iter_index))

    _write_iteration_snapshot(0, dv0)

    eval_payload: dict[str, Any] = {
        "optimization_objective": optimization_objective,
        "blade_optimizer_ran": bool(run_blade_optimizer),
        "grid": dict(grid_meta) if grid_meta is not None else None,
        **design_eval_payload(ev0, dv0),
    }

    opt_res = None
    history_json: Path | None = None
    if run_blade_optimizer:
        if bool(iteration_emit_schema):
            from blade_precompute.section_optimisation.engine.iteration_report import (
                write_iteration_payload_schema,
            )

            _schema_path = out_stage / "iteration_payload_schema.json"
            write_iteration_payload_schema(_schema_path)
            run_log.log_artefact(_schema_path, "iteration_payload_schema")
        def _snap_cb(ev: Any, dv: Any, iter_index: int) -> None:
            _write_iteration_snapshot(iter_index, dv)

        opt = BladeOptimizer(
            sizing.problem,
            method=str(sizing.problem.optimizer_method),
            options={
                "maxiter": int(optimizer_max_iter),
                "ftol": float(sizing.problem.optimizer_ftol),
                "disp": False,
            },
            evaluator=sizing.evaluator,
            run_log=run_log,
            progress=progress,
            on_after_evaluation=_snap_cb,
        )
        if progress is not None and getattr(progress, "enabled", True):
            progress.phase_start("blade_optimizer_slsqp", max_iter=int(optimizer_max_iter))
        opt_res = opt.run(dv0)
        if progress is not None and getattr(progress, "enabled", True):
            progress.phase_end(
                "blade_optimizer_slsqp",
                success=bool(opt_res.success),
                n_iter=int(opt_res.n_iter),
                message=str(opt_res.message),
            )
        ev_opt = opt_res.evaluations[-1] if opt_res.evaluations else sizing.evaluate(opt_res.dv_opt)
        eval_payload["optimised"] = design_eval_payload(ev_opt, opt_res.dv_opt)
        eval_payload["blade_optimizer"] = {
            "success": bool(opt_res.success),
            "message": str(opt_res.message),
            "n_iter": int(opt_res.n_iter),
            "max_iter": int(optimizer_max_iter),
            "scipy_method": str(sizing.problem.optimizer_method),
            "optimizer_ftol": float(sizing.problem.optimizer_ftol),
            "optimizer_n_restarts": int(sizing.problem.optimizer_n_restarts),
            "optimizer_multistart_seed": sizing.problem.optimizer_multistart_seed,
            "stress_recovery": str(sizing.problem.stress_recovery),
        }
        run_log.info_event(
            "optimizer.final",
            success=bool(opt_res.success),
            n_iter=int(opt_res.n_iter),
            message=str(opt_res.message),
        )
        eval_payload["optimizer_history"] = [
            {
                "iteration": int(i + 1),
                "mass_kg": float(ev.mass),
                "stiffness_metric": float(ev.stiffness_metric),
                "specific_stiffness": float(ev.stiffness_metric / max(ev.mass, 1e-30)),
                "max_fi_hashin": float(ev.max_fi_hashin),
                "max_fi_vm": float(ev.max_fi_vm),
            }
            for i, ev in enumerate(opt_res.evaluations)
        ]
        history_json = write_json(
            out_stage / "optimizer_convergence_history.json",
            eval_payload["optimizer_history"],
        )
        run_log.log_artefact(history_json, "optimizer_history_json")

    result_json = write_json(out_stage / "design_eval.json", eval_payload)
    run_log.log_artefact(result_json, "result_json")

    png_paths: list[Path] = []
    try:
        z = np.asarray(bg.z_stations, dtype=np.float64)
        composite_subcomp_names: list[str] | None = None
        try:
            _ref0 = sizing.evaluator._caches[0].result
            if _ref0 is not None and getattr(_ref0, "composite_subcomp_names", None) is not None:
                composite_subcomp_names = [str(x) for x in _ref0.composite_subcomp_names]
        except Exception:
            composite_subcomp_names = None
        ev_opt_plot: Any | None = None
        if opt_res is not None and getattr(opt_res, "evaluations", None):
            _evs = opt_res.evaluations
            if _evs:
                ev_opt_plot = _evs[-1]
        png_paths = write_section_optimisation_pngs(
            out_stage,
            z,
            dv0,
            opt_res,
            ev0=ev0,
            ev_opt=ev_opt_plot,
            problem=sizing.problem,
            composite_subcomp_names=composite_subcomp_names,
        )
        for p in png_paths:
            run_log.log_artefact(p, "png")
    except ImportError:
        pass

    write_json(
        out_stage / "summary.json",
        {
            "result_json": result_json,
            "optimizer_history_json": history_json,
            "png_paths": png_paths,
            # K.1: single-LC assumption provenance
            "assumptions": {
                "single_load_case": True,
                "hydrodynamic_load_invariant_under_sls_tip": True,
                "note": (
                    "One extreme-load .dat file per run. "
                    "SLS tip deflection evaluated against same load case."
                ),
            },
            # I.11: K7 stiffness stack provenance
            "k7_stiffness_provenance": {
                "pre_loop_k7_stack": "section_properties stage (strip midsurface solve at dv0)",
                "in_loop_k7_stack": (
                    "section_properties stage (seeded from pre-loop) unless "
                    "beam_driver='global_beam' (Group H), in which case shell "
                    "homogenisation K7 replaces strip K7 per SLSQP evaluation."
                ),
                "stiffness_solver": getattr(problem, "beam_driver", "prescribed"),
                "stress_recovery": getattr(problem, "stress_recovery", "mitc4"),
            },
            # J.6: buckling knob provenance
            "buckling_config": {
                "enable_panel_buckling": bool(enable_panel_buckling),
                "ks_rho_buckling": float(ks_rho_buckling),
                "enable_global_buckling": bool(enable_global_buckling),
                "global_buckling_lambda_min": float(global_buckling_lambda_min),
                "n_global_buckling_modes": int(n_global_buckling_modes),
            },
            # L.6: orientation bounds provenance
            "orientation_config": {
                "orientation_bounds_set": orientation_bounds is not None,
                "enforce_spanwise_monotone": bool(enforce_spanwise_monotone),
            },
            "iteration_logging": {
                "iteration_dump_npz": bool(iteration_dump_npz),
                "iteration_hotspot_k": int(iteration_hotspot_k),
                "iteration_emit_schema": bool(iteration_emit_schema),
                "iteration_payload_schema_json": str(out_stage / "iteration_payload_schema.json")
                if run_blade_optimizer and bool(iteration_emit_schema)
                else None,
            },
        },
    )
    optimizer_n_iter: int | None = int(opt_res.n_iter) if opt_res is not None else None
    return SectionOptimisationOutputs(
        result_json=result_json,
        png_paths=png_paths,
        optimizer_ran=bool(run_blade_optimizer),
        optimizer_n_iter=optimizer_n_iter,
    )
