"""Precompute stage implementations (geometry, properties, global beam, optimisation)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

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
)
from blade_precompute.orchestration.precompute.grid import station_indices
from blade_precompute.orchestration.precompute.jsonutil import write_json
from blade_precompute.orchestration.precompute.vis import (
    plot_section_properties_station,
    write_beam_model_pngs,
    write_section_optimisation_pngs,
)
from blade_precompute.section_optimisation.core.types import OptimizationObjective

_REPO_ROOT = Path(__file__).resolve().parents[3]


def naca4(m: float, p: float, xx: float) -> str:
    mi = int(np.clip(int(round(float(m))), 0, 9))
    pi = int(np.clip(int(round(float(p))), 0, 9))
    xxi = int(np.clip(int(round(float(xx))), 0, 99))
    return f"{mi:d}{pi:d}{xxi:02d}"


def default_dv0(n_station: int):
    from blade_precompute.section_optimisation.core.types import DesignVector

    n = int(n_station)
    return DesignVector(
        t_skin=np.full(n, 0.012, dtype=np.float64),
        t_cap=np.full(n, 0.050, dtype=np.float64),
        t_web=np.full(n, 0.015, dtype=np.float64),
    )


def design_eval_payload(ev: Any, dv: Any) -> dict[str, Any]:
    return {
        "mass_kg": float(ev.mass),
        "stiffness_metric_int_trace_k7": float(ev.stiffness_metric),
        "stiffness_metric_over_mass": float(ev.stiffness_metric / max(ev.mass, 1e-300)),
        "max_fi_tw": float(ev.max_fi_tw),
        "max_fi_vm": float(ev.max_fi_vm),
        "max_fi_delam": float(ev.max_fi_delam) if ev.max_fi_delam is not None else None,
        "dv": dv,
    }


def section_geometry_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionGeometryOutputs:
    out_stage = (out_dir / "section_geometry").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    sg_sys_path = (_REPO_ROOT / "blade_precompute" / "section_geometry").resolve()
    added_sg_path = False
    if str(sg_sys_path) not in sys.path:
        sys.path.insert(0, str(sg_sys_path))
        added_sg_path = True
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
    finally:
        if added_sg_path:
            try:
                sys.path.remove(str(sg_sys_path))
            except ValueError:
                pass

    idx = station_indices(int(inp.span_r_z_m.shape[0]), plot_station_spec)
    png_paths: list[Path] = []
    geometry_report_json_paths: list[Path] = []
    rz_used: list[float] = []

    for i in idx:
        rz = float(inp.span_r_z_m[i])
        chord = float(inp.chord_m[i])
        code = naca4(inp.naca_m[i], inp.naca_p[i], inp.naca_xx[i])
        airfoil = AirfoilSDF.from_naca(code, chord=chord)
        twist_rad = float(np.deg2rad(inp.twist_deg[i]))
        section = build_section_view(airfoil, orchestration.layout, twist_angle_rad=twist_rad)
        airfoil_b = airfoil.rotate(twist_rad) if abs(twist_rad) > 1e-10 else airfoil
        grid = SDFGrid.from_airfoil(airfoil_b, nx=512, ny=220)

        tag = f"i{i:03d}_rz{rz:.3f}"
        props_json = (out_stage / f"geometry_report_{tag}.json").resolve()
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
        phi0 = grid.eval(section[first_label])
        assert_grid_phi_finite(phi0)
        SectionPropertiesReport(section, grid).to_json(props_json, job_meta=job_meta)
        geometry_report_json_paths.append(props_json)

        if plot_section is not None:
            fig, _ = plot_section(section, grid, title=f"section_geometry: NACA{code}, chord={chord:.3g} @ r_z={rz:.3g} m")
            png = (out_stage / f"section_{tag}.png").resolve()
            fig.savefig(png, dpi=170, bbox_inches="tight")
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
            png_paths.append(png)

        rz_used.append(rz)

    write_json(
        out_stage / "summary.json",
        {
            "stations": [{"i": int(i), "r_z_m": float(inp.span_r_z_m[i])} for i in idx],
            "png_paths": png_paths,
            "geometry_report_json_paths": geometry_report_json_paths,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    return SectionGeometryOutputs(
        station_indices=idx,
        station_r_z_m=rz_used,
        png_paths=png_paths,
        geometry_report_json_paths=geometry_report_json_paths,
    )


def section_properties_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    plot_station_spec: str,
    orchestration: PrecomputeOrchestrationContext,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionPropertiesOutputs:
    out_stage = (out_dir / "section_properties").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
    from blade_precompute.section_properties.api import AnalysisConfig, SectionAnalysis
    from blade_precompute.section_properties.core.types import SectionSolveResult
    from blade_precompute.section_properties.io.section_solve_bundle import save_section_solve_stations_bundle

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    dv0 = default_dv0(int(bg.z_stations.shape[0]))
    section_defs = SectionBuilder.build(dv0, bg)

    z = np.asarray([float(sd.station_z) for sd in section_defs], dtype=np.float64)

    analysis = SectionAnalysis(config=AnalysisConfig(run_panel_buckling=False, merge_tolerance=1e-6))
    results: list[SectionSolveResult] = []
    for sd in section_defs:
        results.append(analysis.solve(sd))

    n = len(results)
    K6 = np.stack([np.asarray(r.K6, dtype=np.float64) for r in results], axis=0).reshape(n, 6, 6)
    K7 = np.stack([np.asarray(r.K7, dtype=np.float64) for r in results], axis=0).reshape(n, 7, 7)

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
            "blade_yaml": blade_yaml.resolve(),
            "n_station": int(n),
            "stations": summary_rows,
            "K6": K6,
            "K7": K7,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    png_paths: list[Path] = []
    for i in station_indices(int(z.shape[0]), plot_station_spec):
        out_png = (out_stage / f"section_station_i{i:03d}_z{float(z[i]):.3f}.png").resolve()
        plot_section_properties_station(section_defs[i], results[i], out_png)
        png_paths.append(out_png)

    bundle_meta = save_section_solve_stations_bundle(out_stage, z, results)
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
) -> BeamModelOutputs:
    out_stage = (out_dir / "global_beam_model").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

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
        return BeamModelOutputs(result_json=stub, png_paths=[])

    geom = BladeGeometry(
        z_stations=np.asarray(bg.z_stations, dtype=np.float64),
        r_ref=np.asarray(bg.r_ref, dtype=np.float64),
        kappa0=np.asarray(bg.kappa0, dtype=np.float64),
        tau0=np.asarray(bg.tau0, dtype=np.float64),
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
    stations = stations_from_arrays(np.asarray(sec.station_z, dtype=np.float64), sec.K6, sec.K7)
    analysis = BeamAnalysis.from_blade_geometry(geom, int(n_beam_nodes), stations, span_axis=2)

    model = analysis.model
    n_nodes = int(model.n_nodes)
    n_elem = int(len(model.elements))
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
    opts = SolverOptions(
        max_iter=70,
        tol_res=5e-2,
        tol_res_rel=5e-3,
        tol_du=1e-7,
        n_gauss=2,
        n_load_steps=18,
        spin_stabilization=1e-5,
        warping_stabilization=1e-3,
        verbose=False,
    )
    res = analysis.solve_static(loads, options=opts)

    try:
        from blade_precompute.global_beam_model.engine.section_recovery import enrich_beam_result_with_section_stress

        res = enrich_beam_result_with_section_stress(
            res,
            station_z=np.asarray(sec.station_z, dtype=np.float64),
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

            save_section_recovery_cache_to_npz(
                res,
                station_z=np.asarray(sec.station_z, dtype=np.float64),
                section_results=sec.section_results,
                section_definitions=sec.section_definitions,
                path=out_stage / "section_recovery_cache.npz",
            )
        except Exception as exc:
            warnings.warn(f"Section recovery cache NPZ not written: {exc}", stacklevel=1)

    tip_disp = np.asarray(res.nodal_positions[-1] - model.X_ref[-1], dtype=np.float64)
    result_json = write_json(
        out_stage / "beam_result.json",
        {
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
                "tsai_wu_fi_max_gp": np.asarray(res.section_tsai_wu_fi_max_gp, dtype=np.float64)
                if res.section_tsai_wu_fi_max_gp is not None
                else None,
                "tsai_wu_fi_max_nodal": np.asarray(res.section_tsai_wu_fi_max_nodal, dtype=np.float64)
                if res.section_tsai_wu_fi_max_nodal is not None
                else None,
                "von_mises_fi_max_gp": np.asarray(res.section_von_mises_fi_max_gp, dtype=np.float64)
                if res.section_von_mises_fi_max_gp is not None
                else None,
                "von_mises_fi_max_nodal": np.asarray(res.section_von_mises_fi_max_nodal, dtype=np.float64)
                if res.section_von_mises_fi_max_nodal is not None
                else None,
                "delamination_fi_max_gp": np.asarray(res.section_delamination_fi_max_gp, dtype=np.float64)
                if res.section_delamination_fi_max_gp is not None
                else None,
                "delamination_fi_max_nodal": np.asarray(res.section_delamination_fi_max_nodal, dtype=np.float64)
                if res.section_delamination_fi_max_nodal is not None
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
                "d_tsai_wu_fi_dz_gp": np.asarray(res.section_d_tsai_wu_fi_dz_gp, dtype=np.float64)
                if res.section_d_tsai_wu_fi_dz_gp is not None
                else None,
                "d_tsai_wu_fi_dz_nodal": np.asarray(res.section_d_tsai_wu_fi_dz_nodal, dtype=np.float64)
                if res.section_d_tsai_wu_fi_dz_nodal is not None
                else None,
                "tsai_wu_fi_ply_envelope_gp": np.asarray(res.section_tsai_wu_fi_ply_envelope_gp, dtype=np.float64)
                if res.section_tsai_wu_fi_ply_envelope_gp is not None
                else None,
                "tsai_wu_fi_ply_envelope_nodal": np.asarray(
                    res.section_tsai_wu_fi_ply_envelope_nodal, dtype=np.float64
                )
                if res.section_tsai_wu_fi_ply_envelope_nodal is not None
                else None,
            },
            "gbt_beam_export": None,
            "grid": dict(grid_meta) if grid_meta is not None else None,
            "orchestration": orchestration.job_meta(),
        },
    )

    png_paths = write_beam_model_pngs(out_stage, model, res, loads)
    write_json(out_stage / "summary.json", {"result_json": result_json, "png_paths": png_paths})
    return BeamModelOutputs(result_json=result_json, png_paths=png_paths)


def section_optimisation_impl(
    inp: PrecomputeInputs,
    out_dir: Path,
    *,
    blade_yaml: Path,
    orchestration: PrecomputeOrchestrationContext,
    run_blade_optimizer: bool = False,
    optimization_objective: OptimizationObjective = "min_mass",
    optimizer_max_iter: int = 120,
    bg_override: Any | None = None,
    grid_meta: Mapping[str, Any] | None = None,
) -> SectionOptimisationOutputs:
    out_stage = (out_dir / "section_optimisation").resolve()
    out_stage.mkdir(parents=True, exist_ok=True)

    from blade_precompute.section_optimisation import BladeOptimizer
    from blade_precompute.section_optimisation.api import BladeDesignProblem
    from blade_precompute.section_optimisation.core.types import DesignProblem, ExtremeLoads

    bg = bg_override if bg_override is not None else BladeDesignProblem.load_geometry(blade_yaml)
    extreme_raw = BladeDesignProblem.load_extreme_loads_dat(inp.extreme_loads_path, z_geometry=None)

    z_geom = np.asarray(bg.z_stations, dtype=np.float64).ravel()
    z_raw = np.asarray(extreme_raw.z_stations, dtype=np.float64).ravel()
    if z_raw.size < 2:
        raise ValueError("ExtremeLoads must contain at least two z stations.")

    def _interp(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        return np.interp(z_geom, z_raw, a)

    extreme = ExtremeLoads(
        z_stations=z_geom,
        N=_interp(extreme_raw.N),
        Vy=_interp(extreme_raw.Vy),
        Vz=_interp(extreme_raw.Vz),
        My=_interp(extreme_raw.My),
        Mz=_interp(extreme_raw.Mz),
        T=_interp(extreme_raw.T),
        B=_interp(extreme_raw.bimoment()),
    )
    problem = DesignProblem(
        blade_geometry=bg,
        extreme_loads=extreme,
        solver=None,
        objective=optimization_objective,
        ks_rho=35.0,
        enable_tier3_delam=False,
        n_workers=1,
    )
    sizing = BladeDesignProblem(problem)

    dv0 = default_dv0(int(bg.z_stations.shape[0]))
    ev0 = sizing.evaluate(dv0)

    eval_payload: dict[str, Any] = {
        "optimization_objective": optimization_objective,
        "blade_optimizer_ran": bool(run_blade_optimizer),
        "grid": dict(grid_meta) if grid_meta is not None else None,
        **design_eval_payload(ev0, dv0),
    }

    opt_res = None
    if run_blade_optimizer:
        opt = BladeOptimizer(
            sizing.problem,
            options={"maxiter": int(optimizer_max_iter), "ftol": 1e-5, "disp": False},
        )
        opt_res = opt.run(dv0)
        ev_opt = opt_res.evaluations[-1] if opt_res.evaluations else sizing.evaluate(opt_res.dv_opt)
        eval_payload["optimised"] = design_eval_payload(ev_opt, opt_res.dv_opt)
        eval_payload["blade_optimizer"] = {
            "success": bool(opt_res.success),
            "message": str(opt_res.message),
            "n_iter": int(opt_res.n_iter),
            "max_iter": int(optimizer_max_iter),
        }

    result_json = write_json(out_stage / "design_eval.json", eval_payload)

    png_paths: list[Path] = []
    try:
        z = np.asarray(bg.z_stations, dtype=np.float64)
        png_paths = write_section_optimisation_pngs(out_stage, z, dv0, opt_res)
    except ImportError:
        pass

    write_json(out_stage / "summary.json", {"result_json": result_json, "png_paths": png_paths})
    return SectionOptimisationOutputs(result_json=result_json, png_paths=png_paths)
