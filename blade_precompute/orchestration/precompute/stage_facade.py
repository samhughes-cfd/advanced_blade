"""Stage orchestrator facades: execute / get_results."""

from __future__ import annotations

from typing import Any

from blade_precompute._utils.run_logging import get_run_logger
from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    BeamModelParams,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionOptimisationOutputs,
    SectionOptimisationParams,
    SectionPropertiesOutputs,
    SectionPropertiesParams,
    SectionShellModelOutputs,
    SectionShellModelParams,
)
from blade_precompute.orchestration.precompute.stages import (
    beam_model_impl,
    section_geometry_impl,
    section_optimisation_impl,
    section_properties_impl,
    section_shell_model_impl,
)


def _run_log_mirror_kwargs(params: Any) -> dict[str, Any]:
    """Optional ``progress.jsonl`` mirror of selected RunLogger INFO events."""
    from blade_precompute._utils.job_progress import mirror_run_log_progress_from_env

    p = getattr(params, "progress", None)
    inp = getattr(params, "inp", None)
    want = mirror_run_log_progress_from_env()
    if inp is not None:
        want = want or bool(getattr(inp, "live_progress_mirror_run_log", False))
    active = bool(want and p is not None and getattr(p, "enabled", True))
    return {"progress_reporter": p if active else None, "mirror_progress_jsonl": active}


class _StageBase:
    def __init__(self) -> None:
        self._executed: bool = False
        self._results: Any | None = None

    def get_results(self) -> Any:
        if not self._executed or self._results is None:
            raise RuntimeError(f"{self.__class__.__name__}.execute() must be called before get_results().")
        return self._results


class SectionGeometryStage(_StageBase):
    def __init__(self, *, params: SectionGeometryParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionGeometryStage:
        if self._executed:
            return self
        run_log = get_run_logger(
            package="section_geometry",
            job_dir=self._params.out_dir,
            dump_level=self._params.inp.log_dump_level,
            **_run_log_mirror_kwargs(self._params),
        )
        with run_log.scope(type(self).__name__):
            self._results = section_geometry_impl(
                self._params.inp,
                self._params.out_dir,
                section_plot_station_spec=self._params.section_plot_station_spec,
                orchestration=self._params.orchestration,
                grid_meta=self._params.grid_meta,
                section_solve_n_workers=self._params.section_solve_n_workers,
                persist_pngs=self._params.persist_pngs,
                subdir_override=self._params.subdir_override,
                run_log=run_log,
                progress=self._params.progress,
            )
        self._executed = True
        return self

    def get_results(self) -> SectionGeometryOutputs:
        return super().get_results()


class SectionShellModelStage(_StageBase):
    def __init__(self, *, params: SectionShellModelParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionShellModelStage:
        if self._executed:
            return self
        run_log = get_run_logger(
            package="section_shell_model",
            job_dir=self._params.out_dir,
            dump_level=self._params.inp.log_dump_level,
            **_run_log_mirror_kwargs(self._params),
        )
        with run_log.scope(type(self).__name__):
            self._results = section_shell_model_impl(
                self._params.inp,
                self._params.out_dir,
                section_plot_station_spec=self._params.section_plot_station_spec,
                orchestration=self._params.orchestration,
                n_elements_per_panel=self._params.n_elements_per_panel,
                dpi=self._params.dpi,
                grid_meta=self._params.grid_meta,
                station_resultants=self._params.station_resultants,
                persist_pngs=self._params.persist_pngs,
                subdir_override=self._params.subdir_override,
                loads_provenance=self._params.loads_provenance,
                use_mitc4_v2_path=self._params.use_mitc4_v2_path,
                run_log=run_log,
                progress=self._params.progress,
            )
        self._executed = True
        return self

    def get_results(self) -> SectionShellModelOutputs:
        return super().get_results()


class SectionPropertiesStage(_StageBase):
    def __init__(self, *, params: SectionPropertiesParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionPropertiesStage:
        if self._executed:
            return self
        run_log = get_run_logger(
            package="section_properties",
            job_dir=self._params.out_dir,
            dump_level=self._params.inp.log_dump_level,
            **_run_log_mirror_kwargs(self._params),
        )
        with run_log.scope(type(self).__name__):
            self._results = section_properties_impl(
                self._params.inp,
                self._params.out_dir,
                blade_yaml=self._params.blade_yaml,
                section_plot_station_spec=self._params.section_plot_station_spec,
                orchestration=self._params.orchestration,
                bg_override=self._params.bg_override,
                grid_meta=self._params.grid_meta,
                section_solve_n_workers=self._params.section_solve_n_workers,
                persist_pngs=self._params.persist_pngs,
                subdir_override=self._params.subdir_override,
                run_log=run_log,
                progress=self._params.progress,
            )
        self._executed = True
        return self

    def get_results(self) -> SectionPropertiesOutputs:
        return super().get_results()


class BeamModelStage(_StageBase):
    def __init__(self, *, params: BeamModelParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> BeamModelStage:
        if self._executed:
            return self
        run_log = get_run_logger(
            package="global_beam_model",
            job_dir=self._params.out_dir,
            dump_level=self._params.inp.log_dump_level,
            **_run_log_mirror_kwargs(self._params),
        )
        with run_log.scope(type(self).__name__):
            self._results = beam_model_impl(
                self._params.inp,
                self._params.sec,
                self._params.out_dir,
                blade_yaml=self._params.blade_yaml,
                n_beam_nodes=self._params.n_beam_nodes,
                orchestration=self._params.orchestration,
                save_section_recovery_cache_npz=self._params.save_section_recovery_cache_npz,
                bg_override=self._params.bg_override,
                grid_meta=self._params.grid_meta,
                enable_shell_recovery_enrichment=self._params.enable_shell_recovery_enrichment,
                shell_recovery_n_elements_per_panel=self._params.shell_recovery_n_elements_per_panel,
                beam_png_span_samples=self._params.beam_png_span_samples,
                persist_pngs=self._params.persist_pngs,
                subdir_override=self._params.subdir_override,
                enable_global_buckling=self._params.enable_global_buckling,
                n_global_buckling_modes=self._params.n_global_buckling_modes,
                run_log=run_log,
                progress=self._params.progress,
                enable_shell_k7_homogenisation=self._params.enable_shell_k7_homogenisation,
                shell_k7_relax=float(self._params.shell_k7_relax),
                shell_k7_outer_max_iter=int(self._params.shell_k7_outer_max_iter),
                shell_k7_tol_rel=float(self._params.shell_k7_tol_rel),
                shell_k7_n_elements_per_panel=self._params.shell_k7_n_elements_per_panel,
            )
        self._executed = True
        return self

    def get_results(self) -> BeamModelOutputs:
        return super().get_results()


class SectionOptimisationStage(_StageBase):
    def __init__(self, *, params: SectionOptimisationParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionOptimisationStage:
        if self._executed:
            return self
        run_log = get_run_logger(
            package="section_optimisation",
            job_dir=self._params.out_dir,
            dump_level=self._params.inp.log_dump_level,
            **_run_log_mirror_kwargs(self._params),
        )
        with run_log.scope(type(self).__name__):
            self._results = section_optimisation_impl(
                self._params.inp,
                self._params.out_dir,
                blade_yaml=self._params.blade_yaml,
                orchestration=self._params.orchestration,
                run_blade_optimizer=self._params.run_blade_optimizer,
                optimization_objective=self._params.optimization_objective,
                optimizer_max_iter=self._params.optimizer_max_iter,
                bg_override=self._params.bg_override,
                grid_meta=self._params.grid_meta,
                design_n_workers=self._params.design_n_workers,
                section_properties=self._params.section_properties,
                seed_section_properties=self._params.seed_section_properties,
                ks_rho=self._params.ks_rho,
                enable_panel_buckling=self._params.enable_panel_buckling,
                ks_rho_buckling=self._params.ks_rho_buckling,
                enable_global_buckling=self._params.enable_global_buckling,
                global_buckling_lambda_min=self._params.global_buckling_lambda_min,
                n_global_buckling_modes=self._params.n_global_buckling_modes,
                orientation_bounds=self._params.orientation_bounds,
                enforce_spanwise_monotone=self._params.enforce_spanwise_monotone,
                debug_stress_projection=self._params.debug_stress_projection,
                beam_driver=self._params.beam_driver,
                distributed_loads_inp=self._params.distributed_loads_inp,
                axial_loading=self._params.axial_loading,
                n_beam_nodes=int(self._params.n_beam_nodes),
                stress_recovery=self._params.stress_recovery,
                mitc4_n_elements_per_panel=int(self._params.mitc4_n_elements_per_panel),
                optimizer_method=str(self._params.optimizer_method),
                optimizer_ftol=float(self._params.optimizer_ftol),
                optimizer_n_restarts=int(self._params.optimizer_n_restarts),
                optimizer_multistart_seed=self._params.optimizer_multistart_seed,
                iteration_dump_npz=bool(self._params.iteration_dump_npz),
                iteration_hotspot_k=int(self._params.iteration_hotspot_k),
                iteration_emit_schema=bool(self._params.iteration_emit_schema),
                run_log=run_log,
                progress=self._params.progress,
                iteration_pipeline_snapshots=bool(self._params.iteration_pipeline_snapshots),
                iteration_snapshot_dpi=int(self._params.iteration_snapshot_dpi),
                iteration_snapshot_pngs=bool(self._params.iteration_snapshot_pngs),
                iteration_snapshot_max=self._params.iteration_snapshot_max,
                iteration_snapshot_stride=int(self._params.iteration_snapshot_stride),
                iteration_snapshot_grid_bundle=self._params.iteration_snapshot_grid_bundle,
                iteration_snapshot_beam_inp=self._params.iteration_snapshot_beam_inp,
            )
        self._executed = True
        return self

    def get_results(self) -> SectionOptimisationOutputs:
        return super().get_results()
