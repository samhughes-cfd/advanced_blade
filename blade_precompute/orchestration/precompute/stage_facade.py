"""Stage orchestrator facades: execute / get_results."""

from __future__ import annotations

from typing import Any

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    BeamModelParams,
    SectionBucklingOutputs,
    SectionBucklingParams,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionOptimisationOutputs,
    SectionOptimisationParams,
    SectionPropertiesOutputs,
    SectionPropertiesParams,
)
from blade_precompute.orchestration.precompute.stages import (
    beam_model_impl,
    section_buckling_impl,
    section_geometry_impl,
    section_optimisation_impl,
    section_properties_impl,
)


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
        self._results = section_geometry_impl(
            self._params.inp,
            self._params.out_dir,
            plot_station_spec=self._params.plot_station_spec,
            orchestration=self._params.orchestration,
            grid_meta=self._params.grid_meta,
        )
        self._executed = True
        return self

    def get_results(self) -> SectionGeometryOutputs:
        return super().get_results()


class SectionPropertiesStage(_StageBase):
    def __init__(self, *, params: SectionPropertiesParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionPropertiesStage:
        if self._executed:
            return self
        self._results = section_properties_impl(
            self._params.inp,
            self._params.out_dir,
            blade_yaml=self._params.blade_yaml,
            plot_station_spec=self._params.plot_station_spec,
            orchestration=self._params.orchestration,
            bg_override=self._params.bg_override,
            grid_meta=self._params.grid_meta,
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
        )
        self._executed = True
        return self

    def get_results(self) -> SectionOptimisationOutputs:
        return super().get_results()


class SectionBucklingStage(_StageBase):
    def __init__(self, *, params: SectionBucklingParams) -> None:
        super().__init__()
        self._params = params

    def execute(self) -> SectionBucklingStage:
        if self._executed:
            return self
        self._results = section_buckling_impl(
            self._params.inp,
            self._params.out_dir,
            blade_yaml=self._params.blade_yaml,
            plot_station_spec=self._params.plot_station_spec,
            orchestration=self._params.orchestration,
            buckling_length_mode=self._params.buckling_length_mode,
            buckling_member_length_m=self._params.buckling_member_length_m,
            bg_override=self._params.bg_override,
            grid_meta=self._params.grid_meta,
        )
        self._executed = True
        return self

    def get_results(self) -> SectionBucklingOutputs:
        return super().get_results()
