"""Precompute input bundles and stage result containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute_context import PrecomputeOrchestrationContext
from blade_precompute.section_optimisation.core.types import OptimizationObjective

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PrecomputeInputs:
    spanwise_path: Path
    extreme_loads_path: Path
    span_r_z_m: NDArray[np.float64]
    chord_m: NDArray[np.float64]
    twist_deg: NDArray[np.float64]
    naca_m: NDArray[np.float64]
    naca_p: NDArray[np.float64]
    naca_xx: NDArray[np.float64]
    loads_r_z_m: NDArray[np.float64]
    q_y_Npm: NDArray[np.float64]
    q_z_Npm: NDArray[np.float64]
    m_x_Nmpm: NDArray[np.float64]


@dataclass(frozen=True)
class SectionGeometryOutputs:
    station_indices: list[int]
    station_r_z_m: list[float]
    png_paths: list[Path]
    geometry_report_json_paths: list[Path]

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import SectionGeometryOutputsVis

        SectionGeometryOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class SectionPropertiesOutputs:
    station_z: NDArray[np.float64]
    K6: NDArray[np.float64]
    K7: NDArray[np.float64]
    results_summary_json: Path
    png_paths: list[Path]
    section_results: tuple[object, ...]
    section_definitions: tuple[object, ...]

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import SectionPropertiesOutputsVis

        SectionPropertiesOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class BeamModelOutputs:
    result_json: Path
    png_paths: list[Path]

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import BeamModelOutputsVis

        BeamModelOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class SectionOptimisationOutputs:
    result_json: Path
    png_paths: list[Path]

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import SectionOptimisationOutputsVis

        SectionOptimisationOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class SectionShellModelOutputs:
    """MITC4/CLPT shell diagnostics under ``section_shell_model/`` (unit resultants)."""

    station_indices: list[int]
    station_r_z_m: list[float]
    png_paths: list[Path]
    summary_json: Path
    skipped: bool = False

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import SectionShellModelOutputsVis

        SectionShellModelOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class LinspaceSpec:
    z_min: float
    z_max: float
    n: int


@dataclass(frozen=True)
class GridConfig:
    geometry: LinspaceSpec
    structural: LinspaceSpec
    plot_station_spec: str
    n_beam_nodes: int
    run_section_shell_model: bool = True
    section_shell_n_elements_per_panel: int = 12
    section_shell_dpi: int = 150


@dataclass(frozen=True)
class SectionGeometryParams:
    inp: PrecomputeInputs
    out_dir: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionPropertiesParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class BeamModelParams:
    inp: PrecomputeInputs
    sec: SectionPropertiesOutputs
    out_dir: Path
    blade_yaml: Path
    n_beam_nodes: int
    orchestration: PrecomputeOrchestrationContext
    save_section_recovery_cache_npz: bool = False
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionOptimisationParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    orchestration: PrecomputeOrchestrationContext
    run_blade_optimizer: bool = False
    optimization_objective: OptimizationObjective = "min_mass"
    optimizer_max_iter: int = 120
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SectionShellModelParams:
    inp: PrecomputeInputs
    out_dir: Path
    plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    n_elements_per_panel: int = 12
    dpi: int = 150
    grid_meta: Mapping[str, Any] | None = None
