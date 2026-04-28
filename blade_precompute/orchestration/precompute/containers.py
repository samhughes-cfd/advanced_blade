"""Precompute input bundles and stage result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from numpy.typing import NDArray

from blade_precompute.orchestration.precompute_context import PrecomputeOrchestrationContext
from blade_precompute.section_optimisation.core.types import OptimisationObjective

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PrecomputeInputs:
    spanwise_path: Path
    extreme_loads_path: Path
    span_r_z_m: NDArray[np.float64]
    # ``radial_pos`` from ``blade_spanwise_distribution.dat`` [m], resampled with ``span_r_z_m``.
    radial_r_m: NDArray[np.float64]
    chord_m: NDArray[np.float64]
    twist_deg: NDArray[np.float64]
    kappa0_x: NDArray[np.float64]
    kappa0_y: NDArray[np.float64]
    kappa0_z: NDArray[np.float64]
    naca_m: NDArray[np.float64]
    naca_p: NDArray[np.float64]
    naca_xx: NDArray[np.float64]
    naca_series: NDArray[np.int64]
    loads_r_z_m: NDArray[np.float64]
    q_y_Npm: NDArray[np.float64]
    q_z_Npm: NDArray[np.float64]
    m_x_Nmpm: NDArray[np.float64]
    log_dump_level: str = "intermediate"
    log_dir: Path | None = None
    # Live ``[precompute]`` stdout + ``<job>/progress.jsonl`` (see ``JobProgressReporter``).
    live_progress: bool = True
    # When True (or env ``ADVANCED_BLADE_PROGRESS_MIRROR_RUNLOG``), mirror selected
    # RunLogger INFO rows into ``progress.jsonl`` only.
    live_progress_mirror_run_log: bool = False


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
    K6: NDArray[np.float64]  # Classical 6x6 section stiffness table per station.
    K7: NDArray[np.float64]  # Warping-augmented 7x7 section stiffness table per station.
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
    beam_n_iterations: int | None = None
    beam_converged: bool | None = None

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import BeamModelOutputsVis

        BeamModelOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class SectionOptimisationOutputs:
    result_json: Path
    png_paths: list[Path]
    optimizer_ran: bool = False
    optimizer_n_iter: int | None = None

    def visualise(self, mode: str = "default") -> None:
        from blade_precompute.orchestration.precompute.vis import SectionOptimisationOutputsVis

        SectionOptimisationOutputsVis(self).plot(mode=mode)


@dataclass(frozen=True)
class SectionShellModelOutputs:
    """MITC4/CLPT shell diagnostics under ``section_shell_model/``."""

    station_indices: list[int]
    station_r_z_m: list[float]
    png_paths: list[Path]
    summary_json: Path
    station_result_json_paths: list[Path] = field(default_factory=list)
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
    """Spanwise discretisation and plot-density for one precompute job.

    ``geometry`` / ``structural`` are ``LinspaceSpec`` tables along ``z`` [m]; the main entrypoint
    uses the same ``z_min``/``z_max`` (from ``precompute_inputs.span_r_z_m``) for both and differs only in ``n``.
    ``section_plot_station_spec`` picks which station indices get 2D section/shell/station PNGs and
    per-station subfolders; use ``all`` (or ``structural``) for every structural index
    (see :func:`~blade_precompute.orchestration.precompute.grid.station_indices`; ``n`` there is
    the stage input station count, i.e. ``structural.n`` in the main job).
    ``beam_png_span_samples`` is the uniform abscissa count for global_beam_model spanwise PNGs
    (interpolation only; independent of ``n_beam_nodes``).
    """

    geometry: LinspaceSpec
    structural: LinspaceSpec
    section_plot_station_spec: str
    n_beam_nodes: int
    beam_png_span_samples: int = 400
    run_section_shell_model: bool = True
    section_shell_n_elements_per_panel: int = 12
    section_shell_dpi: int = 150
    enable_shell_recovery_enrichment: bool = False
    shell_recovery_n_elements_per_panel: int = 4
    # When True (or env ADVANCED_BLADE_SHELL_MITC4_V2=1), route per-station shell
    # through build_section_view → build_shell_mesh_inputs → build_mitc4_mesh instead
    # of the legacy run_section_both / write_section_shell_model_station_outputs path.
    section_shell_use_mitc4_v2: bool = False
    # Design sizing: ProcessPool for per-station midsurface solves (see DesignProblem.n_workers). Default 1
    # avoids nested pools when section_solve_n_workers>1. Raise both carefully on the same job.
    design_n_workers: int = 1
    # section_properties stage: parallel midsurface solves (see solve_dirty_stations). Plotting stays serial.
    section_solve_n_workers: int = 1
    # Blend strip section_properties K7 with MITC4-homogenised K7 before global beam equilibrium.
    enable_shell_k7_homogenisation: bool = False
    shell_k7_relax: float = 1.0
    """Weight on shell K7 in ``K_eff = w * K7_shell + (1-w) * K7_strip``."""
    shell_k7_outer_max_iter: int = 1
    shell_k7_tol_rel: float = 1e-3
    shell_k7_n_elements_per_panel: int | None = None
    """Fallback: use ``section_shell_n_elements_per_panel`` when ``None``."""


def grid_resolution_manifest(cfg: GridConfig) -> dict[str, Any]:
    """Human-oriented summary for ``inputs.json`` (mirrors ``grid_config`` without duplicating logic)."""
    return {
        "span_z_source": "precompute_inputs.span_r_z_m",
        "n_geometry_stations": int(cfg.geometry.n),
        "n_structural_stations": int(cfg.structural.n),
        "geometry_z_min_m": float(cfg.geometry.z_min),
        "geometry_z_max_m": float(cfg.geometry.z_max),
        "structural_z_min_m": float(cfg.structural.z_min),
        "structural_z_max_m": float(cfg.structural.z_max),
        "beam_nodes": int(cfg.n_beam_nodes),
        "beam_png_span_samples": int(cfg.beam_png_span_samples),
        "section_plot_station_spec": str(cfg.section_plot_station_spec),
        "run_section_shell_model": bool(cfg.run_section_shell_model),
        "section_shell_n_elements_per_panel": int(cfg.section_shell_n_elements_per_panel),
        "section_shell_dpi": int(cfg.section_shell_dpi),
        "section_shell_use_mitc4_v2": bool(cfg.section_shell_use_mitc4_v2),
        "enable_shell_recovery_enrichment": bool(cfg.enable_shell_recovery_enrichment),
        "shell_recovery_n_elements_per_panel": int(cfg.shell_recovery_n_elements_per_panel),
        "design_n_workers": int(cfg.design_n_workers),
        "section_solve_n_workers": int(cfg.section_solve_n_workers),
        "enable_shell_k7_homogenisation": bool(cfg.enable_shell_k7_homogenisation),
        "shell_k7_relax": float(cfg.shell_k7_relax),
        "shell_k7_outer_max_iter": int(cfg.shell_k7_outer_max_iter),
        "shell_k7_tol_rel": float(cfg.shell_k7_tol_rel),
        "shell_k7_n_elements_per_panel": cfg.shell_k7_n_elements_per_panel,
    }


def runtime_statistics_manifest(
    *,
    stage_seconds: Mapping[str, float],
    total_wall_s: float,
    run_section_shell_model: bool,
    section_shell_skipped: bool,
    section_geometry_station_count: int,
    section_shell_station_count: int,
    section_properties_station_count: int,
    beam_converged: bool | None,
    beam_n_iterations: int | None,
    optimizer_ran: bool,
    optimizer_n_iter: int | None,
    python_version: str,
    platform: str,
    cpu_count: int | None,
    finished_at_iso: str,
    grid_cfg: GridConfig,
) -> dict[str, Any]:
    stage_payload = {k: float(v) for k, v in stage_seconds.items()}
    denom = max(float(total_wall_s), 1e-12)
    stage_fraction_of_total = {
        k: (float(v) / denom) if float(v) > 0.0 else 0.0 for k, v in stage_payload.items()
    }
    return {
        "finished_at": str(finished_at_iso),
        "python": str(python_version),
        "platform": str(platform),
        "cpu_count": int(cpu_count) if cpu_count is not None else None,
        "total_wall_s": float(total_wall_s),
        "stages": stage_payload,
        "stage_fraction_of_total": stage_fraction_of_total,
        "work_units": {
            "n_geometry_stations": int(grid_cfg.geometry.n),
            "n_structural_stations": int(grid_cfg.structural.n),
            "n_beam_nodes": int(grid_cfg.n_beam_nodes),
            "n_beam_png_span_samples": int(grid_cfg.beam_png_span_samples),
            "section_geometry_station_count": int(section_geometry_station_count),
            "section_shell_station_count": int(section_shell_station_count),
            "section_properties_station_count": int(section_properties_station_count),
            "run_section_shell_model": bool(run_section_shell_model),
            "section_shell_skipped": bool(section_shell_skipped),
            "section_shell_n_elements_per_panel": int(grid_cfg.section_shell_n_elements_per_panel),
            "shell_recovery_n_elements_per_panel": int(grid_cfg.shell_recovery_n_elements_per_panel),
            "design_n_workers": int(grid_cfg.design_n_workers),
            "section_solve_n_workers": int(grid_cfg.section_solve_n_workers),
        },
        "algorithms": {
            "beam_converged": beam_converged,
            "beam_n_iterations": beam_n_iterations,
            "optimizer_ran": bool(optimizer_ran),
            "optimizer_n_iter": optimizer_n_iter,
        },
    }


@dataclass(frozen=True)
class SectionGeometryParams:
    inp: PrecomputeInputs
    out_dir: Path
    section_plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    grid_meta: Mapping[str, Any] | None = None
    section_solve_n_workers: int = 1
    persist_pngs: bool = True
    """When False, skip all PNG writes (JSON-only mode for pre-optimisation pass)."""
    subdir_override: Path | None = None
    """When set, stage writes into ``out_dir / subdir_override`` instead of the default."""
    progress: Any | None = None
    """Optional :class:`~blade_precompute._utils.job_progress.JobProgressReporter`."""


@dataclass(frozen=True)
class SectionPropertiesParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    section_plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    bg_override: Any | None = None
    grid_meta: Mapping[str, Any] | None = None
    section_solve_n_workers: int = 1
    persist_pngs: bool = True
    """When False, skip all PNG writes (JSON-only mode for pre-optimisation pass)."""
    subdir_override: Path | None = None
    """When set, stage writes into ``out_dir / subdir_override`` instead of the default."""
    progress: Any | None = None
    """Optional :class:`~blade_precompute._utils.job_progress.JobProgressReporter`."""


@dataclass(frozen=True)
class BeamModelParams:
    inp: PrecomputeInputs
    sec: SectionPropertiesOutputs  # Section properties handed off to global beam model.
    out_dir: Path
    blade_yaml: Path
    n_beam_nodes: int
    orchestration: PrecomputeOrchestrationContext
    save_section_recovery_cache_npz: bool = False
    bg_override: Any | None = None  # Optional geometry override (else loaded from blade_yaml).
    grid_meta: Mapping[str, Any] | None = None
    enable_shell_recovery_enrichment: bool = False
    shell_recovery_n_elements_per_panel: int = 4
    beam_png_span_samples: int = 400
    persist_pngs: bool = True
    """When False, skip all PNG writes (JSON-only mode for pre-optimisation pass)."""
    subdir_override: Path | None = None
    """When set, stage writes into ``out_dir / subdir_override`` instead of the default."""
    enable_global_buckling: bool = False
    n_global_buckling_modes: int = 5
    progress: Any | None = None
    """Optional :class:`~blade_precompute._utils.job_progress.JobProgressReporter`."""
    enable_shell_k7_homogenisation: bool = False
    shell_k7_relax: float = 1.0
    shell_k7_outer_max_iter: int = 1
    shell_k7_tol_rel: float = 1e-3
    shell_k7_n_elements_per_panel: int | None = None


@dataclass(frozen=True)
class SectionOptimisationParams:
    inp: PrecomputeInputs
    out_dir: Path
    blade_yaml: Path
    orchestration: PrecomputeOrchestrationContext
    run_blade_optimizer: bool = False
    optimization_objective: OptimisationObjective = "min_mass"
    optimizer_max_iter: int = 120
    bg_override: Any | None = None  # Optional geometry override (else loaded from blade_yaml).
    grid_meta: Mapping[str, Any] | None = None
    design_n_workers: int = 1
    # When set (e.g. from section_properties) and seed_section_properties: seed DesignEvaluator to skip
    # re-solving midsurfaces at dv0 that section_properties already computed.
    section_properties: SectionPropertiesOutputs | None = None
    seed_section_properties: bool = True
    ks_rho: float = 35.0
    # --- Group J: buckling knobs (J.6) ---
    enable_panel_buckling: bool = False
    ks_rho_buckling: float = 25.0
    enable_global_buckling: bool = False
    global_buckling_lambda_min: float = 1.5
    n_global_buckling_modes: int = 5
    # --- Group L: orientation bounds (L.6) ---
    # Dict keyed by role ("skin", "cap", "web") → OrientationBounds (imported lazily to avoid circular deps).
    orientation_bounds: Any | None = None
    # --- Group L.9: spanwise monotone thickness ---
    enforce_spanwise_monotone: bool = True
    # --- Fix 1: stress projection diagnostics ---
    debug_stress_projection: bool = False
    # Group H + MITC4 in-loop (see :class:`DesignProblem`)
    beam_driver: str = "prescribed"
    # Fine-span loads grid (e.g. inp_geom) for global_beam / coupled_fe
    distributed_loads_inp: PrecomputeInputs | None = None
    # Optional AxialLoadingConfig (centrifugal + gravity q_x / N); see main_precompute
    axial_loading: Any | None = None
    n_beam_nodes: int = 50
    stress_recovery: str = "mitc4"
    mitc4_n_elements_per_panel: int = 10
    optimizer_method: str = "SLSQP"
    optimizer_ftol: float = 1e-5
    optimizer_n_restarts: int = 0
    optimizer_multistart_seed: int | None = None
    iteration_dump_npz: bool = False
    iteration_hotspot_k: int = 10
    iteration_emit_schema: bool = True
    # Full four-stage snapshots under ``<job>/section_optimisation/iter_NNNN/`` (off by default; expensive).
    iteration_pipeline_snapshots: bool = False
    iteration_snapshot_dpi: int = 96
    iteration_snapshot_pngs: bool = True
    iteration_snapshot_max: int | None = None
    iteration_snapshot_stride: int = 1
    """When ``iteration_pipeline_snapshots`` is True, bundle from main with keys
    ``section_geometry``, ``section_properties``, ``beam``, ``section_shell`` (grid metas),
    plus ``section_plot_station_spec``, ``section_solve_n_workers``, ``n_beam_nodes``,
    ``enable_shell_recovery_enrichment``, ``shell_recovery_n_elements_per_panel``,
    ``beam_png_span_samples``, ``n_elements_per_panel`` (shell), ``use_mitc4_v2_path``,
    ``save_section_recovery_cache_npz``."""
    iteration_snapshot_grid_bundle: Mapping[str, Any] | None = None
    iteration_snapshot_beam_inp: PrecomputeInputs | None = None
    """Fine-span inputs for beam inside iteration snapshots (e.g. same as ``distributed_loads_inp``)."""
    progress: Any | None = None
    """Optional :class:`~blade_precompute._utils.job_progress.JobProgressReporter`."""


@dataclass(frozen=True)
class SectionShellModelParams:
    inp: PrecomputeInputs
    out_dir: Path
    section_plot_station_spec: str
    orchestration: PrecomputeOrchestrationContext
    n_elements_per_panel: int = 12
    dpi: int = 150
    grid_meta: Mapping[str, Any] | None = None
    station_resultants: "Mapping[int, tuple[float, float, float, float, float, float]] | None" = None
    """Per-station ``(N, Vy, Vz, My, Mz, T)`` real-load resultants from the beam solve.
    When None, defaults to unit resultants (kept for standalone/test use only)."""
    persist_pngs: bool = True
    """When False, skip all PNG writes (JSON-only mode for pre-optimisation pass)."""
    subdir_override: Path | None = None
    """When set, stage writes into ``out_dir / subdir_override`` instead of the default."""
    loads_provenance: str = "unit_resultants"
    """Human-readable note recorded in ``summary.json`` describing the loads source."""
    use_mitc4_v2_path: bool = False
    """Route per-station shell through build_section_view → build_shell_mesh_inputs → build_mitc4_mesh.
    Also activated by env var ADVANCED_BLADE_SHELL_MITC4_V2=1."""
    progress: Any | None = None
    """Optional :class:`~blade_precompute._utils.job_progress.JobProgressReporter`."""
