"""Entry point: blade precompute pipeline (orchestration in blade_precompute.orchestration.precompute).

Edit the variables in the "Control settings" section below (not via CLI).
"""

from __future__ import annotations

import atexit
import sys
import os
import platform
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute._utils.job_progress import (
    CONSOLE_LOG_PREFIX,
    JobProgressReporter,
    live_progress_enabled_from_env,
)
from blade_precompute.global_beam_model.engine.axial_loading import AxialLoadingConfig, manifest_dict
from blade_precompute.section_optimisation.core.types import objective_from_str, apply_dv_to_bg
from blade_precompute.orchestration.precompute import (
    BeamModelOutputs,
    BeamModelParams,
    BeamModelStage,
    GridConfig,
    LinspaceSpec,
    SectionGeometryParams,
    SectionGeometryStage,
    SectionOptimisationParams,
    SectionOptimisationStage,
    SectionPropertiesParams,
    SectionPropertiesStage,
    SectionShellModelOutputs,
    SectionShellModelParams,
    SectionShellModelStage,
    build_optim_blade_geometry_from_spanwise,
    build_precompute_orchestration_context,
    grid_resolution_manifest,
    job_span_z_m,
    linspace_from_spec,
    load_inputs,
    resample_precompute_inputs,
    resolve_component_materials_path,
    runtime_statistics_manifest,
    write_json,
)
from blade_precompute.orchestration.precompute.material_library import (
    load_material_library_dat,
    material_resolution_manifest,
    normalize_logical_subcomponent_material_map,
    validate_material_library_bindings,
)
from blade_precompute.orchestration.precompute.stages import section_shell_model_skipped_outputs

_REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Control settings (edit here)
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path("data_library")
# Canonical mode: blade geometry/materials are built from
# ``blade_spanwise_distribution.dat`` + ``material_library.dat``.
#
# Web positions are derived from SYSTEM_TYPE via orch.layout.web_chord_fracs
# (converted to half-chord normalised coords: pos = frac - 0.5).
# Box height is derived from the mean NACA max-thickness ratio (naca_xx / 100).
# Max component thicknesses are enforced via DesignVector t_*_bounds (see
# blade_precompute/section_optimisation/core/types.py).
# Initial ply-angle stacks (skin: [0,±45,90], cap: [0,0], web: [±45]) are the
# canonical starting points owned by build_optim_blade_geometry_from_spanwise.
# GBT / member section buckling: use ``examples/section_buckling`` and ``examples/section_beam_model`` (not precompute).
OUTPUT_BASE_DIR: Path = _REPO_ROOT / "outputs"

# --- Spanwise resolution (feeds ``GridConfig``; see ``grid_resolution_manifest`` in ``inputs.json``) ---
# Root/tip ``z`` [m] come from the loaded spanwise table (``span_r_z_m``); only station counts are tuned here.
# ``N_GEOMETRY``: resampled spanwise table for ``inp_geom`` (loads / beam interpolation path).
N_GEOMETRY: int = 176
# ``N_STRUCTURAL``: ``inp_struct`` / ``bg_struct`` — section geometry, shell, properties, design abscissa.
N_STRUCTURAL: int = 10

# Per-station subfolders under section_geometry / section_properties / section_shell_model (``station_indices`` DSL).
# Default ``all``: one output folder per structural grid station (``N_STRUCTURAL``). Use e.g. ``root,mid,tip`` or
# ``every-3`` for a sparse set when PNG count or wall time matters.
SECTION_PLOT_STATION_SPEC: str = "all"

# Global beam: FE node count along the blade; PNG spanwise curves use ``N_BEAM_PNG_SPAN_SAMPLES`` (interpolation only).
N_BEAM_NODES: int = 50
N_BEAM_PNG_SPAN_SAMPLES: int = 400

# Section shell stage (MITC4 diagnostic figures).
RUN_SECTION_SHELL_MODEL: bool = True
SECTION_SHELL_N_ELEMENTS_PER_PANEL: int = 12
SECTION_SHELL_DPI: int = 50
# Midline → MITC4 mesh path (``section_shell_station_v2`` JSON); or env ``ADVANCED_BLADE_SHELL_MITC4_V2=1``.
SECTION_SHELL_USE_MITC4_V2: bool = True

# Beam equilibrium: blend strip K7 with MITC4-homogenised K7 before global_beam_model solve.
# Env ``ADVANCED_BLADE_SHELL_K7_HOMOGENISE=1`` enables (expensive: one shell assembly per station).
ENABLE_SHELL_K7_HOMOGENISATION: bool = os.environ.get("ADVANCED_BLADE_SHELL_K7_HOMOGENISE", "").strip().lower() in ("1", "true", "yes")
SHELL_K7_RELAX: float = float(os.environ.get("ADVANCED_BLADE_SHELL_K7_RELAX", "1.0"))
SHELL_K7_OUTER_MAX_ITER: int = int(os.environ.get("ADVANCED_BLADE_SHELL_K7_OUTER_MAX_ITER", "3"))
SHELL_K7_TOL_REL: float = float(os.environ.get("ADVANCED_BLADE_SHELL_K7_TOL_REL", "1e-3"))

SAVE_SECTION_RECOVERY_CACHE_NPZ: bool = False
# Thin-wall + MITC4 shell recovery at section_properties stations (adds shell_recovery to beam_result.json).
ENABLE_SHELL_RECOVERY_ENRICHMENT: bool = True
SHELL_RECOVERY_N_ELEMENTS_PER_PANEL: int = 10
# Compact SystemType{X}{Y}-{Z} key — see blade_precompute/section_geometry/docs/system_type_xyz_taxonomy.md
SYSTEM_TYPE: str = "2D-F"
COMPONENT_MATERIALS: Path | None = None

# ``data_library/material_library.dat`` + logical subcomponent -> ``material_id`` (aliases ``spar``/``web`` ok).
# Leave empty ``{}`` to keep input ply assignments only (no .dat swap).
SUBCOMPONENT_MATERIAL_IDS: dict[str, int] = {"skin": 0, "spar_cap": 0, "shear_web": 0, "core": 3}

DESIGN_OPTIMISE: bool = True
DESIGN_OBJECTIVE: str = "min-mass"
DESIGN_MAX_ITER: int = 120
KS_RHO: float = 35.0
# Parallel midsurface solves in design evaluation / SLSQP (ProcessPool). Use 1 if also using
# SECTION_SOLVE_N_WORKERS>1 to limit nested process pools.
DESIGN_N_WORKERS: int = 4
# section_properties stage: parallel per-station midsurface solves.
SECTION_SOLVE_N_WORKERS: int = 4
# Reuse section_properties midsurface results for the first design evaluate(dv0) (recommended).
SEED_DESIGN_FROM_SECTION_PROPERTIES: bool = True
# Run-log verbosity: "summary", "intermediate", "full" (``<job>/<package>/run.log`` volume; not terminal).
# ``[precompute]`` substep lines require ``inp.live_progress`` and env ``ADVANCED_BLADE_LIVE_PROGRESS`` (default on).
LOG_DUMP_LEVEL: str = "intermediate"
# Section optimisation: optional per-SLSQP-iteration ``.npz`` (FI tensors, strip CLT, MITC4 panel ABD @ station 0).
SECTION_OPTIM_DUMP_ITERATION_NPZ: bool = False
SECTION_OPTIM_ITERATION_HOTSPOT_K: int = 10
SECTION_OPTIM_EMIT_ITERATION_SCHEMA: bool = True

# Per-optimisation-iteration full pipeline under ``<job>/section_optimisation/iter_NNNN/`` (expensive; off by default).
# Only written after the pre-loop stages finish and ``section_optimisation`` runs (initial eval → ``iter_0000``, then callbacks).
ITERATION_PIPELINE_SNAPSHOTS: bool = False
ITERATION_SNAPSHOT_DPI: int = 50
ITERATION_SNAPSHOT_PNGS: bool = True
# Maximum snapshot index (exclusive): indices ``0 .. max-1`` (``iter_0000`` ..). ``None`` = unlimited.
# When snapshots are enabled, a finite cap avoids multi-hour jobs (each ``iter_*`` re-runs geometry→shell).
ITERATION_SNAPSHOT_MAX: int | None = 12
ITERATION_SNAPSHOT_STRIDE: int = 2

# --- Group J: Panel + Global Buckling (J.6) ---
# Orthotropic closed-form local panel buckling (orthotropic plate, ESDU 80023 style).
ENABLE_PANEL_BUCKLING: bool = False
# KS aggregation for panel buckling constraint (sharper aggregate than strength rho=35).
KS_RHO_BUCKLING: float = 25.0
# Global beam buckling (K_t - lambda K_g) phi = 0 via scipy eigsh. Requires coupled FE driver.
ENABLE_GLOBAL_BUCKLING: bool = False
# Minimum global buckling eigenvalue (safety factor): c = lambda_crit - GLOBAL_BUCKLING_LAMBDA_MIN >= 0.
GLOBAL_BUCKLING_LAMBDA_MIN: float = 1.5
# Number of lowest global buckling eigenvalues to extract per evaluation.
N_GLOBAL_BUCKLING_MODES: int = 5

# --- Group L: Ply Orientation as Discrete Design Variable ---
# Outer-inner optimisation: outer loop enumerates discrete OrientationMix combos per role;
# inner loop SLSQP optimises continuous thicknesses for each fixed orientation.
# Set ORIENTATION_BOUNDS to a dict {role: OrientationBounds(…)} to enable orientation enumeration.
# Example:
#   from blade_precompute.section_optimisation.core.types import OrientationBounds
#   ORIENTATION_BOUNDS = {"skin": OrientationBounds(n_half_min=2, n_half_max=8, n_biax_min=1)}
ORIENTATION_BOUNDS: "dict | None" = None  # None → fixed orientation from YAML template

# Half-stack ply count bounds for skin orientation enumeration (used when ORIENTATION_BOUNDS is None).
N_HALF_MIN_SKIN: int = 2
N_HALF_MAX_SKIN: int = 10
N_BIAX_MIN_SKIN: int = 1

# Spanwise monotone thickness: t_role[i] >= t_role[i+1] (ply drops only toward tip).
ENFORCE_MONOTONE_THICKNESS: bool = True

DEBUG_STRESS_PROJECTION: bool = False

# --- Axial loading: centrifugal + self-weight (uses mass/length from section_properties) ---
# ``omega = U_INF * TSR / max(radial_r_m)``; ``q_x = mu * (omega^2 * r + g*cos(azimuth))`` [N/m]
U_INF: float = 2.5
TSR: float = 5.0
AZIMUTH_DEG: float = 180 # at 180 deg azimuthal position, weight adds with centrifugal force.
GRAVITY_M_S2: float = 9.81
AXIAL_LOADING_ENABLED: bool = True

# --- In-loop beam + stress (Group H) ---
# ``prescribed``: Tier-B, tabulated extreme-load resultants (default).
# ``global_beam`` / ``coupled_fe``: Tier-A static beam with distributed .dat loads (equilibrated
# resultants each SLSQP eval; section K7/K6 from current thicknesses). Aliases use the same driver today.
BEAM_DRIVER: str = "coupled_fe"
# When ``global_beam``: must match the fine spanwise table used for beam interpolation.
STRESS_RECOVERY: str = "mitc4"  # only supported value (legacy strip_clpt/both coerced in DesignProblem)
MITC4_N_ELEMENTS_PER_PANEL: int = 10
# SciPy: ``SLSQP`` (default) or ``trust-constr``; see DesignProblem in section_optimisation.
DESIGN_OPTIMIZER_METHOD: str = "SLSQP"
DESIGN_OPTIMIZER_FTOL: float = 1e-5
# Optional multistart from random designs in bounds (0 = only initial dv0).
DESIGN_OPTIMIZER_N_RESTARTS: int = 0
DESIGN_OPTIMIZER_MULTISTART_SEED: int | None = None


def _job_dir(base_out: Path) -> Path:
    base_out = base_out.resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base_out.iterdir() if p.is_dir()]
    run_idxs: list[int] = []
    for name in existing:
        if "__run" not in name:
            continue
        token = name.split("__run", 1)[1].split("__", 1)[0]
        if token.isdigit():
            run_idxs.append(int(token))
    next_idx = (max(run_idxs) + 1) if run_idxs else 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj = str(DESIGN_OBJECTIVE).strip().replace("_", "-")
    label = (
        f"{ts}__run{next_idx:03d}"
        f"__obj-{obj}"
        f"__opt-{int(DESIGN_OPTIMISE)}"
        f"__maxiter-{int(DESIGN_MAX_ITER)}"
        f"__nstruct-{int(N_STRUCTURAL)}"
        f"__nbeam-{int(N_BEAM_NODES)}"
    )
    return (base_out / label).resolve()


def main() -> int:
    t_main_start = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    t_stage_start = t_main_start
    design_objective = objective_from_str(DESIGN_OBJECTIVE)

    inp = load_inputs(DATA_DIR)
    inp = replace(inp, log_dump_level=str(LOG_DUMP_LEVEL))
    job = _job_dir(OUTPUT_BASE_DIR)
    job.mkdir(parents=True, exist_ok=True)

    _live_progress_ok = bool(inp.live_progress and live_progress_enabled_from_env())
    progress_rpt = JobProgressReporter(job, enabled=_live_progress_ok)
    if _live_progress_ok:
        progress_rpt.event(
            "job_start",
            job=str(job),
            n_structural=int(N_STRUCTURAL),
            n_beam_nodes=int(N_BEAM_NODES),
            design_optimise=bool(DESIGN_OPTIMISE),
            beam_driver=str(BEAM_DRIVER),
            stress_recovery=str(STRESS_RECOVERY),
            system_type=str(SYSTEM_TYPE),
        )

    _job_exit_state: dict[str, Any] = {"completed": False, "job": job, "live": _live_progress_ok, "pr": progress_rpt}

    def _atexit_if_incomplete() -> None:
        if _job_exit_state["completed"]:
            return
        j = _job_exit_state["job"]
        try:
            write_json(
                j / "job_exit.json",
                {
                    "status": "incomplete",
                    "recorded_at_iso": datetime.now().isoformat(timespec="seconds"),
                    "note": "Process did not finish main() normally (interrupt, crash, or external kill).",
                },
            )
        except Exception:
            pass
        if _job_exit_state["live"]:
            try:
                _job_exit_state["pr"].event("job_exit", status="incomplete")
            except Exception:
                pass

    atexit.register(_atexit_if_incomplete)

    progress_rpt.phase_start("pre_inputs")

    use_blade_spec = False
    mat_dat = (Path(DATA_DIR).resolve() / "material_library.dat").resolve()
    mat_table = None
    logical_norm: dict[str, int] | None = None
    blade_path_metadata = Path("spanwise+material_library")
    if SUBCOMPONENT_MATERIAL_IDS:
        if not mat_dat.is_file():
            raise FileNotFoundError(
                f"SUBCOMPONENT_MATERIAL_IDS is set but material library is missing: {mat_dat}"
            )
        mat_table = load_material_library_dat(mat_dat)
        logical_norm = normalize_logical_subcomponent_material_map(SUBCOMPONENT_MATERIAL_IDS)
    if not SUBCOMPONENT_MATERIAL_IDS:
        raise ValueError(
            "Set non-empty SUBCOMPONENT_MATERIAL_IDS with material_id rows "
            f"from {mat_dat} (skin, spar_cap, shear_web)."
        )
    material_resolution = material_resolution_manifest(
        material_library_path=mat_dat if mat_table is not None else None,
        logical=logical_norm,
        table=mat_table,
    )
    skip_component_index_validation = True
    orchestration_blade_yaml: Path | None
    orchestration_blade_yaml = None
    from blade_precompute.orchestration.component_materials import ComponentMaterialsMap

    cmap_override: ComponentMaterialsMap | None = None
    if logical_norm is not None and "skin" in logical_norm and "spar_cap" in logical_norm and "shear_web" in logical_norm:
        cmap_override = ComponentMaterialsMap(
            skin=int(logical_norm["skin"]),
            spar_cap=int(logical_norm["spar_cap"]),
            shear_web=int(logical_norm["shear_web"]),
        )
    orch = build_precompute_orchestration_context(
        data_dir=DATA_DIR,
        blade_yaml=orchestration_blade_yaml,
        system_type_key=str(SYSTEM_TYPE),
        component_materials_path=COMPONENT_MATERIALS,
        skip_component_index_validation=bool(skip_component_index_validation),
        component_materials_override=cmap_override,
    )

    z_root, z_tip = job_span_z_m(inp)
    gspec = LinspaceSpec(z_min=z_root, z_max=z_tip, n=int(N_GEOMETRY))
    inp_geom = resample_precompute_inputs(inp, linspace_from_spec(gspec))
    sspec = LinspaceSpec(z_min=z_root, z_max=z_tip, n=int(N_STRUCTURAL))
    z_struct = linspace_from_spec(sspec)
    inp_struct = resample_precompute_inputs(inp, z_struct)
    if mat_table is None or logical_norm is None:
        raise RuntimeError("Spanwise precompute path requires a loaded material table and SUBCOMPONENT map.")
    validate_material_library_bindings(
        mat_table,
        logical_norm,
        blade_subcomponent_names=frozenset({"cap_ps", "skin", "web"}),
    )
    bg_struct = build_optim_blade_geometry_from_spanwise(
        inp_struct,
        mat_table=mat_table,
        logical=logical_norm,
        system_layout=orch.layout,
    )
    r_tip_m = float(np.max(np.asarray(inp_struct.radial_r_m, dtype=np.float64)))
    axial_loading_cfg = AxialLoadingConfig(
        u_inf_m_s=float(U_INF),
        tip_speed_ratio=float(TSR),
        r_tip_m=r_tip_m,
        gravity_m_s2=float(GRAVITY_M_S2),
        azimuth_deg=float(AZIMUTH_DEG),
        enabled=bool(AXIAL_LOADING_ENABLED),
    )
    axial_manifest = manifest_dict(axial_loading_cfg, mu_source="section_properties")
    grid_cfg = GridConfig(
        geometry=gspec,
        structural=sspec,
        section_plot_station_spec=str(SECTION_PLOT_STATION_SPEC),
        n_beam_nodes=int(N_BEAM_NODES),
        beam_png_span_samples=int(N_BEAM_PNG_SPAN_SAMPLES),
        run_section_shell_model=bool(RUN_SECTION_SHELL_MODEL),
        section_shell_n_elements_per_panel=int(SECTION_SHELL_N_ELEMENTS_PER_PANEL),
        section_shell_dpi=int(SECTION_SHELL_DPI),
        section_shell_use_mitc4_v2=bool(SECTION_SHELL_USE_MITC4_V2),
        enable_shell_recovery_enrichment=bool(ENABLE_SHELL_RECOVERY_ENRICHMENT),
        shell_recovery_n_elements_per_panel=int(SHELL_RECOVERY_N_ELEMENTS_PER_PANEL),
        design_n_workers=int(DESIGN_N_WORKERS),
        section_solve_n_workers=int(SECTION_SOLVE_N_WORKERS),
        enable_shell_k7_homogenisation=bool(ENABLE_SHELL_K7_HOMOGENISATION),
        shell_k7_relax=float(SHELL_K7_RELAX),
        shell_k7_outer_max_iter=int(SHELL_K7_OUTER_MAX_ITER),
        shell_k7_tol_rel=float(SHELL_K7_TOL_REL),
        shell_k7_n_elements_per_panel=int(SECTION_SHELL_N_ELEMENTS_PER_PANEL),
    )

    write_json(
        job / "inputs.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "config_source": "main_precompute module variables",
            "use_blade_spec": bool(use_blade_spec),
            "blade_source": "spanwise+material_library",
            "spanwise_path": inp.spanwise_path,
            "extreme_loads_path": inp.extreme_loads_path,
            "blade_spec_path": "spanwise+material_library",
            "blade_spec": str(blade_path_metadata),
            "system_type_key": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "component_materials_path": str(resolve_component_materials_path(
                DATA_DIR, COMPONENT_MATERIALS
            ) or "none"),
            "design_optimise": bool(DESIGN_OPTIMISE),
            "design_objective": design_objective,
            "design_max_iter": int(DESIGN_MAX_ITER),
            "design_n_workers": int(DESIGN_N_WORKERS),
            "section_solve_n_workers": int(SECTION_SOLVE_N_WORKERS),
            "ks_rho": float(KS_RHO),
            "seed_design_from_section_properties": bool(SEED_DESIGN_FROM_SECTION_PROPERTIES),
            "beam_driver": str(BEAM_DRIVER),
            "stress_recovery": str(STRESS_RECOVERY),
            "mitc4_n_elements_per_panel": int(MITC4_N_ELEMENTS_PER_PANEL),
            "run_section_shell_model": bool(grid_cfg.run_section_shell_model),
            "iteration_pipeline_snapshots": bool(ITERATION_PIPELINE_SNAPSHOTS),
            "iteration_snapshot_dpi": int(ITERATION_SNAPSHOT_DPI),
            "iteration_snapshot_pngs": bool(ITERATION_SNAPSHOT_PNGS),
            "iteration_snapshot_max": ITERATION_SNAPSHOT_MAX,
            "iteration_snapshot_stride": int(ITERATION_SNAPSHOT_STRIDE),
            "grid_config": grid_cfg,
            "spanwise_resolution": grid_resolution_manifest(grid_cfg),
            "material_resolution": material_resolution,
            "axial_loading": axial_manifest,
        },
    )
    stage_seconds["pre_inputs_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("pre_inputs")

    # ------------------------------------------------------------------
    # Helpers to build stage params with a given bg/persist/subdir
    # ------------------------------------------------------------------
    _sg_grid_meta = {"type": "structural", "linspace": sspec, "geometry_linspace": gspec}
    _sp_grid_meta = {"type": "structural", "linspace": sspec}
    _bm_grid_meta = {
        "type": "beam",
        "structural_linspace": sspec,
        "n_beam_nodes": int(grid_cfg.n_beam_nodes),
        "beam_png_span_samples": int(grid_cfg.beam_png_span_samples),
    }
    _sh_grid_meta = {"type": "structural", "linspace": sspec, "geometry_linspace": gspec}

    def _build_sg_params(*, persist_pngs: bool, subdir_override: Path | None = None) -> SectionGeometryParams:
        return SectionGeometryParams(
            inp=inp_struct, out_dir=job,
            section_plot_station_spec=grid_cfg.section_plot_station_spec,
            orchestration=orch, grid_meta=_sg_grid_meta,
            section_solve_n_workers=int(grid_cfg.section_solve_n_workers),
            persist_pngs=persist_pngs, subdir_override=subdir_override,
            progress=progress_rpt,
        )

    def _build_sp_params(bg: Any, *, persist_pngs: bool, subdir_override: Path | None = None) -> SectionPropertiesParams:
        return SectionPropertiesParams(
            inp=inp_struct, out_dir=job,
            blade_yaml=blade_path_metadata,
            section_plot_station_spec=grid_cfg.section_plot_station_spec,
            orchestration=orch, bg_override=bg, grid_meta=_sp_grid_meta,
            section_solve_n_workers=int(grid_cfg.section_solve_n_workers),
            persist_pngs=persist_pngs, subdir_override=subdir_override,
            progress=progress_rpt,
        )

    def _build_bm_params(bg: Any, sp: Any, *, persist_pngs: bool, subdir_override: Path | None = None) -> BeamModelParams:
        _k7_nep = grid_cfg.shell_k7_n_elements_per_panel
        if _k7_nep is None:
            _k7_nep = int(grid_cfg.section_shell_n_elements_per_panel)
        return BeamModelParams(
            inp=inp_geom, sec=sp, out_dir=job,
            blade_yaml=blade_path_metadata,
            n_beam_nodes=int(grid_cfg.n_beam_nodes),
            orchestration=orch,
            save_section_recovery_cache_npz=bool(SAVE_SECTION_RECOVERY_CACHE_NPZ),
            bg_override=bg, grid_meta=_bm_grid_meta,
            enable_shell_recovery_enrichment=bool(grid_cfg.enable_shell_recovery_enrichment),
            shell_recovery_n_elements_per_panel=int(grid_cfg.shell_recovery_n_elements_per_panel),
            beam_png_span_samples=int(grid_cfg.beam_png_span_samples),
            persist_pngs=persist_pngs, subdir_override=subdir_override,
            progress=progress_rpt,
            enable_shell_k7_homogenisation=bool(grid_cfg.enable_shell_k7_homogenisation),
            shell_k7_relax=float(grid_cfg.shell_k7_relax),
            shell_k7_outer_max_iter=int(grid_cfg.shell_k7_outer_max_iter),
            shell_k7_tol_rel=float(grid_cfg.shell_k7_tol_rel),
            shell_k7_n_elements_per_panel=_k7_nep,
        )

    def _extract_station_resultants(bm_out: BeamModelOutputs) -> dict[int, tuple]:
        """Extract per-station (N, Vy, Vz, My, Mz, T) from beam result JSON."""
        import json as _json
        try:
            with open(bm_out.result_json, "r", encoding="utf-8") as _f:
                _beam_data = _json.load(_f)
            _resultants = np.asarray(_beam_data.get("resultants", []), dtype=np.float64)
            _z_out = _beam_data.get("z_stations_out", None)
            if _resultants.ndim == 2 and _resultants.shape[1] >= 6:
                # Beam JSON stores solver-native columns: N, Vy, Vz, My, Mz, T, B.
                # The shell stage consumes (N, Vy, Vz, My, Mz, T).
                _col_N, _col_Vy, _col_Vz, _col_My, _col_Mz, _col_T = 0, 1, 2, 3, 4, 5
                n_sh = int(inp_struct.span_r_z_m.shape[0])
                _z_struct = np.asarray(inp_struct.span_r_z_m, dtype=np.float64)
                _n_elem = _resultants.shape[0]
                # elem midpoint z: uniform from 0..n_elem-1 in absence of z_out
                if _z_out is not None:
                    _z_elem = np.asarray(_z_out, dtype=np.float64).ravel()
                    if _z_elem.size != _n_elem:
                        _z_elem = np.linspace(float(_z_struct[0]), float(_z_struct[-1]), _n_elem)
                else:
                    _z_elem = np.linspace(float(_z_struct[0]), float(_z_struct[-1]), _n_elem)
                out_map: dict[int, tuple] = {}
                for _si in range(n_sh):
                    _zi = float(_z_struct[_si])
                    _idx = int(np.argmin(np.abs(_z_elem - _zi)))
                    _row = _resultants[_idx]
                    out_map[_si] = (
                        float(_row[_col_N]),
                        float(_row[_col_Vy]),
                        float(_row[_col_Vz]),
                        float(_row[_col_My]),
                        float(_row[_col_Mz]),
                        float(_row[_col_T]),
                    )
                return out_map
        except Exception:
            pass
        return {}

    def _run_shell_stage(
        bm_out: BeamModelOutputs,
        bg: Any,
        *,
        persist_pngs: bool,
        subdir_override: Path | None = None,
        loads_provenance: str,
    ) -> SectionShellModelOutputs:
        station_res = _extract_station_resultants(bm_out) if bm_out is not None else {}
        if grid_cfg.run_section_shell_model:
            return SectionShellModelStage(
                params=SectionShellModelParams(
                    inp=inp_struct, out_dir=job,
                    section_plot_station_spec=grid_cfg.section_plot_station_spec,
                    orchestration=orch,
                    n_elements_per_panel=int(grid_cfg.section_shell_n_elements_per_panel),
                    dpi=int(grid_cfg.section_shell_dpi),
                    grid_meta=_sh_grid_meta,
                    station_resultants=station_res if station_res else None,
                    persist_pngs=persist_pngs,
                    subdir_override=subdir_override,
                    loads_provenance=loads_provenance,
                    use_mitc4_v2_path=bool(grid_cfg.section_shell_use_mitc4_v2),
                    progress=progress_rpt,
                )
            ).execute().get_results()
        return section_shell_model_skipped_outputs(
            job, orchestration=orch, reason="run_section_shell_model=false", grid_meta=_sh_grid_meta,
        )

    # ------------------------------------------------------------------
    # Determine whether to defer PNGs to post-optimisation pass
    # ------------------------------------------------------------------
    _defer_pngs = bool(DESIGN_OPTIMISE)

    _iter_snapshot_bundle: dict[str, Any] | None = None
    if ITERATION_PIPELINE_SNAPSHOTS:
        _iter_snapshot_bundle = {
            "section_geometry": _sg_grid_meta,
            "section_properties": _sp_grid_meta,
            "beam": _bm_grid_meta,
            "section_shell": _sh_grid_meta,
            "section_plot_station_spec": str(grid_cfg.section_plot_station_spec),
            "section_solve_n_workers": int(grid_cfg.section_solve_n_workers),
            "n_beam_nodes": int(grid_cfg.n_beam_nodes),
            "enable_shell_recovery_enrichment": bool(grid_cfg.enable_shell_recovery_enrichment),
            "shell_recovery_n_elements_per_panel": int(grid_cfg.shell_recovery_n_elements_per_panel),
            "beam_png_span_samples": int(grid_cfg.beam_png_span_samples),
            "n_elements_per_panel": int(grid_cfg.section_shell_n_elements_per_panel),
            "use_mitc4_v2_path": bool(grid_cfg.section_shell_use_mitc4_v2),
            "save_section_recovery_cache_npz": bool(SAVE_SECTION_RECOVERY_CACHE_NPZ),
        }

    # --- PRE-LOOP: section_geometry (always JSON; PNGs only when not optimising) ---
    progress_rpt.phase_start("section_geometry")
    t_stage_start = time.perf_counter()
    sg = SectionGeometryStage(params=_build_sg_params(persist_pngs=not _defer_pngs)).execute().get_results()
    stage_seconds["section_geometry_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("section_geometry")

    # --- PRE-LOOP: section_properties (JSON only when deferring) ---
    progress_rpt.phase_start("section_properties")
    t_stage_start = time.perf_counter()
    sp = SectionPropertiesStage(params=_build_sp_params(bg_struct, persist_pngs=not _defer_pngs)).execute().get_results()
    stage_seconds["section_properties_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("section_properties")

    # --- PRE-LOOP: global_beam_model (JSON only when deferring) ---
    progress_rpt.phase_start("global_beam_model")
    t_stage_start = time.perf_counter()
    bm = BeamModelStage(params=_build_bm_params(bg_struct, sp, persist_pngs=not _defer_pngs)).execute().get_results()
    stage_seconds["global_beam_model_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("global_beam_model")

    # --- PRE-LOOP: section_shell_model (real initial-dv loads; JSON only when deferring) ---
    progress_rpt.phase_start("section_shell_model")
    t_stage_start = time.perf_counter()
    sh = _run_shell_stage(
        bm, bg_struct,
        persist_pngs=not _defer_pngs,
        loads_provenance="real_initial_dv" if bm is not None else "unit_resultants",
    )
    stage_seconds["section_shell_model_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("section_shell_model")

    # ------------------------------------------------------------------
    # Section optimisation (unchanged — always writes its own PNGs)
    # ------------------------------------------------------------------
    progress_rpt.phase_start("section_optimisation")
    t_stage_start = time.perf_counter()
    # Build orientation_bounds from main_precompute knobs (L.6)
    _orient_bounds = ORIENTATION_BOUNDS
    if _orient_bounds is None and (N_HALF_MIN_SKIN > 0 or N_HALF_MAX_SKIN > 0):
        try:
            from blade_precompute.section_optimisation.core.types import OrientationBounds as _OB
            _orient_bounds = {
                "skin": _OB(
                    n_half_min=int(N_HALF_MIN_SKIN),
                    n_half_max=int(N_HALF_MAX_SKIN),
                    n_biax_min=int(N_BIAX_MIN_SKIN),
                )
            }
        except Exception:
            _orient_bounds = None

    do_stage = SectionOptimisationStage(
        params=SectionOptimisationParams(
            inp=inp_struct, out_dir=job,
            blade_yaml=blade_path_metadata, orchestration=orch,
            run_blade_optimizer=bool(DESIGN_OPTIMISE),
            optimization_objective=design_objective,
            optimizer_max_iter=int(DESIGN_MAX_ITER),
            bg_override=bg_struct,
            grid_meta={"type": "design", "structural_linspace": sspec},
            design_n_workers=int(grid_cfg.design_n_workers),
            section_properties=sp if SEED_DESIGN_FROM_SECTION_PROPERTIES else None,
            seed_section_properties=bool(SEED_DESIGN_FROM_SECTION_PROPERTIES),
            ks_rho=float(KS_RHO),
            enable_panel_buckling=bool(ENABLE_PANEL_BUCKLING),
            ks_rho_buckling=float(KS_RHO_BUCKLING),
            enable_global_buckling=bool(ENABLE_GLOBAL_BUCKLING),
            global_buckling_lambda_min=float(GLOBAL_BUCKLING_LAMBDA_MIN),
            n_global_buckling_modes=int(N_GLOBAL_BUCKLING_MODES),
            orientation_bounds=_orient_bounds,
            enforce_spanwise_monotone=bool(ENFORCE_MONOTONE_THICKNESS),
            debug_stress_projection=bool(DEBUG_STRESS_PROJECTION),
            beam_driver=str(BEAM_DRIVER),
            distributed_loads_inp=inp_geom
            if str(BEAM_DRIVER).lower() in ("global_beam", "coupled_fe")
            else None,
            axial_loading=axial_loading_cfg,
            n_beam_nodes=int(N_BEAM_NODES),
            stress_recovery=str(STRESS_RECOVERY),
            mitc4_n_elements_per_panel=int(MITC4_N_ELEMENTS_PER_PANEL),
            optimizer_method=str(DESIGN_OPTIMIZER_METHOD),
            optimizer_ftol=float(DESIGN_OPTIMIZER_FTOL),
            optimizer_n_restarts=int(DESIGN_OPTIMIZER_N_RESTARTS),
            optimizer_multistart_seed=DESIGN_OPTIMIZER_MULTISTART_SEED,
            iteration_dump_npz=bool(SECTION_OPTIM_DUMP_ITERATION_NPZ),
            iteration_hotspot_k=int(SECTION_OPTIM_ITERATION_HOTSPOT_K),
            iteration_emit_schema=bool(SECTION_OPTIM_EMIT_ITERATION_SCHEMA),
            progress=progress_rpt,
            iteration_pipeline_snapshots=bool(ITERATION_PIPELINE_SNAPSHOTS),
            iteration_snapshot_dpi=int(ITERATION_SNAPSHOT_DPI),
            iteration_snapshot_pngs=bool(ITERATION_SNAPSHOT_PNGS),
            iteration_snapshot_max=ITERATION_SNAPSHOT_MAX,
            iteration_snapshot_stride=int(ITERATION_SNAPSHOT_STRIDE),
            iteration_snapshot_grid_bundle=_iter_snapshot_bundle,
            iteration_snapshot_beam_inp=(
                inp_geom
                if ITERATION_PIPELINE_SNAPSHOTS
                and str(BEAM_DRIVER).lower() in ("global_beam", "coupled_fe")
                else None
            ),
        )
    )
    do = do_stage.execute().get_results()
    stage_seconds["section_optimisation_s"] = float(time.perf_counter() - t_stage_start)
    progress_rpt.phase_end("section_optimisation")

    # ------------------------------------------------------------------
    # POST-LOOP: re-render all upstream stages with the converged design
    # Only runs when DESIGN_OPTIMISE=True; subdirs get a /final/ suffix.
    # ------------------------------------------------------------------
    sg_final = sg
    sh_final = sh
    sp_final = sp
    bm_final = bm
    dv_resolved_source = "initial_dv0_no_optimisation"

    if _defer_pngs:
        # Resolve which DV to use: converged > best-so-far > initial
        _opt_result = getattr(do, "_opt_result_internal", None)
        # Retrieve opt_result from the stage's stored result JSON for provenance
        import json as _json_mod
        try:
            with open(do.result_json, "r", encoding="utf-8") as _rf:
                _do_data = _json_mod.load(_rf)
            _opt_ran = bool(_do_data.get("blade_optimizer_ran", False))
            _opt_success = bool(_do_data.get("blade_optimizer", {}).get("success", False))
        except Exception:
            _opt_ran = False
            _opt_success = False

        # BladeOptimizer stores dv_best_so_far on the result; access via optimizer internals
        # We re-read the opt result from the orchestration stage internal state
        _dv_resolved = None
        _resolve_note = "initial_dv0_fallback"

        # Try to get the resolved DV from the optimisation stage outputs
        # section_optimisation_impl stores the opt_res — we expose it via design_eval.json
        try:
            from blade_precompute.orchestration.precompute.stages import default_dv0
            from blade_precompute.section_optimisation.api import BladeDesignProblem
            _bg_tmp = bg_struct
            _n_st = int(np.asarray(_bg_tmp.z_stations, dtype=np.float64).ravel().shape[0])
            _dv0_fallback = default_dv0(_n_st)

            # Parse dv_opt from the design_eval.json if optimisation ran
            if _opt_ran and "optimised" in _do_data:
                _dv_dict = _do_data["optimised"].get("dv", {})
                _t_skin = np.asarray(_dv_dict.get("t_skin", _dv0_fallback.t_skin), dtype=np.float64)
                _t_cap = np.asarray(_dv_dict.get("t_cap", _dv0_fallback.t_cap), dtype=np.float64)
                _t_web = np.asarray(_dv_dict.get("t_web", _dv0_fallback.t_web), dtype=np.float64)
                from blade_precompute.section_optimisation.core.types import DesignVector as _DV
                _dv_resolved = _DV(t_skin=_t_skin, t_cap=_t_cap, t_web=_t_web)
                _resolve_note = "dv_opt_converged" if _opt_success else "dv_opt_last_accepted"
            else:
                _dv_resolved = _dv0_fallback
                _resolve_note = "dv0_no_optimisation"
        except Exception as _exc:
            import warnings
            warnings.warn(f"Post-loop DV resolution failed ({_exc}); using default_dv0.", stacklevel=1)

        if _dv_resolved is None:
            from blade_precompute.orchestration.precompute.stages import default_dv0 as _dv0_fn
            _dv_resolved = _dv0_fn(int(np.asarray(bg_struct.z_stations, dtype=np.float64).ravel().shape[0]))
            _resolve_note = "dv0_resolution_error_fallback"

        dv_resolved_source = _resolve_note
        bg_final_geom = apply_dv_to_bg(bg_struct, _dv_resolved)

        progress_rpt.phase_start("section_geometry_final")
        t_stage_start = time.perf_counter()
        sg_final = SectionGeometryStage(
            params=_build_sg_params(persist_pngs=True, subdir_override=Path("section_geometry/final"))
        ).execute().get_results()
        stage_seconds["section_geometry_final_s"] = float(time.perf_counter() - t_stage_start)
        progress_rpt.phase_end("section_geometry_final")

        progress_rpt.phase_start("section_properties_final")
        t_stage_start = time.perf_counter()
        sp_final = SectionPropertiesStage(
            params=_build_sp_params(bg_final_geom, persist_pngs=True, subdir_override=Path("section_properties/final"))
        ).execute().get_results()
        stage_seconds["section_properties_final_s"] = float(time.perf_counter() - t_stage_start)
        progress_rpt.phase_end("section_properties_final")

        progress_rpt.phase_start("global_beam_model_final")
        t_stage_start = time.perf_counter()
        bm_final = BeamModelStage(
            params=_build_bm_params(bg_final_geom, sp_final, persist_pngs=True, subdir_override=Path("global_beam_model/final"))
        ).execute().get_results()
        stage_seconds["global_beam_model_final_s"] = float(time.perf_counter() - t_stage_start)
        progress_rpt.phase_end("global_beam_model_final")

        progress_rpt.phase_start("section_shell_model_final")
        t_stage_start = time.perf_counter()
        sh_final = _run_shell_stage(
            bm_final, bg_final_geom,
            persist_pngs=True,
            subdir_override=Path("section_shell_model/final"),
            loads_provenance=f"real_{_resolve_note}",
        )
        stage_seconds["section_shell_model_final_s"] = float(time.perf_counter() - t_stage_start)
        progress_rpt.phase_end("section_shell_model_final")

    progress_rpt.phase_start("summary_runtime")
    t_stage_start = time.perf_counter()
    write_json(
        job / "summary.json",
        {
            "job_dir": job,
            "system_type_key": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "pre_loop": {
                "section_geometry": sg,
                "section_shell_model": sh,
                "section_properties": sp,
                "global_beam_model": bm,
            },
            "final": {
                "section_geometry": sg_final,
                "section_shell_model": sh_final,
                "section_properties": sp_final,
                "global_beam_model": bm_final,
                "dv_resolved_source": dv_resolved_source,
            } if _defer_pngs else None,
            "section_optimisation": do,
            # Backwards-compat aliases (point to final when optimised, else pre-loop)
            "section_geometry": sg_final,
            "section_shell_model": sh_final,
            "section_properties": sp_final,
            "global_beam_model": bm_final,
            "loads_provenance": {
                "unit_loads_in_pipeline": False,
                "shell_model_pre_loop": "real_initial_dv",
                "shell_model_final": f"real_{dv_resolved_source}" if _defer_pngs else "real_initial_dv",
                "audit": (
                    "Extreme/hydro loads from .dat; axial N and q_x from section_properties mass/length, "
                    "U_INF, TSR, and radial_r_m (see inputs.json axial_loading). "
                    "homogenisation.py uses unit strains for K7 extraction. "
                    "AnalysisConfig.interlaminar_vz=1 is API-only, not in main pipeline."
                ),
            },
            "axial_loading": axial_manifest,
        },
    )
    stage_seconds["summary_s"] = float(time.perf_counter() - t_stage_start)
    total_wall_s = float(time.perf_counter() - t_main_start)
    progress_rpt.phase_end("summary_runtime", total_wall_s=round(total_wall_s, 3))
    if _live_progress_ok:
        progress_rpt.event("job_complete", total_wall_s=round(total_wall_s, 3), job=str(job))

    write_json(
        job / "runtime.json",
        runtime_statistics_manifest(
            stage_seconds=stage_seconds,
            total_wall_s=total_wall_s,
            run_section_shell_model=bool(grid_cfg.run_section_shell_model),
            section_shell_skipped=bool(sh_final.skipped),
            section_geometry_station_count=len(sg_final.station_indices),
            section_shell_station_count=len(sh_final.station_indices),
            section_properties_station_count=int(np.asarray(sp_final.station_z, dtype=np.float64).shape[0]),
            beam_converged=bm_final.beam_converged,
            beam_n_iterations=bm_final.beam_n_iterations,
            optimizer_ran=do.optimizer_ran,
            optimizer_n_iter=do.optimizer_n_iter,
            python_version=sys.version,
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
            finished_at_iso=datetime.now().isoformat(timespec="seconds"),
            grid_cfg=grid_cfg,
        ),
    )
    _job_exit_state["completed"] = True
    write_json(
        job / "job_exit.json",
        {
            "status": "completed",
            "recorded_at_iso": datetime.now().isoformat(timespec="seconds"),
            "total_wall_s": round(total_wall_s, 3),
            "runtime_json": str(job / "runtime.json"),
        },
    )

    print(f"{CONSOLE_LOG_PREFIX} job_dir={job}", flush=True)
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
