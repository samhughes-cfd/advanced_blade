"""Entry point: blade precompute pipeline (orchestration in blade_precompute.orchestration.precompute).

Edit the variables in the "Control settings" section below (not via CLI).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from blade_precompute.section_optimisation.__main__ import _objective_from_cli
from blade_precompute.section_optimisation.api import BladeDesignProblem

from blade_precompute.orchestration.precompute import (
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
    build_precompute_orchestration_context,
    linspace_from_spec,
    load_inputs,
    resample_blade_geometry_to_z,
    resample_precompute_inputs,
    resolve_component_materials_path,
    write_json,
)
from blade_precompute.orchestration.precompute.stages import section_shell_model_skipped_outputs

_REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Control settings (edit here)
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path("data_library")
BLADE_YAML: Path = Path("example_blade_10.yaml")
# GBT / member section buckling: use ``examples/section_buckling`` and ``examples/section_beam_model`` (not precompute).
OUTPUT_BASE_DIR: Path = _REPO_ROOT / "outputs"

N_BEAM_NODES: int = 50
SAVE_SECTION_RECOVERY_CACHE_NPZ: bool = False
PLOT_STATIONS: str = "root,mid,tip"
RUN_SECTION_SHELL_MODEL: bool = True
SYSTEM_TYPE: str = "legacy"
COMPONENT_MATERIALS: Path | None = None

DESIGN_OPTIMISE: bool = False
DESIGN_OBJECTIVE: str = "min-mass"
DESIGN_MAX_ITER: int = 120

# Geometry grid (section_geometry stage): if `*_N` is None, keep source table count.
GRID_GEOMETRY_Z_MIN: float | None = None
GRID_GEOMETRY_Z_MAX: float | None = None
GRID_GEOMETRY_N: int | None = None

# Structural/design station grid: if `*_N` is None, keep YAML station count.
GRID_STRUCTURAL_Z_MIN: float | None = None
GRID_STRUCTURAL_Z_MAX: float | None = None
GRID_STRUCTURAL_N: int | None = None


def _job_dir(base_out: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (base_out.resolve() / ts).resolve()


def main() -> int:
    design_objective = _objective_from_cli(DESIGN_OBJECTIVE)

    inp = load_inputs(DATA_DIR)
    job = _job_dir(OUTPUT_BASE_DIR)
    job.mkdir(parents=True, exist_ok=True)

    orch = build_precompute_orchestration_context(
        data_dir=DATA_DIR,
        blade_yaml=BLADE_YAML,
        system_type_key=str(SYSTEM_TYPE),
        component_materials_path=COMPONENT_MATERIALS,
    )

    z_geom_src = inp.span_r_z_m.ravel()
    gspec = LinspaceSpec(
        z_min=float(GRID_GEOMETRY_Z_MIN) if GRID_GEOMETRY_Z_MIN is not None else float(z_geom_src[0]),
        z_max=float(GRID_GEOMETRY_Z_MAX) if GRID_GEOMETRY_Z_MAX is not None else float(z_geom_src[-1]),
        n=int(GRID_GEOMETRY_N) if GRID_GEOMETRY_N is not None else int(z_geom_src.shape[0]),
    )
    inp_geom = resample_precompute_inputs(inp, linspace_from_spec(gspec))

    blade_yaml_resolved = BLADE_YAML.resolve()
    bg_raw = BladeDesignProblem.load_geometry(blade_yaml_resolved)
    z_struct_src = bg_raw.z_stations.ravel()
    sspec = LinspaceSpec(
        z_min=float(GRID_STRUCTURAL_Z_MIN) if GRID_STRUCTURAL_Z_MIN is not None else float(z_struct_src[0]),
        z_max=float(GRID_STRUCTURAL_Z_MAX) if GRID_STRUCTURAL_Z_MAX is not None else float(z_struct_src[-1]),
        n=int(GRID_STRUCTURAL_N) if GRID_STRUCTURAL_N is not None else int(z_struct_src.shape[0]),
    )
    bg_struct = resample_blade_geometry_to_z(bg_raw, linspace_from_spec(sspec))
    grid_cfg = GridConfig(
        geometry=gspec,
        structural=sspec,
        plot_station_spec=str(PLOT_STATIONS),
        n_beam_nodes=int(N_BEAM_NODES),
        run_section_shell_model=bool(RUN_SECTION_SHELL_MODEL),
        section_shell_n_elements_per_panel=12,
        section_shell_dpi=150,
    )

    write_json(
        job / "inputs.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "config_source": "main_precompute module variables",
            "spanwise_path": inp.spanwise_path,
            "extreme_loads_path": inp.extreme_loads_path,
            "blade_yaml": blade_yaml_resolved,
            "system_type": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "component_materials_path": resolve_component_materials_path(
                DATA_DIR, COMPONENT_MATERIALS
            ),
            "design_optimise": bool(DESIGN_OPTIMISE),
            "design_objective": design_objective,
            "design_max_iter": int(DESIGN_MAX_ITER),
            "run_section_shell_model": bool(grid_cfg.run_section_shell_model),
            "grid_config": grid_cfg,
        },
    )

    sg_stage = SectionGeometryStage(
        params=SectionGeometryParams(
            inp=inp_geom,
            out_dir=job,
            plot_station_spec=PLOT_STATIONS,
            orchestration=orch,
            grid_meta={"type": "geometry", "linspace": gspec},
        )
    )
    sg = sg_stage.execute().get_results()

    if grid_cfg.run_section_shell_model:
        sh_stage = SectionShellModelStage(
            params=SectionShellModelParams(
                inp=inp_geom,
                out_dir=job,
                plot_station_spec=PLOT_STATIONS,
                orchestration=orch,
                n_elements_per_panel=int(grid_cfg.section_shell_n_elements_per_panel),
                dpi=int(grid_cfg.section_shell_dpi),
                grid_meta={"type": "section_shell_model", "linspace": gspec},
            )
        )
        sh: SectionShellModelOutputs = sh_stage.execute().get_results()
    else:
        sh = section_shell_model_skipped_outputs(
            job,
            orchestration=orch,
            reason="run_section_shell_model=false",
            grid_meta={"type": "section_shell_model", "linspace": gspec},
        )

    sp_stage = SectionPropertiesStage(
        params=SectionPropertiesParams(
            inp=inp_geom,
            out_dir=job,
            blade_yaml=blade_yaml_resolved,
            plot_station_spec=PLOT_STATIONS,
            orchestration=orch,
            bg_override=bg_struct,
            grid_meta={"type": "structural", "linspace": sspec},
        )
    )
    sp = sp_stage.execute().get_results()

    bm_stage = BeamModelStage(
        params=BeamModelParams(
            inp=inp_geom,
            sec=sp,
            out_dir=job,
            blade_yaml=blade_yaml_resolved,
            n_beam_nodes=int(N_BEAM_NODES),
            orchestration=orch,
            save_section_recovery_cache_npz=bool(SAVE_SECTION_RECOVERY_CACHE_NPZ),
            bg_override=bg_struct,
            grid_meta={
                "type": "beam",
                "structural_linspace": sspec,
                "n_beam_nodes": int(N_BEAM_NODES),
            },
        )
    )
    bm = bm_stage.execute().get_results()

    do_stage = SectionOptimisationStage(
        params=SectionOptimisationParams(
            inp=inp_geom,
            out_dir=job,
            blade_yaml=blade_yaml_resolved,
            orchestration=orch,
            run_blade_optimizer=bool(DESIGN_OPTIMISE),
            optimization_objective=design_objective,
            optimizer_max_iter=int(DESIGN_MAX_ITER),
            bg_override=bg_struct,
            grid_meta={"type": "design", "structural_linspace": sspec},
        )
    )
    do = do_stage.execute().get_results()

    write_json(
        job / "summary.json",
        {
            "job_dir": job,
            "system_type": orch.system_type_key,
            "component_materials": orch.component_materials.to_dict(),
            "section_geometry": sg,
            "section_shell_model": sh,
            "section_properties": sp,
            "global_beam_model": bm,
            "section_optimisation": do,
        },
    )

    print(str(job))
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
