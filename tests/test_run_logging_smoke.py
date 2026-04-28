from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    BeamModelParams,
    PrecomputeInputs,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionOptimisationOutputs,
    SectionOptimisationParams,
    SectionPropertiesOutputs,
    SectionPropertiesParams,
    SectionShellModelOutputs,
    SectionShellModelParams,
)
from blade_precompute.orchestration.precompute.stage_facade import (
    BeamModelStage,
    SectionGeometryStage,
    SectionOptimisationStage,
    SectionPropertiesStage,
    SectionShellModelStage,
)
from blade_precompute.orchestration.precompute.stages import section_shell_model_skipped_outputs


def _minimal_inputs(tmp_path: Path) -> PrecomputeInputs:
    z = np.array([0.0, 1.0], dtype=np.float64)
    return PrecomputeInputs(
        spanwise_path=tmp_path / "span.dat",
        extreme_loads_path=tmp_path / "loads.dat",
        span_r_z_m=z,
        radial_r_m=z,
        chord_m=np.array([1.0, 1.0], dtype=np.float64),
        twist_deg=np.array([0.0, 0.0], dtype=np.float64),
        kappa0_x=np.zeros(2, dtype=np.float64),
        kappa0_y=np.zeros(2, dtype=np.float64),
        kappa0_z=np.zeros(2, dtype=np.float64),
        naca_m=np.array([2.0, 2.0], dtype=np.float64),
        naca_p=np.array([4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0], dtype=np.float64),
        naca_series=np.array([4, 4], dtype=np.int64),
        loads_r_z_m=z,
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
        log_dump_level="intermediate",
    )


def test_stage_facades_write_required_run_logs(tmp_path: Path, monkeypatch: Any) -> None:
    inp = _minimal_inputs(tmp_path)
    out_dir = tmp_path / "job"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fake implementations keep the smoke test deterministic and fast.
    def _sg_impl(*args: Any, **kwargs: Any) -> SectionGeometryOutputs:
        return SectionGeometryOutputs([], [], [], [])

    def _sp_impl(*args: Any, **kwargs: Any) -> SectionPropertiesOutputs:
        return SectionPropertiesOutputs(
            station_z=np.array([0.0], dtype=np.float64),
            K6=np.zeros((1, 6, 6), dtype=np.float64),
            K7=np.zeros((1, 7, 7), dtype=np.float64),
            results_summary_json=out_dir / "section_properties" / "summary.json",
            png_paths=[],
            section_results=(),
            section_definitions=(),
        )

    def _bm_impl(*args: Any, **kwargs: Any) -> BeamModelOutputs:
        return BeamModelOutputs(
            result_json=out_dir / "global_beam_model" / "beam_result.json",
            png_paths=[],
            beam_n_iterations=0,
            beam_converged=True,
        )

    def _so_impl(*args: Any, **kwargs: Any) -> SectionOptimisationOutputs:
        return SectionOptimisationOutputs(
            result_json=out_dir / "section_optimisation" / "design_eval.json",
            png_paths=[],
            optimizer_ran=False,
            optimizer_n_iter=None,
        )

    def _ssm_impl(*args: Any, **kwargs: Any) -> SectionShellModelOutputs:
        return SectionShellModelOutputs(
            station_indices=[],
            station_r_z_m=[],
            png_paths=[],
            summary_json=out_dir / "section_shell_model" / "summary.json",
            station_result_json_paths=[],
            skipped=False,
        )

    monkeypatch.setattr(
        "blade_precompute.orchestration.precompute.stage_facade.section_geometry_impl",
        _sg_impl,
    )
    monkeypatch.setattr(
        "blade_precompute.orchestration.precompute.stage_facade.section_properties_impl",
        _sp_impl,
    )
    monkeypatch.setattr(
        "blade_precompute.orchestration.precompute.stage_facade.beam_model_impl",
        _bm_impl,
    )
    monkeypatch.setattr(
        "blade_precompute.orchestration.precompute.stage_facade.section_optimisation_impl",
        _so_impl,
    )
    monkeypatch.setattr(
        "blade_precompute.orchestration.precompute.stage_facade.section_shell_model_impl",
        _ssm_impl,
    )

    SectionGeometryStage(
        params=SectionGeometryParams(inp=inp, out_dir=out_dir, section_plot_station_spec="all", orchestration=None)
    ).execute()
    shell_stage = SectionShellModelStage(
        params=SectionShellModelParams(inp=inp, out_dir=out_dir, section_plot_station_spec="all", orchestration=None)
    )
    shell_stage.execute()
    sp = SectionPropertiesStage(
        params=SectionPropertiesParams(
            inp=inp,
            out_dir=out_dir,
            blade_yaml=tmp_path / "dummy.json",
            section_plot_station_spec="all",
            orchestration=None,
        )
    ).execute().get_results()
    BeamModelStage(
        params=BeamModelParams(
            inp=inp,
            sec=sp,
            out_dir=out_dir,
            blade_yaml=tmp_path / "dummy.json",
            n_beam_nodes=4,
            orchestration=None,
        )
    ).execute()
    SectionOptimisationStage(
        params=SectionOptimisationParams(
            inp=inp,
            out_dir=out_dir,
            blade_yaml=tmp_path / "dummy.json",
            orchestration=None,
        )
    ).execute()

    required = [
        out_dir / "section_geometry" / "run.log",
        out_dir / "section_properties" / "run.log",
        out_dir / "section_shell_model" / "run.log",
        out_dir / "global_beam_model" / "run.log",
        out_dir / "section_optimisation" / "run.log",
    ]
    for path in required:
        assert path.is_file(), f"Missing required run log: {path}"
        body = path.read_text(encoding="utf-8")
        assert "stage.start" in body
        assert "stage.end" in body

    manifest = out_dir / "logs.manifest.json"
    assert manifest.is_file()
    assert manifest.read_text(encoding="utf-8").strip()


def test_shell_skipped_outputs_still_write_run_log(tmp_path: Path) -> None:
    class _OrchStub:
        @staticmethod
        def job_meta() -> dict[str, Any]:
            return {"smoke": True}

    out_dir = tmp_path / "job"
    out_dir.mkdir(parents=True, exist_ok=True)
    section_shell_model_skipped_outputs(
        out_dir,
        orchestration=_OrchStub(),
        reason="smoke-skip",
    )
    log_path = out_dir / "section_shell_model" / "run.log"
    assert log_path.is_file()
    body = log_path.read_text(encoding="utf-8")
    assert "stage.skipped" in body
