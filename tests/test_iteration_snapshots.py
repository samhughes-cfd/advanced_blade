from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import PrecomputeInputs
from blade_precompute.orchestration.precompute.containers import (
    BeamModelOutputs,
    SectionGeometryOutputs,
    SectionPropertiesOutputs,
    SectionShellModelOutputs,
)
from blade_precompute.orchestration.precompute.stages import run_pipeline_snapshot_to_dir
from blade_precompute.orchestration.system_layout import resolve_system_type


def _inp() -> PrecomputeInputs:
    return PrecomputeInputs(
        spanwise_path=Path("x"),
        extreme_loads_path=Path("y"),
        span_r_z_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        radial_r_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6, 1.2], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        naca_m=np.array([0.0, 2.0, 4.0], dtype=np.float64),
        naca_p=np.array([0.0, 4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
    )


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="2D-CN",
        layout=resolve_system_type("2D-CN"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


def _bundle() -> dict:
    return {
        "section_geometry": {},
        "section_properties": {},
        "beam": {},
        "section_shell": {},
        "section_plot_station_spec": "root",
        "section_solve_n_workers": 1,
        "n_beam_nodes": 5,
        "enable_shell_recovery_enrichment": False,
        "shell_recovery_n_elements_per_panel": 4,
        "beam_png_span_samples": 50,
        "n_elements_per_panel": 4,
        "use_mitc4_v2_path": False,
        "save_section_recovery_cache_npz": False,
    }


@pytest.fixture
def snapshot_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    from blade_precompute.orchestration.precompute import stages as st

    def _sg(inp, out_dir, **kwargs):
        d = Path(out_dir) / "section_geometry"
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text("sg", encoding="utf-8")
        gj = d / "g.json"
        gj.write_text("{}", encoding="utf-8")
        return SectionGeometryOutputs(
            station_indices=[0],
            station_r_z_m=[0.0],
            png_paths=[],
            geometry_report_json_paths=[gj],
        )

    def _sp(inp, out_dir, **kwargs):
        d = Path(out_dir) / "section_properties"
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text("sp", encoding="utf-8")
        sj = d / "section_solve_summary.json"
        sj.write_text("{}", encoding="utf-8")
        z = np.asarray(inp.span_r_z_m, dtype=np.float64)
        n = int(z.shape[0])
        return SectionPropertiesOutputs(
            station_z=z,
            K6=np.zeros((n, 6, 6), dtype=np.float64),
            K7=np.zeros((n, 7, 7), dtype=np.float64),
            results_summary_json=sj,
            png_paths=[],
            section_results=(),
            section_definitions=(),
        )

    def _bm(inp, sec, out_dir, **kwargs):
        d = Path(out_dir) / "global_beam_model"
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text("bm", encoding="utf-8")
        rj = d / "beam_result.json"
        rj.write_text(
            '{"resultants": [[0,0,0,0,0,0]], "z_stations_out": [0.0]}',
            encoding="utf-8",
        )
        return BeamModelOutputs(result_json=rj, png_paths=[], beam_n_iterations=1, beam_converged=True)

    def _sh(inp, out_dir, **kwargs):
        d = Path(out_dir) / "section_shell_model"
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker.txt").write_text("sh", encoding="utf-8")
        sj = d / "summary.json"
        sj.write_text("{}", encoding="utf-8")
        return SectionShellModelOutputs(
            station_indices=[0],
            station_r_z_m=[0.0],
            png_paths=[],
            summary_json=sj,
        )

    monkeypatch.setattr(st, "section_geometry_impl", _sg)
    monkeypatch.setattr(st, "section_properties_impl", _sp)
    monkeypatch.setattr(st, "beam_model_impl", _bm)
    monkeypatch.setattr(st, "section_shell_model_impl", _sh)


def test_run_pipeline_snapshot_creates_four_package_dirs(
    tmp_path: Path, snapshot_mocks: None
) -> None:
    root = tmp_path / "iter_0000"
    inp = _inp()
    run_pipeline_snapshot_to_dir(
        root,
        inp_struct=inp,
        inp_geom=inp,
        bg=MagicMock(),
        orchestration=_orch(),
        blade_yaml=Path("spanwise+material_library"),
        bundle=_bundle(),
        dpi=72,
        persist_pngs=False,
        loads_provenance="snapshot_iter_0000",
    )
    assert (root / "section_geometry" / "marker.txt").read_text(encoding="utf-8") == "sg"
    assert (root / "section_properties" / "marker.txt").read_text(encoding="utf-8") == "sp"
    assert (root / "global_beam_model" / "marker.txt").read_text(encoding="utf-8") == "bm"
    assert (root / "section_shell_model" / "marker.txt").read_text(encoding="utf-8") == "sh"
