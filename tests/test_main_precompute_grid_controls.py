from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch
import os

import numpy as np
import json

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import (
    LinspaceSpec,
    PrecomputeInputs,
    linspace_from_spec,
    resample_precompute_inputs,
    station_indices,
    station_subdir_name,
)
from blade_precompute.orchestration.precompute.stages import section_geometry_impl
from blade_precompute.orchestration.system_layout import resolve_system_type


def test_station_indices_all_and_every_k() -> None:
    assert station_indices(5, "all") == [0, 1, 2, 3, 4]
    assert station_indices(5, "structural") == [0, 1, 2, 3, 4]
    assert station_indices(8, "every-3") == [0, 3, 6]
    assert station_indices(8, "root,every-3,tip") == [0, 3, 6, 7]


def test_station_subdir_name_format() -> None:
    assert station_subdir_name(0, 0.0) == "station_i000_z0.000"
    assert station_subdir_name(12, 4.5) == "station_i012_z4.500"


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="2D-CN",
        layout=resolve_system_type("2D-CN"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


@patch("blade_precompute.section_geometry.interface.plot.plot_section")
@patch("blade_precompute.section_geometry.geometry.grid.SDFGrid")
@patch("blade_precompute.section_geometry.interface.export.SectionPropertiesReport")
def test_section_geometry_impl_writes_under_station_subdirs(
    mock_report_cls: MagicMock,
    mock_sdf_cls: MagicMock,
    mock_plot: MagicMock,
    tmp_path: Path,
) -> None:
    """Geometry reports and section PNGs are written under ``station_i*_z*`` (SDF/plot mocked for speed)."""
    mock_plot.return_value = (MagicMock(), None)
    mock_grid = MagicMock()
    mock_grid.eval.return_value = np.array([1.0], dtype=np.float64)
    mock_sdf_cls.from_airfoil.return_value = mock_grid
    mock_inst = MagicMock()

    def _to_json(path: Path, job_meta: object | None = None) -> None:
        Path(path).write_text("{}", encoding="utf-8")

    mock_inst.to_json.side_effect = _to_json
    mock_report_cls.return_value = mock_inst

    inp = PrecomputeInputs(
        spanwise_path=Path("span.dat"),
        extreme_loads_path=Path("loads.dat"),
        span_r_z_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        radial_r_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6, 1.2], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
        naca_m=np.array([0.0, 2.0, 4.0], dtype=np.float64),
        naca_p=np.array([0.0, 4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )
    out = section_geometry_impl(
        inp,
        tmp_path,
        section_plot_station_spec="root",
        orchestration=_orch(),
        grid_meta={"type": "structural", "linspace": None},
    )
    sub = station_subdir_name(0, 0.0)
    stage = tmp_path / "section_geometry"
    assert (stage / sub / "geometry_report_i000_rz0.000.json").is_file()
    assert out.png_paths
    assert out.png_paths[0].parent.name == sub


def test_geometry_resample_uses_linspace_count() -> None:
    inp = PrecomputeInputs(
        spanwise_path="a",  # type: ignore[arg-type]
        extreme_loads_path="b",  # type: ignore[arg-type]
        span_r_z_m=np.array([0.0, 4.0, 8.0]),
        radial_r_m=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6, 1.2]),
        twist_deg=np.array([0.0, 0.0, 0.0]),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
        naca_m=np.array([0.0, 2.0, 4.0]),
        naca_p=np.array([0.0, 4.0, 4.0]),
        naca_xx=np.array([12.0, 12.0, 12.0]),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0]),
        q_y_Npm=np.array([1.0, 1.0]),
        q_z_Npm=np.array([2.0, 2.0]),
        m_x_Nmpm=np.array([3.0, 3.0]),
    )
    z = linspace_from_spec(LinspaceSpec(0.0, 8.0, 5))
    out = resample_precompute_inputs(inp, z)
    assert out.span_r_z_m.shape == (5,)
    np.testing.assert_allclose(out.span_r_z_m, np.linspace(0.0, 8.0, 5))
    np.testing.assert_allclose(out.radial_r_m, np.interp(z, [0, 4, 8], [0, 1, 2]))
    assert out.loads_r_z_m.shape == (2,)
    assert out.naca_series.shape == (5,)
    assert np.all(np.isin(out.naca_series, (4, 5, 6)))


@patch("blade_precompute.orchestration.precompute.stages._section_geometry_station_task")
@patch("blade_precompute.orchestration.precompute.stages.ProcessPoolExecutor")
def test_section_geometry_impl_parallel_path_uses_station_workers(
    mock_pool_cls: MagicMock,
    mock_station_task: MagicMock,
    tmp_path: Path,
) -> None:
    """When section_solve_n_workers > 1, stage uses process fan-out path."""

    class _Pool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kw):
            fut: Future[dict] = Future()
            fut.set_result(fn(**kw))
            return fut

    mock_pool_cls.return_value = _Pool()

    def _task(**kw):
        i = int(kw["i"])
        rz = float(kw["rz"])
        station_dir = tmp_path / "section_geometry" / kw["station_subdir"]
        station_dir.mkdir(parents=True, exist_ok=True)
        props = station_dir / f"geometry_report_i{i:03d}_rz{rz:.3f}.json"
        props.write_text("{}", encoding="utf-8")
        png = station_dir / f"section_i{i:03d}_rz{rz:.3f}.png"
        png.write_text("", encoding="utf-8")
        return {
            "i": i,
            "r_z_m": rz,
            "station_subdir": kw["station_subdir"],
            "props_json": props,
            "png": png,
        }

    mock_station_task.side_effect = _task

    inp = PrecomputeInputs(
        spanwise_path=Path("span.dat"),
        extreme_loads_path=Path("loads.dat"),
        span_r_z_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        radial_r_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6, 1.2], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
        naca_m=np.array([0.0, 2.0, 4.0], dtype=np.float64),
        naca_p=np.array([0.0, 4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )
    out = section_geometry_impl(
        inp,
        tmp_path,
        section_plot_station_spec="all",
        orchestration=_orch(),
        grid_meta={"type": "geometry"},
        section_solve_n_workers=2,
    )
    assert mock_pool_cls.called
    assert mock_station_task.call_count == 3
    assert len(out.geometry_report_json_paths) == 3
    assert len(out.png_paths) == 3


@patch("blade_precompute.section_geometry.interface.plot.plot_section")
@patch("blade_precompute.section_geometry.geometry.grid.SDFGrid")
@patch("blade_precompute.section_geometry.interface.export.SectionPropertiesReport")
def test_section_geometry_impl_records_ir_flag_in_summary(
    mock_report_cls: MagicMock,
    mock_sdf_cls: MagicMock,
    mock_plot: MagicMock,
    tmp_path: Path,
) -> None:
    mock_plot.return_value = (MagicMock(), None)
    mock_grid = MagicMock()
    mock_grid.eval.return_value = np.array([1.0], dtype=np.float64)
    mock_sdf_cls.from_airfoil.return_value = mock_grid
    mock_inst = MagicMock()

    def _to_json(path: Path, job_meta: object | None = None) -> None:
        Path(path).write_text("{}", encoding="utf-8")

    mock_inst.to_json.side_effect = _to_json
    mock_report_cls.return_value = mock_inst

    inp = PrecomputeInputs(
        spanwise_path=Path("span.dat"),
        extreme_loads_path=Path("loads.dat"),
        span_r_z_m=np.array([0.0, 4.0], dtype=np.float64),
        radial_r_m=np.array([0.0, 4.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6], dtype=np.float64),
        twist_deg=np.zeros(2, dtype=np.float64),
        kappa0_x=np.zeros(2, dtype=np.float64),
        kappa0_y=np.zeros(2, dtype=np.float64),
        kappa0_z=np.zeros(2, dtype=np.float64),
        naca_m=np.array([0.0, 2.0], dtype=np.float64),
        naca_p=np.array([0.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0], dtype=np.float64),
        naca_series=np.full(2, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )

    old = os.environ.get("SECTION_GEOMETRY_USE_CSG_IR")
    os.environ["SECTION_GEOMETRY_USE_CSG_IR"] = "1"
    try:
        section_geometry_impl(
            inp,
            tmp_path,
            section_plot_station_spec="all",
            orchestration=_orch(),
            grid_meta={"type": "structural", "linspace": None},
        )
    finally:
        if old is None:
            os.environ.pop("SECTION_GEOMETRY_USE_CSG_IR", None)
        else:
            os.environ["SECTION_GEOMETRY_USE_CSG_IR"] = old

    summary_path = tmp_path / "section_geometry" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["section_geometry_eval"]["use_csg_ir"] is True
