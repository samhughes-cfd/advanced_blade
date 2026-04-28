from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import (
    PrecomputeInputs,
    SectionGeometryOutputs,
    SectionGeometryParams,
    SectionGeometryStage,
)
from blade_precompute.orchestration.system_layout import resolve_system_type


def _dummy_inputs() -> PrecomputeInputs:
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


def test_stage_get_results_requires_execute() -> None:
    st = SectionGeometryStage(
        params=SectionGeometryParams(
            inp=_dummy_inputs(),
            out_dir=Path("."),
            section_plot_station_spec="root",
            orchestration=_orch(),
        )
    )
    with pytest.raises(RuntimeError):
        st.get_results()


@patch("blade_precompute.orchestration.precompute.stage_facade.section_geometry_impl")
def test_execute_returns_self_and_is_idempotent(mock_impl: object, tmp_path: Path) -> None:
    mock_impl.return_value = SectionGeometryOutputs(
        station_indices=[0],
        station_r_z_m=[0.0],
        png_paths=[],
        geometry_report_json_paths=[],
    )
    st = SectionGeometryStage(
        params=SectionGeometryParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch(),
        )
    )
    r1 = st.execute()
    assert r1 is st
    r2 = st.execute()
    assert r2 is st
    assert mock_impl.call_count == 1
