from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import (
    PrecomputeInputs,
    SectionShellModelOutputs,
    SectionShellModelParams,
    SectionShellModelStage,
)
from blade_precompute.orchestration.system_layout import resolve_system_type


def _dummy_inputs() -> PrecomputeInputs:
    return PrecomputeInputs(
        spanwise_path=Path("x"),
        extreme_loads_path=Path("y"),
        span_r_z_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        chord_m=np.array([2.0, 1.6, 1.2], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        naca_m=np.array([0.0, 2.0, 4.0], dtype=np.float64),
        naca_p=np.array([0.0, 4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="legacy",
        layout=resolve_system_type("legacy"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


@patch("blade_precompute.orchestration.precompute.stage_facade.section_shell_model_impl")
def test_section_shell_model_stage_execute(mock_impl: object, tmp_path: Path) -> None:
    summary = tmp_path / "section_shell_model" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    mock_impl.return_value = SectionShellModelOutputs(
        station_indices=[0],
        station_r_z_m=[0.0],
        png_paths=[tmp_path / "a.png"],
        summary_json=summary,
        skipped=False,
    )
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            plot_station_spec="root",
            orchestration=_orch(),
        )
    )
    st.execute()
    r = st.get_results()
    assert not r.skipped
    assert r.png_paths
    mock_impl.assert_called_once()


def test_section_shell_model_get_results_requires_execute() -> None:
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=Path("."),
            plot_station_spec="root",
            orchestration=_orch(),
        )
    )
    with pytest.raises(RuntimeError):
        st.get_results()
