from __future__ import annotations

import numpy as np
import pytest

from main_precompute import (
    PrecomputeInputs,
    SectionGeometryParams,
    SectionGeometryStage,
)


def _dummy_inputs() -> PrecomputeInputs:
    return PrecomputeInputs(
        spanwise_path="x",  # type: ignore[arg-type]
        extreme_loads_path="y",  # type: ignore[arg-type]
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


def test_stage_get_results_requires_execute() -> None:
    st = SectionGeometryStage(
        SectionGeometryParams(
            inp=_dummy_inputs(),
            out_dir=".",  # type: ignore[arg-type]
            plot_station_spec="root",
            orchestration=None,  # type: ignore[arg-type]
        )
    )
    with pytest.raises(RuntimeError):
        st.get_results()
