from __future__ import annotations

import numpy as np

from blade_precompute.orchestration.precompute import (
    LinspaceSpec,
    PrecomputeInputs,
    linspace_from_spec,
    resample_precompute_inputs,
    station_indices,
)


def test_station_indices_all_and_every_k() -> None:
    assert station_indices(5, "all") == [0, 1, 2, 3, 4]
    assert station_indices(8, "every-3") == [0, 3, 6]
    assert station_indices(8, "root,every-3,tip") == [0, 3, 6, 7]


def test_geometry_resample_uses_linspace_count() -> None:
    inp = PrecomputeInputs(
        spanwise_path="a",  # type: ignore[arg-type]
        extreme_loads_path="b",  # type: ignore[arg-type]
        span_r_z_m=np.array([0.0, 4.0, 8.0]),
        chord_m=np.array([2.0, 1.6, 1.2]),
        twist_deg=np.array([0.0, 0.0, 0.0]),
        naca_m=np.array([0.0, 2.0, 4.0]),
        naca_p=np.array([0.0, 4.0, 4.0]),
        naca_xx=np.array([12.0, 12.0, 12.0]),
        loads_r_z_m=np.array([0.0, 8.0]),
        q_y_Npm=np.array([1.0, 1.0]),
        q_z_Npm=np.array([2.0, 2.0]),
        m_x_Nmpm=np.array([3.0, 3.0]),
    )
    z = linspace_from_spec(LinspaceSpec(0.0, 8.0, 5))
    out = resample_precompute_inputs(inp, z)
    assert out.span_r_z_m.shape == (5,)
    np.testing.assert_allclose(out.span_r_z_m, np.linspace(0.0, 8.0, 5))
    assert out.loads_r_z_m.shape == (2,)
