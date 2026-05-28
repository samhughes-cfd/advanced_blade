"""Regression tests for beam-result JSON load handoff to section-shell stages."""

from __future__ import annotations

import json

import numpy as np

from blade_precompute.orchestration.precompute.containers import BeamModelOutputs, PrecomputeInputs
from blade_precompute.orchestration.precompute.stages import _station_resultants_for_shell_from_beam


def _minimal_inputs(z: np.ndarray, tmp_path) -> PrecomputeInputs:
    n = int(z.shape[0])
    zeros = np.zeros(n, dtype=np.float64)
    return PrecomputeInputs(
        spanwise_path=tmp_path / "spanwise.dat",
        extreme_loads_path=tmp_path / "loads.dat",
        span_r_z_m=z,
        radial_r_m=z.copy(),
        chord_m=np.ones(n, dtype=np.float64),
        twist_deg=zeros.copy(),
        kappa0_x=zeros.copy(),
        kappa0_y=zeros.copy(),
        kappa0_z=zeros.copy(),
        naca_m=zeros.copy(),
        naca_p=zeros.copy(),
        naca_xx=np.full(n, 12.0, dtype=np.float64),
        naca_series=np.full(n, 4, dtype=np.int64),
        loads_r_z_m=z.copy(),
        q_y_Npm=zeros.copy(),
        q_z_Npm=zeros.copy(),
        m_x_Nmpm=zeros.copy(),
    )


def test_station_resultants_for_shell_reads_beam_json_order(tmp_path) -> None:
    z = np.array([0.0, 1.0], dtype=np.float64)
    result_json = tmp_path / "beam_result.json"
    result_json.write_text(
        json.dumps(
            {
                "z_stations_out": z.tolist(),
                # Beam solver/native JSON order: N, Vy, Vz, My, Mz, T, B.
                "resultants": [
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                    [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
                ],
            }
        ),
        encoding="utf-8",
    )

    out = _station_resultants_for_shell_from_beam(
        BeamModelOutputs(result_json=result_json, png_paths=[]),
        _minimal_inputs(z, tmp_path),
    )

    assert out == {
        0: (10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        1: (11.0, 21.0, 31.0, 41.0, 51.0, 61.0),
    }
