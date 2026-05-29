from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_precompute.global_beam_model.engine.axial_loading import AxialLoadingConfig
from blade_precompute.orchestration.precompute.stages import (
    _add_axial_loading_to_distributed_q,
    _blend_shell_k7_with_strip,
)


def test_shell_k7_blend_keeps_strip_stiffness_for_failed_stations() -> None:
    strip = np.stack([np.eye(7), np.eye(7) * 2.0], axis=0)
    shell = np.stack([np.eye(7) * 10.0, np.zeros((7, 7))], axis=0)
    per_station = [
        {"station_index": 0, "ok": True},
        {"station_index": 1, "ok": False, "error": "assembly failed"},
    ]

    blended = _blend_shell_k7_with_strip(shell, strip, per_station, relax=1.0)

    np.testing.assert_allclose(blended[0], shell[0])
    np.testing.assert_allclose(blended[1], strip[1])


def test_beam_stage_axial_loading_adds_spanwise_line_load() -> None:
    inp = SimpleNamespace(
        span_r_z_m=np.array([0.0, 10.0], dtype=np.float64),
        radial_r_m=np.array([2.0, 12.0], dtype=np.float64),
    )
    sec = SimpleNamespace(
        station_z=np.array([0.0, 10.0], dtype=np.float64),
        section_results=(
            SimpleNamespace(mass_per_length=3.0),
            SimpleNamespace(mass_per_length=5.0),
        ),
    )
    cfg = AxialLoadingConfig(
        u_inf_m_s=2.0,
        tip_speed_ratio=10.0,
        r_tip_m=10.0,
        gravity_m_s2=0.0,
        azimuth_deg=90.0,
        enabled=True,
    )
    z_mid = np.array([2.5, 7.5], dtype=np.float64)
    q = np.zeros((2, 3), dtype=np.float64)
    q[:, 2] = 100.0

    out = _add_axial_loading_to_distributed_q(q, z_mid, inp, sec, cfg, span_axis=2)

    expected_mu = np.array([3.5, 4.5], dtype=np.float64)
    expected_r = np.array([4.5, 9.5], dtype=np.float64)
    np.testing.assert_allclose(out[:, 2], 100.0 + expected_mu * (2.0**2) * expected_r)
    np.testing.assert_allclose(out[:, :2], 0.0)
