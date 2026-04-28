from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blade_precompute.orchestration.precompute.containers import PrecomputeInputs
from blade_precompute.global_beam_model.engine.shell_enrichment import shell_recovery_payload


def _tiny_inp() -> PrecomputeInputs:
    z = np.linspace(0.0, 10.0, 5, dtype=np.float64)
    return PrecomputeInputs(
        spanwise_path=Path("sw"),
        extreme_loads_path=Path("el"),
        span_r_z_m=z,
        radial_r_m=np.linspace(0.0, 10.0, z.size, dtype=np.float64),
        chord_m=np.full_like(z, 1.0),
        twist_deg=np.zeros_like(z),
        naca_m=np.zeros_like(z),
        naca_p=np.full_like(z, 4.0),
        naca_xx=np.full_like(z, 12.0),
        naca_series=np.full_like(z, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 10.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )


def test_shell_recovery_payload_skipped_empty_spars() -> None:
    res = SimpleNamespace(z_stations_out=np.array([0.0, 5.0]), resultants=np.zeros((2, 7)))
    out = shell_recovery_payload(res, _tiny_inp(), np.array([2.0]), [], n_elements_per_panel=2)
    assert out["skipped"] is True
    assert out["reason"] == "no_shear_webs_for_thin_wall_shell"


def test_shell_recovery_payload_skipped_no_z_stations_out() -> None:
    res = SimpleNamespace(z_stations_out=None, resultants=np.zeros((1, 7)))
    out = shell_recovery_payload(res, _tiny_inp(), np.array([1.0]), [0.15, 0.5], n_elements_per_panel=2)
    assert out["skipped"] is True


@pytest.mark.parametrize("n_elem", [2, 3])
def test_shell_recovery_payload_one_station_smoke(n_elem: int) -> None:
    zg = np.linspace(0.0, 10.0, 8, dtype=np.float64)
    R = np.zeros((zg.shape[0], 7), dtype=np.float64)
    R[:, 0] = 1.0e3
    R[:, 5] = 50.0
    res = SimpleNamespace(z_stations_out=zg, resultants=R)
    inp = _tiny_inp()
    z_station = np.array([float(np.mean(zg))], dtype=np.float64)
    out = shell_recovery_payload(
        res,
        inp,
        z_station,
        [0.15, 0.5],
        n_elements_per_panel=n_elem,
    )
    assert out.get("skipped") is False
    assert out["n_stations"] == 1
    assert len(out["stations"]) == 1
    row = out["stations"][0]
    assert "max_clpt_hashin_fi" in row
    assert float(row["max_clpt_hashin_fi"]) >= 0.0
    assert len(row["beam_resultants_N_Vy_Vz_My_Mz_T_B"]) == 7
