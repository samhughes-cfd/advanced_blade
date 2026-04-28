"""Resample high-resolution blade spec to fewer spanwise stations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blade_precompute.section_optimisation.io.blade_geometry_loader import load_blade_geometry
from blade_precompute.section_optimisation.io.resample_blade_spec import resample_blade_spec


def test_resample_blade_spec_hires_to_10_matches_loader(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    hires_path = root / "example_blade_hires.json"
    if not hires_path.is_file():
        pytest.skip("example_blade_hires.json not in repo root")
    raw = json.loads(hires_path.read_text(encoding="utf-8"))
    out = resample_blade_spec(raw, n_stations=10)
    p = tmp_path / "blade.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    g = load_blade_geometry(p)
    assert g.z_stations.shape == (10,)
    assert g.r_ref.shape == (10, 3)
    assert g.chord.shape == (10,)
    assert g.twist.shape == (10,)
    assert g.kappa0.shape == (10, 3)
    assert len(g.airfoil_profiles) == 10


def test_resample_preserves_r_ref_z_column() -> None:
    root = Path(__file__).resolve().parents[1]
    hires_path = root / "example_blade_hires.json"
    if not hires_path.is_file():
        pytest.skip("example_blade_hires.json not in repo root")
    raw = json.loads(hires_path.read_text(encoding="utf-8"))
    out = resample_blade_spec(raw, n_stations=10)
    z = np.asarray(out["blade"]["z_stations"], dtype=np.float64)
    r2 = np.asarray(out["blade"]["r_ref"], dtype=np.float64)[:, 2]
    np.testing.assert_allclose(r2, z, rtol=0, atol=1e-12)
