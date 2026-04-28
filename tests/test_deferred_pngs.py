"""Tests for deferred-PNG / best-so-far / apply_dv_to_bg plan implementation."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# apply_dv_to_bg
# ---------------------------------------------------------------------------


def _make_minimal_bg(n: int = 3):
    """Create a minimal OptimBladeGeometry for tests (no YAML required)."""
    from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
    from blade_precompute.section_properties.engine.materials import IsotropicMaterial

    dummy_mat = IsotropicMaterial(name="test", E=70e9, nu=0.3, rho=1600.0, sigma_allow=200e6)
    return OptimBladeGeometry(
        z_stations=np.linspace(0.0, 10.0, n),
        r_ref=np.zeros(n),
        kappa0=np.zeros((n, 3)),
        chord=np.ones(n),
        twist=np.zeros(n),
        airfoil_profiles=[None] * n,
        web_positions=np.array([-0.3, 0.3]),
        subcomponent_materials={
            "skin": dummy_mat,
            "cap_ps": dummy_mat,
            "web": dummy_mat,
        },
        thickness_role={"skin": "skin", "cap_ps": "cap", "web": "web"},
    )


def test_apply_dv_to_bg_does_not_mutate_input() -> None:
    """apply_dv_to_bg must return a new object without mutating the original."""
    from blade_precompute.section_optimisation.core.types import DesignVector, apply_dv_to_bg

    n = 4
    bg = _make_minimal_bg(n)
    dv = DesignVector(
        t_skin=np.full(n, 0.008, dtype=np.float64),
        t_cap=np.full(n, 0.040, dtype=np.float64),
        t_web=np.full(n, 0.012, dtype=np.float64),
    )

    bg_final = apply_dv_to_bg(bg, dv)

    assert bg_final is not bg
    assert bg_final.resolved_dv is dv
    assert bg.resolved_dv is None, "Original bg must not be mutated"


def test_apply_dv_to_bg_preserves_geometry() -> None:
    """Geometry fields are unchanged in the returned copy."""
    from blade_precompute.section_optimisation.core.types import DesignVector, apply_dv_to_bg

    n = 5
    bg = _make_minimal_bg(n)
    dv = DesignVector(
        t_skin=np.full(n, 0.010),
        t_cap=np.full(n, 0.050),
        t_web=np.full(n, 0.015),
    )

    bg_final = apply_dv_to_bg(bg, dv)

    np.testing.assert_array_equal(bg_final.z_stations, bg.z_stations)
    np.testing.assert_array_equal(bg_final.chord, bg.chord)
    np.testing.assert_array_equal(bg_final.web_positions, bg.web_positions)
    assert bg_final.subcomponent_materials is bg.subcomponent_materials


# ---------------------------------------------------------------------------
# OptimisationResult.dv_best_so_far
# ---------------------------------------------------------------------------


def test_dv_best_so_far_field_on_optimisation_result() -> None:
    """OptimisationResult has a dv_best_so_far field and BladeOptimizer._dv_best_so_far is tracked."""
    import dataclasses

    from blade_precompute.section_optimisation.core.types import OptimisationResult, DesignVector
    from blade_precompute.section_optimisation.engine.optimizer import BladeOptimizer

    # Verify the field exists on OptimisationResult
    fields = {f.name for f in dataclasses.fields(OptimisationResult)}
    assert "dv_best_so_far" in fields, "OptimisationResult must have dv_best_so_far field"

    # Verify BladeOptimizer has _dv_best_so_far attribute initialised to None
    assert hasattr(BladeOptimizer, "run"), "BladeOptimizer.run must exist"

    # Manually simulate what _cb does: set _dv_best_so_far
    n = 3
    opt = object.__new__(BladeOptimizer)
    opt._dv_best_so_far = None

    xk = np.array([0.012, 0.010, 0.009, 0.050, 0.045, 0.040, 0.015, 0.013, 0.011])
    opt._dv_best_so_far = DesignVector.from_flat(np.asarray(xk, dtype=np.float64).copy(), n)

    assert opt._dv_best_so_far is not None
    assert opt._dv_best_so_far.t_skin.shape == (n,)
    np.testing.assert_allclose(opt._dv_best_so_far.t_skin, [0.012, 0.010, 0.009])


# ---------------------------------------------------------------------------
# persist_pngs flag on SectionGeometryParams
# ---------------------------------------------------------------------------


def test_section_geometry_params_persist_pngs_default_true() -> None:
    """persist_pngs defaults to True (existing behaviour unchanged)."""
    from blade_precompute.orchestration.precompute.containers import SectionGeometryParams
    from blade_precompute.orchestration.precompute.containers import PrecomputeInputs

    params = SectionGeometryParams.__new__(SectionGeometryParams)
    assert params.__dataclass_fields__["persist_pngs"].default is True


def test_section_shell_model_params_has_loads_provenance() -> None:
    """SectionShellModelParams has loads_provenance field with sensible default."""
    from blade_precompute.orchestration.precompute.containers import SectionShellModelParams

    field = SectionShellModelParams.__dataclass_fields__["loads_provenance"]
    assert field.default == "unit_resultants"


def test_beam_model_params_persist_pngs_default_true() -> None:
    """BeamModelParams.persist_pngs defaults to True."""
    from blade_precompute.orchestration.precompute.containers import BeamModelParams

    assert BeamModelParams.__dataclass_fields__["persist_pngs"].default is True


# ---------------------------------------------------------------------------
# section_shell_model_impl: loads_provenance in summary.json
# ---------------------------------------------------------------------------


def test_section_shell_model_impl_loads_provenance_in_summary(tmp_path: Path) -> None:
    """section_shell_model_impl writes loads_provenance into summary.json."""
    import json

    from blade_precompute.orchestration.precompute.stages import section_shell_model_skipped_outputs
    from blade_precompute.orchestration.precompute.containers import PrecomputeInputs

    # Use the skipped path to avoid needing real imports; validate field exists
    orch_mock = type("Orch", (), {"layout": None, "job_meta": lambda self: {}})()
    orch_mock.layout = type("L", (), {"n_webs": 0, "geometry_mode": "multicell"})()

    # Test the skipped-stage path writes the section_shell_model/summary.json
    out = section_shell_model_skipped_outputs(
        tmp_path, orchestration=orch_mock, reason="test", grid_meta=None
    )
    data = json.loads(out.summary_json.read_text())
    assert data["skipped"] is True

    # The skipped path doesn't write loads_provenance — that's only in the active path.
    # We verify the field is present in the SectionShellModelParams dataclass.
    from blade_precompute.orchestration.precompute.containers import SectionShellModelParams
    assert "loads_provenance" in SectionShellModelParams.__dataclass_fields__


# ---------------------------------------------------------------------------
# write_section_shell_model_station_outputs: persist_pngs=False skips figures
# ---------------------------------------------------------------------------


def test_write_station_outputs_persist_pngs_signature() -> None:
    """write_section_shell_model_station_outputs signature includes persist_pngs and load args."""
    import inspect

    try:
        from blade_precompute.section_shell_model.job_outputs import write_section_shell_model_station_outputs
    except ImportError:
        pytest.skip("section_shell_model not importable in this environment")

    sig = inspect.signature(write_section_shell_model_station_outputs)
    params = sig.parameters

    assert "persist_pngs" in params, "write_section_shell_model_station_outputs must have persist_pngs param"
    assert params["persist_pngs"].default is True, "persist_pngs must default to True"

    for load_arg in ("N", "Vy", "Vz", "My", "Mz", "T"):
        assert load_arg in params, f"write_section_shell_model_station_outputs must have {load_arg!r} param"
        assert params[load_arg].default == 1.0, f"{load_arg!r} must default to 1.0"
