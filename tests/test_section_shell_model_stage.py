from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.orchestration import PrecomputeOrchestrationContext
from blade_precompute.orchestration.component_materials import ComponentMaterialsMap
from blade_precompute.orchestration.precompute import (
    BeamModelOutputs,
    GridConfig,
    LinspaceSpec,
    PrecomputeInputs,
    SectionShellModelOutputs,
    SectionShellModelParams,
    SectionShellModelStage,
)
from blade_precompute.orchestration.precompute.shell_spars import section_shell_spars_from_layout
from blade_precompute.orchestration.precompute.stages import (
    _station_resultants_for_shell_from_beam,
    section_shell_model_skipped_outputs,
)
from blade_precompute.orchestration.system_layout import resolve_system_type


def _dummy_inputs() -> PrecomputeInputs:
    return PrecomputeInputs(
        spanwise_path=Path("x"),
        extreme_loads_path=Path("y"),
        span_r_z_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        radial_r_m=np.array([0.0, 4.0, 8.0], dtype=np.float64),
        # Root chord 2.0 m hits a singular Bredt matrix for this 2D-CN shell path; keep ≤~1.6 m for smoke stability.
        chord_m=np.array([1.6, 1.6, 1.2], dtype=np.float64),
        twist_deg=np.zeros(3, dtype=np.float64),
        kappa0_x=np.zeros(3, dtype=np.float64),
        kappa0_y=np.zeros(3, dtype=np.float64),
        kappa0_z=np.zeros(3, dtype=np.float64),
        # Cambered 4-digit at every station (2412-style); m=p=0 at root makes a
        # symmetric airfoil that hits a singular Bredt system for 2D-CN shells.
        naca_m=np.array([2.0, 2.0, 2.0], dtype=np.float64),
        naca_p=np.array([4.0, 4.0, 4.0], dtype=np.float64),
        naca_xx=np.array([12.0, 12.0, 12.0], dtype=np.float64),
        naca_series=np.full(3, 4, dtype=np.int64),
        loads_r_z_m=np.array([0.0, 8.0], dtype=np.float64),
        q_y_Npm=np.zeros(2, dtype=np.float64),
        q_z_Npm=np.zeros(2, dtype=np.float64),
        m_x_Nmpm=np.zeros(2, dtype=np.float64),
    )


def _orch() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="2D-CN",
        layout=resolve_system_type("2D-CN"),
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
            section_plot_station_spec="root",
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
            section_plot_station_spec="root",
            orchestration=_orch(),
        )
    )
    with pytest.raises(RuntimeError):
        st.get_results()


def test_section_shell_spars_from_layout_2b_cn() -> None:
    layout = resolve_system_type("2D-CN")
    assert section_shell_spars_from_layout(layout) == [0.15, 0.5]


def test_section_shell_spars_from_layout_0a_empty() -> None:
    layout = resolve_system_type("0A")
    assert section_shell_spars_from_layout(layout) == []


def test_section_shell_spars_from_layout_mismatch_raises() -> None:
    layout = replace(resolve_system_type("2D-CN"), web_chord_fracs=(0.15,))
    with pytest.raises(ValueError, match="web_chord_fracs length"):
        section_shell_spars_from_layout(layout)


def test_section_shell_model_skipped_outputs_writes_summary(tmp_path: Path) -> None:
    orch = _orch()
    out = section_shell_model_skipped_outputs(
        tmp_path,
        orchestration=orch,
        reason="run_section_shell_model=false",
        grid_meta={"type": "section_shell_model"},
    )
    assert out.skipped
    assert not out.png_paths
    data = (tmp_path / "section_shell_model" / "summary.json").read_text(encoding="utf-8")
    assert '"skipped": true' in data
    assert "run_section_shell_model=false" in data
    assert "station_result_json_paths" in data


def test_station_resultants_for_shell_reads_beam_json_order(tmp_path: Path) -> None:
    result_json = tmp_path / "beam_result.json"
    result_json.write_text(
        json.dumps(
            {
                "z_stations_out": [0.0, 4.0, 8.0],
                "resultants": [
                    [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                    [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
                    [12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0],
                ],
            }
        ),
        encoding="utf-8",
    )
    bm_out = BeamModelOutputs(result_json=result_json, png_paths=[])

    station_res = _station_resultants_for_shell_from_beam(bm_out, _dummy_inputs())

    assert station_res[0] == pytest.approx((10.0, 20.0, 30.0, 40.0, 50.0, 60.0))
    assert station_res[1] == pytest.approx((11.0, 21.0, 31.0, 41.0, 51.0, 61.0))
    assert station_res[2] == pytest.approx((12.0, 22.0, 32.0, 42.0, 52.0, 62.0))


def test_grid_config_section_shell_fields() -> None:
    g = LinspaceSpec(0.0, 1.0, 3)
    s = LinspaceSpec(0.0, 1.0, 2)
    cfg = GridConfig(
        geometry=g,
        structural=s,
        section_plot_station_spec="root",
        n_beam_nodes=10,
        run_section_shell_model=False,
        section_shell_n_elements_per_panel=8,
        section_shell_dpi=100,
    )
    assert cfg.run_section_shell_model is False
    assert cfg.section_shell_n_elements_per_panel == 8
    assert cfg.section_shell_dpi == 100


def test_section_shell_model_stage_smoke_unmocked(tmp_path: Path) -> None:
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch(),
            n_elements_per_panel=4,
            dpi=72,
        )
    )
    r = st.execute().get_results()
    assert not r.skipped
    assert r.summary_json.is_file()
    payload = json.loads(r.summary_json.read_text(encoding="utf-8"))
    assert payload.get("skipped") is False
    assert payload.get("spars") == [0.15, 0.5]
    assert r.png_paths
    assert r.station_result_json_paths
    assert len(r.station_result_json_paths) == len(payload["station_result_json_paths"])
    for jp in r.station_result_json_paths:
        assert jp.is_file()
        station_payload = json.loads(jp.read_text(encoding="utf-8"))
        assert station_payload.get("schema") == "section_shell_station_v1"
        assert station_payload.get("station_tag")
        assert "thin_wall" in station_payload
        assert "unit_section_resultants" in station_payload


# ---------------------------------------------------------------------------
# PR3 v2 path tests
# ---------------------------------------------------------------------------


def _orch_0a() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="0A",
        layout=resolve_system_type("0A"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


def _orch_0b() -> PrecomputeOrchestrationContext:
    return PrecomputeOrchestrationContext(
        system_type_key="0B",
        layout=resolve_system_type("0B"),
        component_materials=ComponentMaterialsMap(skin=0, spar_cap=0, shear_web=0),
    )


def test_v2_path_2d_cn_produces_v2_schema(tmp_path: Path) -> None:
    """v2 flag routes 2D-CN through build_shell_mesh_inputs → schema v2."""
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch(),
            n_elements_per_panel=4,
            dpi=72,
            use_mitc4_v2_path=True,
        )
    )
    r = st.execute().get_results()
    assert not r.skipped
    assert r.station_result_json_paths
    for jp in r.station_result_json_paths:
        assert jp.is_file()
        sp = json.loads(jp.read_text(encoding="utf-8"))
        assert sp.get("schema") == "section_shell_station_v2"
        assert sp.get("station_tag")
        assert "thin_wall" in sp
        assert "unit_section_resultants" in sp
        assert "mesh_summary" in sp
        # 2D-CN has 2 webs → skin + web + cap panels
        tw = sp["thin_wall"]
        assert tw["n_panels"] >= 1
        assert tw["n_total_elements"] > 0


def test_v2_path_0a_skin_panels_only(tmp_path: Path) -> None:
    """0A + v2 flag produces a mesh with skin panels only (no web panels)."""
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch_0a(),
            n_elements_per_panel=4,
            dpi=72,
            use_mitc4_v2_path=True,
        )
    )
    r = st.execute().get_results()
    assert not r.skipped
    assert r.station_result_json_paths
    for jp in r.station_result_json_paths:
        sp = json.loads(jp.read_text(encoding="utf-8"))
        assert sp.get("schema") == "section_shell_station_v2"
        assert sp.get("layout_key") == "0A"
        tw = sp["thin_wall"]
        panels = tw["panels"]
        kinds = {p["kind"] for p in panels}
        assert "web" not in kinds, f"0A must not have web panels; got kinds={kinds}"
        assert "skin" in kinds, f"0A must have at least one skin panel; got kinds={kinds}"


def test_v2_path_0b_no_web_panels(tmp_path: Path) -> None:
    """0B + v2 flag produces mesh with no web panels (caps deferred — skin only for now)."""
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch_0b(),
            n_elements_per_panel=4,
            dpi=72,
            use_mitc4_v2_path=True,
        )
    )
    r = st.execute().get_results()
    assert not r.skipped
    assert r.station_result_json_paths
    for jp in r.station_result_json_paths:
        sp = json.loads(jp.read_text(encoding="utf-8"))
        assert sp.get("layout_key") == "0B"
        tw = sp["thin_wall"]
        kinds = {p["kind"] for p in tw["panels"]}
        assert "web" not in kinds


def test_v2_path_legacy_unchanged(tmp_path: Path) -> None:
    """Legacy path (use_mitc4_v2_path=False) still produces v1 schema."""
    st = SectionShellModelStage(
        params=SectionShellModelParams(
            inp=_dummy_inputs(),
            out_dir=tmp_path,
            section_plot_station_spec="root",
            orchestration=_orch(),
            n_elements_per_panel=4,
            dpi=72,
            use_mitc4_v2_path=False,
        )
    )
    r = st.execute().get_results()
    assert r.station_result_json_paths
    for jp in r.station_result_json_paths:
        sp = json.loads(jp.read_text(encoding="utf-8"))
        assert sp.get("schema") == "section_shell_station_v1"
