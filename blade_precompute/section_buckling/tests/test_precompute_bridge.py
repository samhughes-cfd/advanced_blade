"""Bridge from SectionDefinition (section_properties) to GBT buckling."""

from pathlib import Path

import numpy as np

from blade_precompute.section_buckling.gbt import CrossSectionModalAnalysis, SectionLoads
from blade_precompute.section_buckling.interface.plots import (
    plot_buckling_member_overview_grid,
    plot_cross_section_mode_wireframes,
)
from blade_precompute.section_buckling.interface.precompute import (
    analyze_station_buckling,
    section_definition_to_gbt_cross_section,
    wall_definitions_from_line_mesh,
)
from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.materials import IsotropicMaterial as SPIso


def _minimal_section_definition() -> SectionDefinition:
    m = SPIso(name="al", E=70e9, nu=0.33, rho=2700.0, sigma_allow=200e6)
    pts = np.array([[0.0, 0.0], [0.0, -0.05], [0.02, -0.05]], dtype=np.float64)
    sub = SubcomponentGeometry(name="path", midsurface_coords=pts, material=m, thickness=2e-3)
    return SectionDefinition(station_z=2.5, subcomponents=[sub])


def test_section_definition_to_gbt_cross_section_builds_strips():
    sd = _minimal_section_definition()
    cs = section_definition_to_gbt_cross_section(sd)
    assert cs.n_strips >= 2
    assert cs.n_nodes >= 2


def test_analyze_station_buckling_positive_lambda_cr():
    sd = _minimal_section_definition()
    out = analyze_station_buckling(
        sd,
        section_loads=SectionLoads(N=-5e3, My=0.0, Mz=0.0),
        member_length_m=0.5,
        n_cross_section_modes=6,
        n_member_modes=4,
        n_elem=12,
        signature_n_pts=5,
        convergence_elem_counts=[4, 8],
    )
    assert out.get("error") is None
    mb = out.get("member_buckling") or {}
    assert float(mb["lambda_cr"]) > 0.0
    assert out.get("wall_source") == "polyline_fallback"


def _two_part_section() -> SectionDefinition:
    """Skin + web meeting at one node (merged line mesh)."""
    m = SPIso(name="al", E=70e9, nu=0.33, rho=2700.0, sigma_allow=200e6)
    skin = SubcomponentGeometry(
        name="skin_outer",
        midsurface_coords=np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64),
        material=m,
        thickness=1.5e-3,
    )
    web = SubcomponentGeometry(
        name="shear_web_ps",
        midsurface_coords=np.array([[0.1, 0.0], [0.1, 0.08]], dtype=np.float64),
        material=m,
        thickness=2e-3,
    )
    return SectionDefinition(station_z=0.0, subcomponents=[skin, web])


def test_line_mesh_wall_count_matches_expectation():
    sd = _two_part_section()
    pairs = wall_definitions_from_line_mesh(sd)
    assert len(pairs) == 2
    cs_lm = section_definition_to_gbt_cross_section(sd, use_line_mesh=True)
    cs_pl = section_definition_to_gbt_cross_section(sd, use_line_mesh=False)
    assert cs_lm.n_strips >= 2
    assert cs_pl.n_strips >= 2


def test_per_subcomponent_buckling_entries():
    sd = _two_part_section()
    out = analyze_station_buckling(
        sd,
        section_loads=SectionLoads(N=-8e3, My=0.0, Mz=0.0),
        member_length_m=0.4,
        n_cross_section_modes=6,
        n_member_modes=4,
        n_elem=10,
        signature_n_pts=6,
        convergence_elem_counts=[4, 8],
        include_per_subcomponent=True,
    )
    assert out.get("wall_source") == "line_mesh"
    parts = out.get("per_subcomponent") or []
    assert len(parts) == 2
    for p in parts:
        an = p.get("analysis") or {}
        assert an.get("error") is None
        assert float(an["member_buckling"]["lambda_cr"]) > 0.0


def test_plot_cross_section_mode_wireframes_writes_png(tmp_path: Path) -> None:
    sd = _two_part_section()
    sec = section_definition_to_gbt_cross_section(sd)
    modal = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0e4)).run(n_modes=4)
    outp = tmp_path / "section_modes.png"
    plot_cross_section_mode_wireframes(sec, modal, outp, station_z_m=0.5, n_modes_plot=2)
    assert outp.is_file()
    assert outp.stat().st_size > 200


def test_analyze_station_buckling_wireframe_paths(tmp_path: Path) -> None:
    sd = _two_part_section()
    coupled = tmp_path / "coupled"
    coupled.mkdir(parents=True, exist_ok=True)
    parts = tmp_path / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    p_modes = coupled / "section_modes_wireframe.png"
    p_mem = coupled / "member_coupled_approx.png"
    out = analyze_station_buckling(
        sd,
        section_loads=SectionLoads(N=-8e3, My=0.0, Mz=0.0),
        member_length_m=0.4,
        n_cross_section_modes=6,
        n_member_modes=4,
        n_elem=10,
        signature_n_pts=6,
        convergence_elem_counts=[4, 8],
        include_per_subcomponent=True,
        section_modes_wireframe_png=p_modes,
        member_coupled_section_wireframe_png=p_mem,
        part_modes_wireframe_out_dir=parts,
        part_modes_wireframe_tag="t0",
    )
    paths = out.get("_wireframe_png_paths") or []
    assert len(paths) >= 2
    assert p_modes.is_file() and p_modes.stat().st_size > 200
    assert p_mem.is_file() and p_mem.stat().st_size > 200
    assert (parts / "skin_outer" / "section_modes_wireframe.png").is_file()
    assert (parts / "shear_web_ps" / "section_modes_wireframe.png").is_file()


def test_member_overview_grid_writes_png(tmp_path: Path) -> None:
    sd = _two_part_section()
    out = analyze_station_buckling(
        sd,
        section_loads=SectionLoads(N=-8e3, My=0.0, Mz=0.0),
        member_length_m=0.4,
        n_cross_section_modes=6,
        n_member_modes=4,
        n_elem=10,
        signature_n_pts=6,
        convergence_elem_counts=[4, 8],
        include_per_subcomponent=False,
    )
    outp = tmp_path / "overview.png"
    plot_buckling_member_overview_grid(out, outp, suptitle="test overview")
    assert outp.is_file() and outp.stat().st_size > 300
