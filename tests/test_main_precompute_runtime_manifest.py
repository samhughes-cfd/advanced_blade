from __future__ import annotations

from blade_precompute.orchestration.precompute import (
    GridConfig,
    LinspaceSpec,
    runtime_statistics_manifest,
)


def test_runtime_statistics_manifest_has_expected_shape_and_consistency() -> None:
    grid_cfg = GridConfig(
        geometry=LinspaceSpec(z_min=0.0, z_max=9.625, n=176),
        structural=LinspaceSpec(z_min=0.0, z_max=9.625, n=10),
        section_plot_station_spec="root,mid,tip",
        n_beam_nodes=50,
        beam_png_span_samples=400,
        run_section_shell_model=True,
        section_shell_n_elements_per_panel=12,
        section_shell_dpi=150,
        enable_shell_recovery_enrichment=True,
        shell_recovery_n_elements_per_panel=10,
        design_n_workers=4,
        section_solve_n_workers=4,
    )
    stages = {
        "pre_inputs_s": 1.2,
        "section_geometry_s": 2.0,
        "section_shell_model_s": 3.0,
        "section_properties_s": 4.0,
        "global_beam_model_s": 5.0,
        "section_optimisation_s": 6.0,
        "summary_s": 0.4,
    }
    total_wall_s = 22.5

    payload = runtime_statistics_manifest(
        stage_seconds=stages,
        total_wall_s=total_wall_s,
        run_section_shell_model=True,
        section_shell_skipped=False,
        section_geometry_station_count=3,
        section_shell_station_count=3,
        section_properties_station_count=10,
        beam_converged=True,
        beam_n_iterations=27,
        optimizer_ran=True,
        optimizer_n_iter=18,
        python_version="3.11.9",
        platform="Windows-11-10.0.26100-SP0",
        cpu_count=16,
        finished_at_iso="2026-04-25T13:50:00",
        grid_cfg=grid_cfg,
    )

    assert payload["finished_at"] == "2026-04-25T13:50:00"
    assert payload["total_wall_s"] == total_wall_s
    assert payload["stages"] == stages
    assert payload["work_units"]["n_structural_stations"] == 10
    assert payload["work_units"]["section_shell_skipped"] is False
    assert payload["algorithms"]["beam_n_iterations"] == 27
    assert payload["algorithms"]["optimizer_n_iter"] == 18

    fractions = payload["stage_fraction_of_total"]
    assert set(fractions.keys()) == set(stages.keys())
    assert fractions["section_optimisation_s"] == stages["section_optimisation_s"] / total_wall_s
    assert sum(stages.values()) <= total_wall_s
