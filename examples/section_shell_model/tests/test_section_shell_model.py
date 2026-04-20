"""Tests for section_shell_model MVP (numpy + pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent.parent
_STRESS_ROOT = _EXAMPLES / "section_stress_model"

# Insert order: last insert(0) wins first on sys.path. Put stress_model first so
# ``lib`` is section_stress_model/lib (not section_shell_model/lib).
sys.path.insert(0, str(_EXAMPLES))
sys.path.insert(0, str(_STRESS_ROOT))


from section_shell_model.lib.local_clpt_shell import (  # noqa: E402
    default_skin_strengths_pa,
    solve_station_clpt_shell,
)
from section_shell_model.lib.recovery_adapter import (  # noqa: E402
    panel_station_shell_resultants,
    run_section_with_shell_mapping,
)


def test_shell_resultants_match_membrane_mapping():
    """Nx, Nxy match laminate_clpt membrane_resultants_from_shell_stress."""
    from lib.laminate_clpt import membrane_resultants_from_shell_stress  # type: ignore[import-untyped]
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], dB_dx=0.0, B=0.0)
    panels, q_tot, sig_p = out[0], out[3], out[4]

    ref = panel_station_shell_resultants(
        panels, q_tot, sig_p, panel_index=0, station_index=None
    )
    t = ref.thickness_m
    N_direct = membrane_resultants_from_shell_stress(
        ref.sigma_xx_pa, 0.0, ref.tau_xy_pa, t
    )

    assert np.allclose(ref.to_N_vec(), N_direct, rtol=0, atol=1e-9)
    assert np.allclose(ref.to_M_vec(), np.zeros(3), atol=0.0)
    assert ref.provenance["Nx"].kind.value == "derived"
    assert ref.provenance["Ny"].kind.value == "placeholder"


def test_tsai_wu_fi_matches_direct_clpt_pipeline():
    """solve_station_clpt_shell reproduces stress-model ply FI for same N, M."""
    from lib.laminate_clpt import clpt_ply_failure_indices  # type: ignore[import-untyped]
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=0.0)
    panels, q_tot, sig_p = out[0], out[3], out[4]
    ref = panel_station_shell_resultants(panels, q_tot, sig_p, panel_index=0)
    plies = panels[0].lam.build_plies()
    st = default_skin_strengths_pa()

    shell_res = solve_station_clpt_shell(
        ref,
        plies,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
    )

    fi_tw, _, _, _ = clpt_ply_failure_indices(
        plies,
        ref.to_N_vec(),
        ref.to_M_vec(),
        st["Xt"],
        st["Xc"],
        st["Yt"],
        st["Yc"],
        st["S12"],
    )

    assert np.allclose(shell_res.fi_tsai_wu, fi_tw, rtol=0, atol=1e-12)


def test_run_section_with_shell_mapping_has_reference():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    bundle = run_section_with_shell_mapping(
        air,
        [0.25, 0.60],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
    )
    assert bundle.reference_resultants is not None
    assert bundle.I_omega >= 0.0
    assert np.isfinite(bundle.y_sc) and np.isfinite(bundle.z_sc)
