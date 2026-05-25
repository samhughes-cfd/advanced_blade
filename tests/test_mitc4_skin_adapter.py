"""LaminateDefinition → multi_cell Laminate bridge for MITC4 in-loop stress."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES = _REPO / "examples"
_STRESS_ROOT = _EXAMPLES / "section_stress_model"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_EXAMPLES))
sys.path.insert(0, str(_STRESS_ROOT))

from blade_precompute.section_properties.engine.laminate import LaminateDefinition  # noqa: E402
from blade_precompute.section_properties.engine.materials import OrthotropicPly  # noqa: E402
from blade_precompute.section_optimisation.engine.mitc4_eval import (  # noqa: E402
    LaminateDefinitionMitc4SkinAdapter,
    _coerce_skin_lam_for_mitc4,
    _spar_x_metres,
)


def _ud_ply(t: float = 0.0002) -> OrthotropicPly:
    return OrthotropicPly(
        name="ud",
        E1=140e9,
        E2=10e9,
        G12=5e9,
        nu12=0.28,
        rho=1600.0,
        t_ply=t,
        Xt=1500e6,
        Xc=1200e6,
        Yt=50e6,
        Yc=200e6,
        S12=80e6,
        Zt=40e6,
        S13=30e6,
        S23=30e6,
    )


def test_coerce_laminate_definition_exposes_mitc4_skin_adapter() -> None:
    ld = LaminateDefinition(plies=[(_ud_ply(), 45.0), (_ud_ply(), -45.0)], shear_lag_correction=True)
    skin = _coerce_skin_lam_for_mitc4(ld)
    assert isinstance(skin, LaminateDefinitionMitc4SkinAdapter)
    assert hasattr(skin, "E") and float(skin.E) > 1e6
    assert hasattr(skin, "nu") and 0.0 < float(skin.nu) < 0.5
    assert float(skin.t) == pytest.approx(ld.total_thickness())
    plies = skin.build_plies()
    assert len(plies) == 2
    assert float(plies[0].theta_deg) == pytest.approx(45.0)
    assert float(plies[1].theta_deg) == pytest.approx(-45.0)


def test_adapter_abd_matches_laminate_definition_membrane() -> None:
    """MITC4 ``abd_stack`` on adapted plies should match ``LaminateDefinition.build_ABD`` A-block."""
    from lib.laminate_clpt import abd_stack  # type: ignore[import-untyped]

    ld = LaminateDefinition(plies=[(_ud_ply(0.00025), 0.0), (_ud_ply(0.00025), 90.0)], shear_lag_correction=True)
    skin = LaminateDefinitionMitc4SkinAdapter(ld)
    A_m, _, _ = abd_stack(skin.build_plies())
    A_ld = ld.build_ABD()[:3, :3]
    assert np.allclose(A_m, A_ld, rtol=1e-9, atol=1e-3)


def test_coerce_passes_through_non_laminate_definition() -> None:
    from multi_cell_blade_section import Laminate  # type: ignore[import-untyped]

    raw = Laminate(E=20e9, t=0.006, nu=0.35, n_plies=4)
    assert _coerce_skin_lam_for_mitc4(raw) is raw
    assert _coerce_skin_lam_for_mitc4(None) is None


def test_spar_x_metres_uses_half_chord_web_positions() -> None:
    spars = _spar_x_metres(np.array([-0.35, 0.0], dtype=np.float64), chord_m=1.7)
    assert spars == pytest.approx([0.255, 0.85])


def test_build_section_uses_scaled_airfoil_trailing_edge() -> None:
    from multi_cell_blade_section import build_section, naca_four_digit  # type: ignore[import-untyped]

    chord = 1.6
    airfoil = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    airfoil[:, 0] *= chord

    panels, _booms, webs_geom, n_cells = build_section(airfoil, [0.24, 0.80])

    panel_x = np.concatenate([np.asarray(p.nodes, dtype=np.float64)[:, 0] for p in panels])
    assert n_cells == 3
    assert len(webs_geom) == 2
    assert float(np.max(panel_x)) == pytest.approx(chord)
