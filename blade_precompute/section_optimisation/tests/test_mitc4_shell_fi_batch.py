"""mitc4_shell_fi_batch: MITC4 N/M → CLPT ply Hashin path (mocked shell solve)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from blade_precompute.section_optimisation.core.types import OptimBladeGeometry
from blade_precompute.section_optimisation.engine.mitc4_eval import mitc4_shell_fi_batch
from blade_precompute.section_properties.engine.geometry import SectionDefinition, SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import OrthotropicPly
from blade_precompute.section_shell_model.lib.types import ShellPanelResultants


def _ply0() -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=30e9,
        E2=8e9,
        G12=3e9,
        nu12=0.3,
        rho=1.5e3,
        t_ply=0.0002,
        Xt=500e6,
        Xc=300e6,
        Yt=50e6,
        Yc=200e6,
        S12=40e6,
        Zt=1e6,
        S13=1e6,
        S23=1e6,
    )


def _one_station_section() -> SectionDefinition:
    lam = LaminateDefinition(plies=[(_ply0(), 0.0)], shear_lag_correction=True)
    line = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    sub = SubcomponentGeometry(name="skin", midsurface_coords=line, material=lam, thickness=0.002)
    return SectionDefinition(station_z=0.0, subcomponents=[sub])


def _one_station_bg() -> OptimBladeGeometry:
    z = np.array([0.0], dtype=np.float64)
    r = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    air = np.array([[0.0, 0.0], [0.1, 0.02], [0.2, 0.0]], dtype=np.float64)
    return OptimBladeGeometry(
        z_stations=z,
        r_ref=r,
        kappa0=np.zeros((1, 3), dtype=np.float64),
        chord=np.array([1.0], dtype=np.float64),
        twist=np.zeros(1, dtype=np.float64),
        airfoil_profiles=[air],
        web_positions=np.array([-0.3, 0.3], dtype=np.float64),
        subcomponent_materials={},
    )


@patch("blade_precompute.section_optimisation.engine.mitc4_eval.run_section_with_mitc4_shell")
def test_mitc4_shell_fi_batch_maps_nm_to_hashin_envelope(mock_run) -> None:
    """``ShellPanelResultants`` with known N, M is fed to ``clpt_ply_failure_indices``; mask and tensor shapes are consistent."""

    def _make_bundle() -> object:
        pan = SimpleNamespace()
        pan.name = "USkin C1"
        r0 = ShellPanelResultants(
            Nx=5e3,
            Ny=0.0,
            Nxy=0.0,
            Mx=10.0,
            My=0.0,
            Mxy=0.0,
            sigma_xx_pa=2e5,
        )
        b = SimpleNamespace()
        b.sig_p = ((2e5,),)
        b.all_panel_mitc4_results = [[r0]]
        b.panels = [pan]
        b.mitc4_panel_abd = None
        b.mitc4_panel_thickness_m = None
        b.mitc4_panel_G_eff = None
        b.mitc4_panel_labels = None
        return b

    mock_run.side_effect = lambda *a, **k: _make_bundle()

    sdef = _one_station_section()
    bg = _one_station_bg()
    r_sec = np.zeros((1, 7), dtype=np.float64)
    n_p = len(sdef.subcomponents[0].material.plies)
    n_ply_max = max(2, n_p)

    fi_s, fi_h, msk, fi_vm, msk_vm = mitc4_shell_fi_batch(
        [sdef],
        r_sec,
        bg,
        n_ply_max,
        ["skin"],
        n_elements_per_panel=2,
        clt_station0_sink=None,
    )

    assert mock_run.call_count == 1
    assert fi_s.shape == (1,)
    assert np.isfinite(fi_s[0])
    assert fi_h.shape == (1, 1, n_ply_max)
    assert msk.shape == (1, 1)
    assert fi_vm.shape == (1, 0)
    assert msk_vm.shape == (1, 0)
    assert bool(msk[0, 0])
    assert bool(np.isfinite(fi_h[0, 0, 0]))
    assert float(np.max(fi_h[0, 0, :n_p])) >= 0.0
