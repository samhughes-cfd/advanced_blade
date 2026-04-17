"""Unit tests for the cross-section modal analysis pipeline."""

import numpy as np
import pytest

from blade_precompute.section_beam_model.gbt import (
    CrossSection,
    CrossSectionModalAnalysis,
    DEFAULT_BEAM_EXPORT_MODE_LABELS,
    IsotropicMaterial,
    SectionLoads,
    WallDefinition,
    select_modes,
)
from blade_precompute.section_beam_model.gbt.prebuckling import PreBucklingAnalysis
from blade_precompute.section_beam_model.gbt.section_stiffness_export import gbt_to_beam_stiffness

MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)

def make_c_section():
    return CrossSection([
        WallDefinition([0,0],    [0,-0.1],    MAT, n_strips=4, name='web'),
        WallDefinition([0,0],    [0.05, 0],   MAT, n_strips=2, name='top'),
        WallDefinition([0,-0.1], [0.05,-0.1], MAT, n_strips=2, name='bot'),
    ])


def make_box_section(b: float = 0.06, h: float = 0.20) -> CrossSection:
    hb, hh = 0.5 * b, 0.5 * h
    return CrossSection(
        [
            WallDefinition([-hb, -hh], [hb, -hh], MAT, n_strips=8, name="bot"),
            WallDefinition([hb, -hh], [hb, hh], MAT, n_strips=4, name="right"),
            WallDefinition([hb, hh], [-hb, hh], MAT, n_strips=8, name="top"),
            WallDefinition([-hb, hh], [-hb, -hh], MAT, n_strips=4, name="left"),
        ]
    )

def test_modal_runs():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    assert len(result.eigenvalues) == 4

def test_eigenvalues_positive():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=6)
    assert np.all(result.eigenvalues > 0)

def test_modal_rigidity_positive():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    for k in range(len(result.eigenvalues)):
        assert result.modal_rigidity(k) > 0, f"D_{k} not positive"

def test_mode_shapes_shape():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    assert result.modes.shape[1] == 4

def test_classify_mode_returns_string():
    sec = make_c_section()
    result = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    for k in range(len(result.eigenvalues)):
        cls = result.classify_mode(k)
        assert isinstance(cls, str)
        assert cls in ('rigid_body','local','distortional','global','undetermined')


def test_select_modes_by_label():
    sec = make_box_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=24)
    sel = select_modes(full, mode_labels=["bending_x", "bending_y"])
    assert len(sel.eigenvalues) == 2
    labs = {sel.classify_export_mode(k) for k in range(2)}
    assert labs == {"bending_x", "bending_y"}


def test_select_modes_by_count():
    sec = make_c_section()
    full = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=10)
    sel = select_modes(full, n_modes=4)
    assert len(sel.eigenvalues) == 4
    assert np.allclose(np.sort(sel.eigenvalues), np.sort(full.eigenvalues)[:4])


def test_gbt_to_beam_stiffness_isotropic():
    sec = make_box_section()
    loads = SectionLoads(N=-1.0)
    ref = PreBucklingAnalysis(sec, loads).section_properties()
    full = CrossSectionModalAnalysis(sec, loads).run(n_modes=28)
    sel = select_modes(full, mode_labels=list(DEFAULT_BEAM_EXPORT_MODE_LABELS))
    st = gbt_to_beam_stiffness(full, sel, section=sec)
    assert st.EA == pytest.approx(ref["EA"], rel=1e-9, abs=1.0)
    assert st.EI_x > 0.0 and st.EI_y > 0.0 and st.GJ > 0.0
    assert st.GA_x > 0.0 and st.GA_y > 0.0
