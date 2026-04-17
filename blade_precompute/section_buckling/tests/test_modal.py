"""Unit tests for the cross-section modal analysis pipeline."""
import sys; sys.path.insert(0, '/home/user/output/gbt_module')
import numpy as np
import pytest
from gbt import (IsotropicMaterial, WallDefinition, CrossSection,
                 SectionLoads, CrossSectionModalAnalysis, KirchhoffKinematics)

MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)

def make_c_section():
    return CrossSection([
        WallDefinition([0,0],    [0,-0.1],    MAT, n_strips=4, name='web'),
        WallDefinition([0,0],    [0.05, 0],   MAT, n_strips=2, name='top'),
        WallDefinition([0,-0.1], [0.05,-0.1], MAT, n_strips=2, name='bot'),
    ])

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
