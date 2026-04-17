"""Unit tests for member.py buckling analysis."""
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pytest
from gbt import (IsotropicMaterial, WallDefinition, CrossSection,
                 SectionLoads, CrossSectionModalAnalysis,
                 BoundaryConditions, MemberBucklingAnalysis, KirchhoffKinematics)

MAT = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)

def setup():
    sec = CrossSection([
        WallDefinition([0,0],    [0,-0.1],    MAT, n_strips=4, name='web'),
        WallDefinition([0,0],    [0.05,0],    MAT, n_strips=2, name='top'),
        WallDefinition([0,-0.1], [0.05,-0.1], MAT, n_strips=2, name='bot'),
    ])
    modal = CrossSectionModalAnalysis(sec, SectionLoads(N=-1.0)).run(n_modes=4)
    return modal

def test_lambda_cr_positive():
    modal = setup()
    res = MemberBucklingAnalysis(modal, length=1.0, n_elem=10, n_modes=3).run()
    assert res.lambda_cr > 0

def test_simply_supported_lower_than_clamped():
    modal = setup()
    lam_ss = MemberBucklingAnalysis(
        modal, 1.0, BoundaryConditions.simply_supported(), n_elem=10, n_modes=3).run().lambda_cr
    lam_cc = MemberBucklingAnalysis(
        modal, 1.0, BoundaryConditions.clamped_clamped(), n_elem=10, n_modes=3).run().lambda_cr
    assert lam_ss < lam_cc, f"SS={lam_ss:.3e} should be < CC={lam_cc:.3e}"

def test_longer_member_lower_lambda():
    modal = setup()
    lam_short = MemberBucklingAnalysis(
        modal, 0.5, BoundaryConditions.simply_supported(), n_elem=10, n_modes=3).run().lambda_cr
    lam_long  = MemberBucklingAnalysis(
        modal, 2.0, BoundaryConditions.simply_supported(), n_elem=10, n_modes=3).run().lambda_cr
    assert lam_short > lam_long, "Shorter member should buckle at higher load"

def test_convergence_study_returns_dict():
    modal = setup()
    ana = MemberBucklingAnalysis(modal, 1.0, n_elem=8, n_modes=3)
    conv = ana.convergence_study(elem_counts=[4, 8, 16])
    assert 'lambda_cr' in conv
    assert len(conv['lambda_cr']) == 3

def test_signature_curve_shape():
    modal = setup()
    ana = MemberBucklingAnalysis(modal, 1.0, n_elem=8, n_modes=3)
    sig = ana.signature_curve(n_pts=5)
    assert len(sig['half_wave_lengths']) == 5
    assert len(sig['lambda_cr']) == 5
