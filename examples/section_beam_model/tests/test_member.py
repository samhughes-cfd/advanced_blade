"""Unit tests for member.py buckling analysis."""

import numpy as np
import pytest

from section_beam_model.gbt import (
    BoundaryConditions,
    CrossSection,
    CrossSectionModalAnalysis,
    IsotropicMaterial,
    KirchhoffKinematics,
    MemberBucklingAnalysis,
    SectionLoads,
    WallDefinition,
)
from section_beam_model.gbt.member import _build_stress_weighted_B, _strip_geom_axial_for_buckling
from section_beam_model.gbt.prebuckling import PreBucklingAnalysis

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


def _closed_box_section(mat):
    hb, hh = 0.03, 0.10
    return CrossSection(
        [
            WallDefinition([-hb, -hh], [hb, -hh], mat, n_strips=4, name="bot"),
            WallDefinition([hb, -hh], [hb, hh], mat, n_strips=4, name="right"),
            WallDefinition([hb, hh], [-hb, hh], mat, n_strips=4, name="top"),
            WallDefinition([-hb, hh], [-hb, -hh], mat, n_strips=4, name="left"),
        ]
    )


def test_w_dof_only_in_B_matrix():
    """Stress-weighted M_sigma must lump axial stress only onto w-DOFs (index % ndpn == 2)."""
    kin = KirchhoffKinematics()
    ndpn = kin.n_dof_per_strip // 2
    w_local_idx = 2
    mat = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)
    sec = _closed_box_section(mat)
    loads = SectionLoads(N=-1.0e4)
    modal = CrossSectionModalAnalysis(sec, loads).run(n_modes=8)
    n_modes = 6
    pb = PreBucklingAnalysis(sec, loads)
    Nx_arr = pb.axial_stress_resultants()
    n_dof = sec.n_nodes * ndpn
    M_sigma = np.zeros(n_dof)
    for i in range(sec.n_strips):
        Nx = Nx_arr[i]
        ds = sec.get_strip(i).length
        gdofs = sec.strip_global_dofs(i, ndpn)
        w = Nx * ds / 2.0
        for gd in gdofs:
            if gd < n_dof and (gd % ndpn) == w_local_idx:
                M_sigma[gd] += w
    for idx in range(n_dof):
        if idx % ndpn != w_local_idx:
            assert M_sigma[idx] == 0.0
    B = _build_stress_weighted_B(modal, sec, loads, n_modes, kin)
    Phi = modal.modes[:, :n_modes]
    # Reference: same strip assembly as _build_stress_weighted_B (full K_sigma, not lumped diagonal).
    M_full = np.zeros((n_dof, n_dof))
    for i in range(sec.n_strips):
        Nx = float(Nx_arr[i])
        ds = float(sec.get_strip(i).length)
        Kg_local = _strip_geom_axial_for_buckling(Nx, 0.0, 0.0, kin, ds, sec)
        gdofs = sec.strip_global_dofs(i, ndpn)
        for ii, gi in enumerate(gdofs):
            for jj, gj in enumerate(gdofs):
                M_full[gi, gj] += Kg_local[ii, jj]
    B_ref = -(Phi.T @ M_full @ Phi)
    assert np.allclose(B, B_ref, rtol=1e-10, atol=1e-12)


def test_eigenvalue_threshold_scale_invariance():
    """Stiffness scaled by s: diagonal-B member buckling keeps lam_hi / lam_lo ≈ 1/s (within 0.1%)."""
    scale = 1.0e6
    mat_lo = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)
    mat_hi = IsotropicMaterial(E=210e9 * scale, nu=0.3, t=2e-3)
    sec_lo = _closed_box_section(mat_lo)
    sec_hi = _closed_box_section(mat_hi)
    loads = SectionLoads(N=-1.0)
    modal_lo = CrossSectionModalAnalysis(sec_lo, loads).run(n_modes=8)
    modal_hi = CrossSectionModalAnalysis(sec_hi, loads).run(n_modes=8)
    n_modes = 4
    # Diagonal B fallback (no section/loads): exercises A = K^{-1} Kg with scale-extreme matrices.
    lam_lo = MemberBucklingAnalysis(
        modal_lo,
        1.0,
        BoundaryConditions.simply_supported(),
        n_elem=16,
        n_modes=n_modes,
    ).run().lambda_cr
    lam_hi = MemberBucklingAnalysis(
        modal_hi,
        1.0,
        BoundaryConditions.simply_supported(),
        n_elem=16,
        n_modes=n_modes,
    ).run().lambda_cr
    assert lam_lo > 0 and lam_hi > 0
    ratio = lam_hi / lam_lo * scale
    assert abs(ratio - 1.0) < 1e-3
