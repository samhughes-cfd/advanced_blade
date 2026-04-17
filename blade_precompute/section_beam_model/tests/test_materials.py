"""Unit tests for materials.py."""

import numpy as np
import pytest

from blade_precompute.section_beam_model.gbt import IsotropicMaterial, Lamina, LaminateMaterial

def test_isotropic_abd_shape():
    mat = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)
    abd = mat.abd_matrix()
    assert abd.shape == (6, 6)

def test_isotropic_abd_symmetry():
    mat = IsotropicMaterial(E=70e9, nu=0.33, t=3e-3)
    abd = mat.abd_matrix()
    assert np.allclose(abd, abd.T, atol=1e-6)

def test_isotropic_abd_positive_definite():
    mat = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)
    abd = mat.abd_matrix()
    eigvals = np.linalg.eigvalsh(abd)
    assert np.all(eigvals > 0), f"Non-positive eigenvalues: {eigvals}"

def test_isotropic_shear_stiffness():
    mat = IsotropicMaterial(E=210e9, nu=0.3, t=2e-3)
    Ks = mat.shear_stiffness()
    assert Ks.shape == (2, 2)
    assert Ks[0, 0] > 0

def test_laminate_abd_shape():
    plies = [Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=0,  t=0.25e-3),
             Lamina(E1=140e9, E2=10e9, G12=5e9, nu12=0.3, angle=90, t=0.25e-3)]
    mat = LaminateMaterial(plies)
    abd = mat.abd_matrix()
    assert abd.shape == (6, 6)

def test_laminate_symmetric_layup_zero_B():
    """Symmetric [0/90]_s laminate should have zero B matrix."""
    plies = [Lamina(E1=140e9,E2=10e9,G12=5e9,nu12=0.3,angle=0, t=0.25e-3),
             Lamina(E1=140e9,E2=10e9,G12=5e9,nu12=0.3,angle=90,t=0.25e-3),
             Lamina(E1=140e9,E2=10e9,G12=5e9,nu12=0.3,angle=90,t=0.25e-3),
             Lamina(E1=140e9,E2=10e9,G12=5e9,nu12=0.3,angle=0, t=0.25e-3)]
    mat = LaminateMaterial(plies)
    abd = mat.abd_matrix()
    B_block = abd[:3, 3:]
    assert np.allclose(B_block, 0, atol=1e-3), f"B block not zero: {B_block}"
