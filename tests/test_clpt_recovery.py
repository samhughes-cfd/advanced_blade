"""CLPT ply stresses vs closed-form for symmetric cross-ply (membrane only)."""
import numpy as np

from section_model.engine.clpt_recovery import clpt_ply_stresses_section_frame
from section_model.engine.laminate import LaminateDefinition
from section_model.engine.materials import OrthotropicPly


def _sym_laminate():
    p = OrthotropicPly(
        name="p",
        E1=40e9,
        E2=10e9,
        G12=4e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=0.000125,
        Xt=1e9,
        Xc=1e9,
        Yt=1e9,
        Yc=1e9,
        S12=1e9,
        Zt=1e9,
        S13=1e9,
        S23=1e9,
    )
    return LaminateDefinition(plies=[(p, 0.0), (p, 90.0), (p, 90.0), (p, 0.0)])


def test_clpt_membrane_only():
    lam = _sym_laminate()
    abd = lam.build_ABD()
    abd_inv = np.linalg.inv(abd)
    N = np.array([1e5, 0.0, 0.0, 0.0, 0.0, 0.0])
    eps0 = abd_inv @ N
    sig = lam.build_Q_bar()[0] @ eps0[:3]
    R = N.reshape(1, 1, 6)
    qb = lam.build_Q_bar()[None, ...]
    zp = lam.ply_depths()[None, ...]
    sig_rec = clpt_ply_stresses_section_frame(R, abd_inv[None, ...], qb, zp)
    np.testing.assert_allclose(sig_rec[0, 0, 0], sig, rtol=1e-5)
