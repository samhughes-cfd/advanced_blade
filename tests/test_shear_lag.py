"""Reissner shear-lag factor matches tanh formula."""
import math

import numpy as np

from section_model.engine.laminate import LaminateDefinition
from section_model.engine.materials import OrthotropicPly


def _cap_lam():
    p = OrthotropicPly(
        name="c",
        E1=120e9,
        E2=8e9,
        G12=5e9,
        nu12=0.3,
        rho=1600.0,
        t_ply=1e-3,
        Xt=1e9,
        Xc=1e9,
        Yt=1e9,
        Yc=1e9,
        S12=1e9,
        Zt=1e9,
        S13=1e9,
        S23=1e9,
    )
    return LaminateDefinition(plies=[(p, 0.0)] * 4)


def test_phi_sl_formula():
    lam = _cap_lam()
    b_cap = 0.08
    t_skin = 0.002
    ply0, _ = lam.plies[0]
    e1, g12 = ply0.E1, ply0.G12
    t_cap = lam.total_thickness()
    lam2 = g12 / (e1 * t_cap * t_skin)
    lam_bar = math.sqrt(lam2)
    x = 0.5 * lam_bar * b_cap
    phi = 1.0 if x < 1e-12 else math.tanh(x) / x
    lam2_sl = lam.apply_shear_lag(b_cap, t_skin)
    A0 = lam.build_ABD()[0, 0]
    A1 = lam2_sl.build_ABD()[0, 0]
    np.testing.assert_allclose(A1 / A0, phi, rtol=1e-6)
