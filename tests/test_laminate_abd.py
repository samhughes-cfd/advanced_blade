"""ABD assembly vs hand calculation for a single 0° ply."""
import numpy as np

from section_model.engine.laminate import LaminateDefinition
from section_model.engine.materials import OrthotropicPly, plane_stress_Q


def test_abd_single_ply():
    ply = OrthotropicPly(
        name="t",
        E1=137.9e9,
        E2=9.79e9,
        G12=5.0e9,
        nu12=0.28,
        rho=1600.0,
        t_ply=0.0002,
        Xt=1e9,
        Xc=1e9,
        Yt=1e9,
        Yc=1e9,
        S12=1e9,
        Zt=1e9,
        S13=1e9,
        S23=1e9,
    )
    lam = LaminateDefinition(plies=[(ply, 0.0)])
    ABD = lam.build_ABD()
    Q = plane_stress_Q(ply)
    h = ply.t_ply
    A_hand = Q * h
    np.testing.assert_allclose(ABD[0:3, 0:3], A_hand, rtol=1e-5)
    np.testing.assert_allclose(ABD[3:6, 3:6], (1.0 / 12.0) * Q * h**3, rtol=1e-5)
