"""Tests for section physics helpers (numpy-only)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.laminate_clpt import (
    Ply,
    abd_stack,
    default_rectangular_plies,
    stiffness_Q_from_engineering,
)
from lib.sectorial_warping import (
    omega_increment,
    open_outline_from_airfoil,
    polygon_area_signed,
)
from lib.timoshenko_section import global_shear_stiffness_from_panels
from lib.vlasov_thinwall import axial_resultant_from_warping_stress


def test_Q_isotropic_plane_stress_roundtrip():
    E, nu = 70e9, 0.33
    G = E / (2 * (1 + nu))
    Q = stiffness_Q_from_engineering(E, E, G, nu)
    assert Q[0, 0] > 0 and Q[1, 1] > 0 and Q[2, 2] > 0


def test_abd_symmetric_isotropic():
    plies = default_rectangular_plies(40e9, 0.33, 0.004, n=4)
    A, B, D = abd_stack(plies)
    assert np.allclose(B, 0.0, atol=1e-6)
    assert A[0, 0] > 0 and D[0, 0] > 0


def test_omega_increment_rectangle():
    # Unit square (0,0)-(1,0)-(1,1)-(0,1); pole at origin, first edge
    d = omega_increment(0, 0, 1, 0, 0, 0)
    assert abs(d) < 1e-12


def test_polygon_area_unit_square():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    assert abs(polygon_area_signed(sq) - 1.0) < 1e-9


def test_timoshenko_ga_positive():
    plies = [Ply(E1=40e9, E2=40e9, G12=15e9, nu12=0.33, theta_deg=0, t=0.001)]
    GA_y, GA_z = global_shear_stiffness_from_panels(
        [(1.0, 1.0)], [plies], panel_lengths=[1.0]
    )
    assert GA_y > 0 and GA_z > 0


def test_vlasov_axial_resultant_routine_runs():
    loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]])
    sig = np.array([1.0, -1.0, 1.0, -1.0])
    ntot = axial_resultant_from_warping_stress(loop, sig, t=0.01)
    assert isinstance(ntot, float)


def test_open_outline_nonempty():
    from multi_cell_blade_section import naca_four_digit

    af = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=40)
    op = open_outline_from_airfoil(af)
    assert op.shape[0] >= 20 and op.shape[1] == 2


def test_blade_frames_rotation_identity():
    from lib.blade_frames import rotation_B_to_S, shear_B_to_S

    R = rotation_B_to_S(0.0)
    assert np.allclose(R, np.eye(2))
    vy, vz = shear_B_to_S(3.0, 4.0, 0.0)
    assert abs(vy - 3.0) < 1e-9 and abs(vz - 4.0) < 1e-9


def test_blade_frames_inverse_pair():
    from lib.blade_frames import rotation_B_to_S

    th = 0.3
    v_b = np.array([1.1, -2.2])
    R = rotation_B_to_S(th)
    v_s = R @ v_b
    v_b2 = rotation_B_to_S(-th) @ v_s
    assert np.allclose(v_b2, v_b)


def test_skin_laminate_equal_gauge_geometry():
    """n · t_ply equals total thickness; stack has n equal-thickness plies."""
    from multi_cell_blade_section import Laminate, skin_laminate

    t_ply = 1.5e-3
    n = 4
    lam = skin_laminate(20e9, 0.35, t_ply, n)
    assert abs(lam.t - n * t_ply) < 1e-15
    plies = lam.build_plies()
    assert len(plies) == n
    for ply in plies:
        assert abs(ply.t - t_ply) < 1e-15

    base = Laminate(E=20e9, t=0.006, n_plies=4)
    assert abs(base.t - base.n_plies * (base.t / base.n_plies)) < 1e-15


def test_beam_vlasov_fd_uniform():
    from lib.beam_vlasov_1d import solve_nonuniform_torsion_fd

    L = 1.0
    n = 15
    x = np.linspace(0, L, n)
    EI = 2e5
    GJ = 1e5
    m = np.zeros(n)
    m[n // 2] = 500.0
    phi, p2, B = solve_nonuniform_torsion_fd(x, EI, GJ, m)
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(B))


def test_warping_secondary_q_zero_when_omega_zero():
    from lib.warping_shear import q_omega_secondary_open_vertices

    verts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    omega_hat = np.zeros(3)
    q = q_omega_secondary_open_vertices(
        verts, omega_hat, t=0.01, I_omega=1.0, dB_dx=1e6
    )
    assert np.allclose(q, 0.0, atol=1e-12)


def test_warping_secondary_q_linear_accumulation():
    """Constant ω̂ on a straight strip → uniform dq/ds → linear q(s)."""
    from lib.warping_shear import q_omega_secondary_open_vertices

    verts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    omega_hat = np.ones(3)
    t = 1.0
    I_omega = 1.0
    dB_dx = 1.0
    q = q_omega_secondary_open_vertices(verts, omega_hat, t, I_omega, dB_dx)
    # Segment length 1 each; om_m = 1; dq = -1 * t * 1 * 1 = -1 per segment
    assert abs(q[0]) < 1e-12
    assert abs(q[1] - (-1.0)) < 1e-9
    assert abs(q[2] - (-2.0)) < 1e-9


def test_warping_secondary_dB_zero_gives_zero_panel_q():
    from lib.warping_shear import q_omega_secondary_panels_particular

    loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1]])
    om = np.array([0.0, 0.0, 0.0])
    qv = np.array([0.0, 0.0, 0.0])

    class _P:
        def __init__(self, nodes, t, label="skin"):
            self.nodes = np.asarray(nodes, dtype=float)
            self.label = label

            class _L:
                pass

            self.lam = _L()
            self.lam.t = t

    panels = [_P(loop, 0.01)]
    out = q_omega_secondary_panels_particular(
        loop, om, qv, panels, dB_dx=0.0, I_omega=1.0
    )
    assert len(out) == 1
    assert np.allclose(out[0], 0.0)


def test_run_section_includes_primary_and_warp_shear():
    from multi_cell_blade_section import naca_four_digit, run_section

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], dB_dx=0.0, B=0.0)
    assert len(out) == 18
    q_primary, q_warp = out[16], out[17]
    assert len(q_primary) == len(q_warp)
    for a, b in zip(q_primary, q_warp):
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(b))
    # No warping shear when dB/dx = 0
    for b in q_warp:
        assert np.allclose(b, 0.0, atol=1e-15)


def test_run_section_nonzero_dB_dx_finite_warp_q():
    from multi_cell_blade_section import naca_four_digit, run_section

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], dB_dx=5e3, B=0.0)
    q_warp = out[17]
    qw = np.concatenate(q_warp)
    assert np.all(np.isfinite(qw))
    assert np.max(np.abs(qw)) > 1e-9
