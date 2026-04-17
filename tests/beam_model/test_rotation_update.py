"""SO(3) exponential map and quaternion update consistency."""

from __future__ import annotations

import numpy as np

from beam_model.engine.kinematics import exp_quat, exp_so3, quat_normalize, quat_to_rotmat, update_orientation


def test_exp_so3_orthogonality() -> None:
    th = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    R = exp_so3(th)
    I = R @ R.T
    assert np.allclose(I, np.eye(3), atol=1e-10)


def test_exp_matches_quaternion_left_update() -> None:
    rng = np.random.default_rng(0)
    q = quat_normalize(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    dth = 0.03 * rng.standard_normal(3).astype(np.float64)
    R0 = quat_to_rotmat(q)
    R1_exp = exp_so3(dth) @ R0
    q1 = update_orientation(q, dth)
    R1_q = quat_to_rotmat(q1)
    assert np.allclose(R1_exp, R1_q, atol=1e-6)


def test_exp_quat_half_angle() -> None:
    dth = np.array([0.0, 0.0, 0.2], dtype=np.float64)
    q_inc = exp_quat(dth)
    Rq = quat_to_rotmat(q_inc)
    Re = exp_so3(dth)
    assert np.allclose(Rq, Re, atol=1e-6)
