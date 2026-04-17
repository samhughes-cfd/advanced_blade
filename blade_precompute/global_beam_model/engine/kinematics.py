"""
beam_model/kinematics.py
========================
Unit quaternions ``q = [w, x, y, z]`` (scalar-first) and SO(3) utilities.

Orientation convention
----------------------
``R = quat_to_rotmat(q)`` maps coordinates from **reference** local frame
(beam +X along undeformed tangent) to **spatial** (global) frame.

Newton updates
--------------
Infinitesimal spatial spin vector ``dtheta`` (3,) updates orientation via the
**left** incremental quaternion::

    dq = exp_quat(dtheta / 2)   # finite rotation of magnitude ||dtheta||
    q_new = normalize(dq * q_old)

This is equivalent to ``R_new = exp(dtheta^) @ R_old`` with ``exp`` the
exponential map on SO(3), up to numerical normalization of ``q``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew-symmetric matrix [v]_× for cross-product as matrix multiply."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def rotmat_from_small_curvature(kappa: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotation matrix ``R ≈ I + [κ]×`` with SVD re-orthogonalisation.

    Used for **Tier B** prescribed-resultant workflows (small reference-curvature
    correction to the nodal triad). See :mod:`global_beam_model.core.tier_paths`.
    """
    k = np.asarray(kappa, dtype=np.float64).ravel()
    if k.size != 3:
        return np.eye(3, dtype=np.float64)
    r = np.eye(3, dtype=np.float64) + skew(k)
    u, _, vt = np.linalg.svd(r)
    return (u @ vt).astype(np.float64)


def axial(S: NDArray[np.float64]) -> NDArray[np.float64]:
    """Axial vector of skew-symmetric ``S`` (3,3)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=np.float64)


def quat_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(np.linalg.norm(q))
    if n < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return (q / n).astype(np.float64)


def quat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` (scalar-first)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_conj(q: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quat_to_rotmat(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix ``R`` (3,3) from unit quaternion ``q``."""
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stable Shepperd's method: rotation matrix → unit quaternion."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def exp_so3(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rodrigues: exp([theta]_×) with theta possibly large."""
    t = np.linalg.norm(theta)
    if t < 1e-14:
        return np.eye(3, dtype=np.float64) + skew(theta)
    k = theta / t
    K = skew(k)
    return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def exp_quat(dtheta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Quaternion representing rotation ``exp([dtheta]_×)``.

    For small ``||dtheta||``, ``exp_quat(dtheta/2)`` is the incremental
    quaternion used in Newton updates (left multiply on stored ``q``).
    """
    t = float(np.linalg.norm(dtheta))
    if t < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = dtheta / t
    half = 0.5 * t
    w = np.cos(half)
    s = np.sin(half)
    return quat_normalize(np.array([w, axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64))


def update_orientation(q: NDArray[np.float64], dtheta: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Apply spatial spin ``dtheta`` to quaternion ``q`` (left incremental map).

    ``R_new = exp([dtheta]_×) @ R_old``  <=>  ``q_new = normalize(exp_q(dtheta/2)*q_old)``
    """
    dq = exp_quat(dtheta)
    return quat_normalize(quat_mul(dq, q))


def slerp(q0: NDArray[np.float64], q1: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Spherical linear interpolation, ``t`` in ``[0, 1]``."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        # Near-parallel: lerp + normalize
        q = q0 + t * (q1 - q0)
        return quat_normalize(q)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def slerp_derivative(q0: NDArray[np.float64], q1: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    """Derivative ``d/dt slerp(q0,q1,t)`` as 4-vector."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return quat_normalize(q1 - q0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin((1.0 - t) * theta_0) / sin_theta_0
    s1 = np.sin(t * theta_0) / sin_theta_0
    ds0 = -theta_0 * np.cos((1.0 - t) * theta_0) / sin_theta_0
    ds1 = theta_0 * np.cos(t * theta_0) / sin_theta_0
    return ds0 * q0 + ds1 * q1


def quat_rotate_vector(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate vector ``v`` by quaternion ``q``: ``R(q) @ v``."""
    R = quat_to_rotmat(q)
    return R @ v


def quat_rotate_vector_inv(q: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply ``R(q).T @ v`` (inverse / transpose rotation)."""
    R = quat_to_rotmat(q)
    return R.T @ v


def relative_rotation_vector(q_from: NDArray[np.float64], q_to: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Rotation vector ``phi`` (3,) such that ``exp([phi]_×) ≈ R(q_from)ᵀ R(q_to)``.

    Uses the quaternion logarithm of ``q_rel = conj(q_from) ⊗ q_to``.
    """
    q_rel = quat_normalize(quat_mul(quat_conj(quat_normalize(q_from)), quat_normalize(q_to)))
    if q_rel[0] < 0.0:
        q_rel = -q_rel
    w = float(q_rel[0])
    v = q_rel[1:4].astype(np.float64)
    nv = float(np.linalg.norm(v))
    if nv < 1e-14:
        # small-angle form
        return 2.0 * v / max(w, 1e-14)
    axis = v / nv
    ang = 2.0 * float(np.arctan2(nv, w))
    return axis * ang


def quat_align_axis_to_vector(
    axis_ref: NDArray[np.float64],
    v_spatial: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Unit quaternion with ``R(q) @ axis_ref = v_hat`` (``v_spatial`` normalized).

    ``axis_ref`` is typically ``[1,0,0]`` so local +X aligns with beam tangent.
    """
    a = np.asarray(axis_ref, dtype=np.float64).reshape(3)
    v = np.asarray(v_spatial, dtype=np.float64).reshape(3)
    na = float(np.linalg.norm(a))
    nv = float(np.linalg.norm(v))
    if na < 1e-14 or nv < 1e-14:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a = a / na
    v = v / nv
    c = float(np.clip(np.dot(a, v), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if c < -1.0 + 1e-12:
        # 180°: pick any axis ⟂ a
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ortho, a)) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, ortho)
        axis = axis / np.linalg.norm(axis)
        return quat_normalize(
            np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
        )
    axis_rot = np.cross(a, v)
    s = float(np.linalg.norm(axis_rot))
    axis_rot = axis_rot / s
    ang = float(np.arccos(c))
    half = 0.5 * ang
    w = np.cos(half)
    si = np.sin(half)
    return quat_normalize(np.array([w, axis_rot[0] * si, axis_rot[1] * si, axis_rot[2] * si], dtype=np.float64))
