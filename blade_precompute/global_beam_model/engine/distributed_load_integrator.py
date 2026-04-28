"""
Cantilever-style integration of distributed line loads and torque about beam x.

Convention
----------
- ``z`` is strictly increasing from **root** (``z[0]``) to **tip** (``z[-1]``).
- **Free tip**: internal resultants at the tip are finite from equilibrium sums;
  shear and torque at the last node are **zero** (no load beyond the tip).

Distributed inputs (section y–z frame, SI)
-------------------------------------------
- ``q_y``, ``q_z`` [N/m] line loads.
- ``m_x`` [N·m/m] distributed torque about the beam longitudinal axis (``T``).

Integration direction
---------------------
For each station ``i``, resultants equal the integral of applied distributions **from**
``z[i]`` **to** the tip ``z[-1]`` (trapezoidal rule on each segment).

Shear::

    Vy(z_i) = ∫_{z_i}^{z_tip} q_y(s) ds,
    Vz(z_i) = ∫_{z_i}^{z_tip} q_z(s) ds.

Bending (decoupled principal coupling used here)::

    My(z_i) = ∫_{z_i}^{z_tip} Vz(s) ds,
    Mz(z_i) = -∫_{z_i}^{z_tip} Vy(s) ds.

Torque::

    T(z_i) = ∫_{z_i}^{z_tip} m_x(s) ds.

Axial force ``N`` is filled from an optional line load ``q_x`` [N/m] (same
cantilever-to-tip trapezoidal rule as shear). If ``q_x`` is omitted, ``N`` is zeros.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _trapz_piecewise_to_tip(z: NDArray[np.float64], f: NDArray[np.float64]) -> NDArray[np.float64]:
    """∫_{z[i]}^{z[-1]} f(s) ds with f linear between tabulated z nodes."""
    z = np.asarray(z, dtype=np.float64).ravel()
    f = np.asarray(f, dtype=np.float64).ravel()
    if z.shape != f.shape:
        raise ValueError("z and f must have the same shape.")
    n = z.shape[0]
    if n < 2:
        raise ValueError("Need at least two z stations for integration.")
    dz = np.diff(z)
    seg = 0.5 * (f[:-1] + f[1:]) * dz
    # sum seg[i:] for each i
    rev_cum = np.cumsum(seg[::-1])
    out = np.zeros(n, dtype=np.float64)
    out[:-1] = rev_cum[::-1]
    out[-1] = 0.0
    return out


def _trapz_piecewise_to_tip_2d(
    z: NDArray[np.float64], f: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Same as _trapz_piecewise_to_tip but for f with shape (n_t, n_z)."""
    z = np.asarray(z, dtype=np.float64).ravel()
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != z.shape[0]:
        raise ValueError("f must have shape (n_t, n_z) matching z length.")
    n_t = f.shape[0]
    out = np.zeros_like(f, dtype=np.float64)
    for it in range(n_t):
        out[it, :] = _trapz_piecewise_to_tip(z, f[it, :])
    return out


@dataclass
class IntegratedResultants:
    """Spanwise internal resultants aligned with ``z`` (beam seven-vector subset)."""

    z: NDArray[np.float64]
    N: NDArray[np.float64]
    Vy: NDArray[np.float64]
    Vz: NDArray[np.float64]
    My: NDArray[np.float64]
    Mz: NDArray[np.float64]
    T: NDArray[np.float64]


class DistributedLoadIntegrator:
    """Integrate ``(q_x optional, q_y, q_z, m_x)`` along ``z`` into ``(N, Vy, Vz, My, Mz, T)``."""

    @staticmethod
    def integrate(
        z: NDArray[np.float64],
        q_y: NDArray[np.float64],
        q_z: NDArray[np.float64],
        m_x: NDArray[np.float64],
        q_x: NDArray[np.float64] | None = None,
    ) -> IntegratedResultants:
        z = np.asarray(z, dtype=np.float64).ravel()
        q_y = np.asarray(q_y, dtype=np.float64).ravel()
        q_z = np.asarray(q_z, dtype=np.float64).ravel()
        m_x = np.asarray(m_x, dtype=np.float64).ravel()
        n_z = int(z.size)
        if n_z == 0 or not (n_z == q_y.size == q_z.size == m_x.size):
            raise ValueError("z, q_y, q_z, m_x must have the same non-zero length.")
        if z.size < 2:
            raise ValueError("Need at least two z stations.")
        if np.any(np.diff(z) <= 0.0):
            raise ValueError("z must be strictly increasing (root to tip).")

        Vy = _trapz_piecewise_to_tip(z, q_y)
        Vz = _trapz_piecewise_to_tip(z, q_z)
        T = _trapz_piecewise_to_tip(z, m_x)
        My = _trapz_piecewise_to_tip(z, Vz)
        Mz = -_trapz_piecewise_to_tip(z, Vy)
        if q_x is None:
            N = np.zeros_like(z, dtype=np.float64)
        else:
            qxa = np.asarray(q_x, dtype=np.float64).ravel()
            if qxa.shape[0] != n_z:
                raise ValueError("q_x must have the same length as z when provided.")
            N = _trapz_piecewise_to_tip(z, qxa)
        return IntegratedResultants(z=z, N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T)

    @staticmethod
    def integrate_timeseries(
        z: NDArray[np.float64],
        q_y: NDArray[np.float64],
        q_z: NDArray[np.float64],
        m_x: NDArray[np.float64],
        q_x: NDArray[np.float64] | None = None,
    ) -> tuple[IntegratedResultants, ...]:
        """
        Integrate each time row of ``(q_y, q_z, m_x)`` with shape ``(n_t, n_z)``.

        Returns a tuple of ``IntegratedResultants`` length ``n_t``.
        """
        z = np.asarray(z, dtype=np.float64).ravel()
        q_y = np.asarray(q_y, dtype=np.float64)
        q_z = np.asarray(q_z, dtype=np.float64)
        m_x = np.asarray(m_x, dtype=np.float64)
        if q_y.ndim != 2 or q_z.shape != q_y.shape or m_x.shape != q_y.shape:
            raise ValueError("q_y, q_z, m_x must have the same 2D shape (n_t, n_z).")
        if q_y.shape[1] != z.size:
            raise ValueError("Last axis must match z length.")
        n_t = q_y.shape[0]
        Vy = _trapz_piecewise_to_tip_2d(z, q_y)
        Vz = _trapz_piecewise_to_tip_2d(z, q_z)
        T = _trapz_piecewise_to_tip_2d(z, m_x)
        My = _trapz_piecewise_to_tip_2d(z, Vz)
        Mz = -_trapz_piecewise_to_tip_2d(z, Vy)
        if q_x is None:
            N = np.zeros((n_t, z.size), dtype=np.float64)
        else:
            qxa = np.asarray(q_x, dtype=np.float64)
            if qxa.shape != (n_t, z.size):
                raise ValueError("q_x must have shape (n_t, n_z) when provided.")
            N = np.zeros((n_t, z.size), dtype=np.float64)
            for it in range(n_t):
                N[it, :] = _trapz_piecewise_to_tip(z, qxa[it, :])
        return tuple(
            IntegratedResultants(
                z=z.copy(),
                N=N[it],
                Vy=Vy[it],
                Vz=Vz[it],
                My=My[it],
                Mz=Mz[it],
                T=T[it],
            )
            for it in range(n_t)
        )
