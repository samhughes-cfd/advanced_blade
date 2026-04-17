"""
Classical laminate theory (CLT) for midsurface strips.

ABD assembly (midplane :math:`z=0`, :math:`z_{k+1}-z_k` = ply thickness):

.. math::

    A_{ij} = \\sum_k \\bar{Q}^{(k)}_{ij} (z_{k+1}-z_k)

    B_{ij} = \\frac{1}{2} \\sum_k \\bar{Q}^{(k)}_{ij} (z_{k+1}^2-z_k^2)

    D_{ij} = \\frac{1}{3} \\sum_k \\bar{Q}^{(k)}_{ij} (z_{k+1}^3-z_k^3)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .materials import OrthotropicPly, plane_stress_Q


def _R_stiffness_transform(theta_rad: float) -> NDArray[np.float64]:
    """Transform Q from material axes to laminate axes (Barbero / Jones convention)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [
            [c * c, s * s, 2.0 * s * c],
            [s * s, c * c, -2.0 * s * c],
            [-s * c, s * c, c * c - s * s],
        ],
        dtype=np.float64,
    )


def _Q_bar_single(Q: NDArray[np.float64], theta_deg: float) -> NDArray[np.float64]:
    R = _R_stiffness_transform(np.deg2rad(theta_deg))
    return R @ Q @ R.T


def _T_strain_lam_to_mat(theta_deg: float) -> NDArray[np.float64]:
    """
    Engineering strain vector transform: eps_mat = T @ eps_lam.

    Laminate axes (1,2) align with strip tangent/normal; ply angle θ is
    rotation from laminate-1 to material-1 (same as Q̄ rotation).
    """
    return _R_stiffness_transform(np.deg2rad(theta_deg))


@dataclass
class LaminateDefinition:
    plies: List[Tuple[OrthotropicPly, float]]  # (ply, angle_deg)
    shear_lag_correction: bool = True
    _abd_override: NDArray[np.float64] | None = field(default=None, repr=False, compare=False)

    def build_ABD(self) -> NDArray[np.float64]:
        """Full (6, 6) ABD matrix in SI units [N/m, N, N·m, N·m²]."""
        if self._abd_override is not None:
            return self._abd_override.copy()
        abd = np.zeros((6, 6), dtype=np.float64)
        z_inner = -0.5 * self.total_thickness()
        for ply, ang in self.plies:
            t = ply.t_ply
            z_outer = z_inner + t
            Qb = _Q_bar_single(plane_stress_Q(ply), ang)
            dz = z_outer - z_inner
            abd[0:3, 0:3] += Qb * dz
            abd[0:3, 3:6] += 0.5 * Qb * (z_outer**2 - z_inner**2)
            abd[3:6, 0:3] += 0.5 * Qb * (z_outer**2 - z_inner**2)
            abd[3:6, 3:6] += (1.0 / 3.0) * Qb * (z_outer**3 - z_inner**3)
            z_inner = z_outer
        return abd

    def build_Q_bar(self) -> NDArray[np.float64]:
        """Shape (n_ply, 3, 3) transformed stiffness per ply."""
        n = len(self.plies)
        out = np.zeros((n, 3, 3), dtype=np.float64)
        for k, (ply, ang) in enumerate(self.plies):
            out[k] = _Q_bar_single(plane_stress_Q(ply), ang)
        return out

    def build_T_ply(self) -> NDArray[np.float64]:
        """Shape (n_ply, 3, 3): strain laminate → material for each ply."""
        n = len(self.plies)
        out = np.zeros((n, 3, 3), dtype=np.float64)
        for k, (_, ang) in enumerate(self.plies):
            out[k] = _T_strain_lam_to_mat(ang)
        return out

    def ply_depths(self) -> NDArray[np.float64]:
        """Midplane z-coordinate of each ply centre, shape (n_ply,)."""
        z = -0.5 * self.total_thickness()
        depths = []
        for ply, _ in self.plies:
            t = ply.t_ply
            depths.append(z + 0.5 * t)
            z += t
        return np.array(depths, dtype=np.float64)

    def ply_interfaces_z(self) -> NDArray[np.float64]:
        """z from bottom (-h/2) to top (+h/2), shape (n_ply+1,)."""
        z = -0.5 * self.total_thickness()
        zs = [z]
        for ply, _ in self.plies:
            z += ply.t_ply
            zs.append(z)
        return np.array(zs, dtype=np.float64)

    def total_thickness(self) -> float:
        return float(sum(p.t_ply for p, _ in self.plies))

    def equivalent_density(self) -> float:
        """Thickness-weighted average density [kg/m³]."""
        h = self.total_thickness()
        if h <= 0:
            return 0.0
        m = 0.0
        for ply, _ in self.plies:
            m += ply.rho * ply.t_ply
        return m / h

    def scale_thickness(self, t_new: float) -> LaminateDefinition:
        """Scale all ply thicknesses uniformly so total thickness = t_new."""
        h0 = self.total_thickness()
        if h0 <= 0:
            raise ValueError("Cannot scale zero-thickness laminate.")
        s = t_new / h0
        new_plies: List[Tuple[OrthotropicPly, float]] = []
        for ply, ang in self.plies:
            pnew = OrthotropicPly(
                name=ply.name,
                E1=ply.E1,
                E2=ply.E2,
                G12=ply.G12,
                nu12=ply.nu12,
                rho=ply.rho,
                t_ply=ply.t_ply * s,
                Xt=ply.Xt,
                Xc=ply.Xc,
                Yt=ply.Yt,
                Yc=ply.Yc,
                S12=ply.S12,
                Zt=ply.Zt,
                S13=ply.S13,
                S23=ply.S23,
            )
            new_plies.append((pnew, ang))
        return LaminateDefinition(
            plies=new_plies,
            shear_lag_correction=self.shear_lag_correction,
            _abd_override=None,
        )

    def apply_shear_lag(self, b_cap: float, t_skin: float) -> LaminateDefinition:
        """
        Return a new laminate whose **A11** entry is multiplied by Reissner
        shear-lag factor :math:`\\phi_{SL}` (effective axial stiffness reduction).

        .. math::

            \\phi_{SL} = \\frac{\\tanh(\\lambda b/2)}{\\lambda b/2},\\quad
            \\lambda^2 = \\frac{G_{12}}{E_1\\,t_{cap}\\,t_{skin}}}

        Uses **first ply** :math:`E_1,G_{12}` and total laminate thickness as
        :math:`t_{cap}`. Preliminary-design correction only.
        """
        if not self.plies:
            return self
        ply0, _ = self.plies[0]
        e1, g12 = ply0.E1, ply0.G12
        t_cap = self.total_thickness()
        if b_cap <= 0 or t_skin <= 0 or t_cap <= 0 or e1 <= 0:
            raise ValueError("b_cap, t_skin, t_cap, E1 must be positive for shear lag.")
        lam2 = g12 / (e1 * t_cap * t_skin)
        lam = np.sqrt(max(lam2, 0.0))
        x = 0.5 * lam * b_cap
        if x < 1e-12:
            phi = 1.0
        else:
            phi = float(np.tanh(x) / x)
        abd = self.build_ABD().copy()
        abd[0, 0] *= phi
        if not self.shear_lag_correction and b_cap > 0.15:
            warnings.warn(
                "Wide cap without shear_lag_correction on laminate: axial load "
                "sharing may be unconservative (Reissner shear lag not applied).",
                UserWarning,
                stacklevel=2,
            )
        return LaminateDefinition(
            plies=list(self.plies),
            shear_lag_correction=self.shear_lag_correction,
            _abd_override=abd,
        )


def tsai_wu_polynomial(sigma: NDArray[np.float64], Xt: float, Xc: float, Yt: float, Yc: float, S12: float) -> NDArray[np.float64]:
    """
    Tsai–Wu failure polynomial :math:`F_i\\sigma_i + F_{ij}\\sigma_i\\sigma_j`
    evaluated at ply stresses (failure surface = 1).

    Parameters
    ----------
    sigma
        (..., 3) = [σ11, σ22, τ12] material frame [Pa].
    """
    s11, s22, s12 = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S12 * S12)
    F12 = -0.5 * np.sqrt(max(F11 * F22, 0.0))
    return (
        F1 * s11
        + F2 * s22
        + F11 * s11**2
        + F22 * s22**2
        + F66 * s12**2
        + 2.0 * F12 * s11 * s22
    )


def tsai_wu_fi(sigma: NDArray[np.float64], Xt: float, Xc: float, Yt: float, Yc: float, S12: float) -> NDArray[np.float64]:
    """
    Failure index :math:`FI = g` where :math:`g=1` is on the Tsai–Wu surface
    (same polynomial as :func:`tsai_wu_polynomial`).
    """
    return tsai_wu_polynomial(sigma, Xt, Xc, Yt, Yc, S12)
