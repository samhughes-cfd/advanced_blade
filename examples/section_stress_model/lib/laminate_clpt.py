"""
Classical Laminate Plate Theory (CLPT) helpers for orthotropic plies.

Convention (laminate axes 1,2,3):
  1 = ply fibre direction (often aligned with panel tangent in UD plies)
  2 = in-plane transverse
  3 = through-thickness (out-of-plane)

Beam/section usage:
  Laminate mid-plane strains ε⁰ and curvatures κ relate to force/moment resultants
  [N] = [A B][ε⁰], [M] = [B D][κ]  (standard ABD).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def compliance_orthotropic(E1: float, E2: float, G12: float, nu12: float) -> np.ndarray:
    """3×3 plane-stress compliance [S] in material axes (sigma1, sigma2, tau12)."""
    nu21 = nu12 * E2 / E1
    den = 1.0 - nu12 * nu21
    if abs(den) < 1e-30:
        den = 1e-30
    S11 = 1.0 / E1
    S22 = 1.0 / E2
    S12 = -nu12 / E1
    S66 = 1.0 / G12
    return np.array([[S11, S12, 0.0], [S12, S22, 0.0], [0.0, 0.0, S66]], dtype=float)


def stiffness_Q_from_engineering(E1: float, E2: float, G12: float, nu12: float) -> np.ndarray:
    """Plane-stress stiffness [Q] (3×3) in material axes."""
    S = compliance_orthotropic(E1, E2, G12, nu12)
    return np.linalg.inv(S)


def Q_bar(theta_deg: float, Q_mat: np.ndarray) -> np.ndarray:
    """Transform Q from material to laminate axes (rotation about z, angle theta_deg)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    T = np.array(
        [[c * c, s * s, 2 * s * c], [s * s, c * c, -2 * s * c], [-s * c, s * c, c * c - s * s]],
        dtype=float,
    )
    return T.T @ Q_mat @ T


def Q_bar_transverse(theta_deg: float, G23: float, G13: float) -> tuple[float, float]:
    """
    Effective transverse shear stiffnesses Q44, Q55 in laminate axes (theta from material 1-axis).

    Material axes: Q44_mat = G23, Q55_mat = G13 for transverse shear in 2-3 and 1-3 planes.
    Rotated about 3-axis: Q44', Q55' from rotation of the 2×2 transverse shear block.
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    G23, G13 = float(G23), float(G13)
    # Rotation of [Q44 Q45; Q45 Q55] with Q45=0 in material axes
    Q44p = G23 * c * c + G13 * s * s
    Q55p = G23 * s * s + G13 * c * c
    return Q44p, Q55p


@dataclass
class Ply:
    """Single orthotropic ply (material axes 1 = fibre direction)."""
    E1: float
    E2: float
    G12: float
    nu12: float
    theta_deg: float
    t: float
    G23: float | None = None
    G13: float | None = None

    def Q_material(self) -> np.ndarray:
        return stiffness_Q_from_engineering(self.E1, self.E2, self.G12, self.nu12)

    def Q_laminate(self) -> np.ndarray:
        return Q_bar(self.theta_deg, self.Q_material())

    def transverse_shear_moduli(self) -> tuple[float, float]:
        """G23, G13 — default from isotropic-in-transverse-plane approximation if missing."""
        if self.G23 is not None and self.G13 is not None:
            return self.G23, self.G13
        # Rule-of-thumb: out-of-plane shear moduli from E2, nu23
        nu23 = 0.45
        g23 = self.E2 / (2.0 * (1.0 + nu23))
        g13 = self.E1 / (2.0 * (1.0 + 0.3))
        return g23, g13


def abd_stack(plies: list[Ply]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble A, B, D for a laminate with plies ordered from bottom (z=-h/2) to top."""
    if not plies:
        raise ValueError("plies must be non-empty")
    n = len(plies)
    ts = [p.t for p in plies]
    h = float(sum(ts))
    z0 = -h / 2.0
    z_bot = z0
    A = np.zeros((3, 3), dtype=float)
    B = np.zeros((3, 3), dtype=float)
    D = np.zeros((3, 3), dtype=float)
    for p in plies:
        t = p.t
        z_mid = z_bot + t / 2.0
        Q = p.Q_laminate()
        A += Q * t
        B += Q * t * z_mid
        D += Q * (t * z_mid**2 + t**3 / 12.0)
        z_bot += t
    return A, B, D


def laminate_midstrains_curvatures(
    A: np.ndarray, B: np.ndarray, D: np.ndarray, N_vec: np.ndarray, M_vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve [N; M] = [[A,B],[B,D]] [eps0; kappa]."""
    Am = np.block([[A, B], [B, D]])
    rhs = np.concatenate([N_vec, M_vec])
    sol = np.linalg.solve(Am, rhs)
    return sol[:3], sol[3:]


def ply_stresses_bottom_top(
    plies: list[Ply], eps0: np.ndarray, kappa: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Ply-level σ = [σ11, σ22, τ12]^T at bottom and top of each ply (laminate axes).
    Strains at z: ε = ε0 + z * κ (CLPT).
    """
    h = sum(p.t for p in plies)
    z0 = -h / 2.0
    z_bot = z0
    out: list[tuple[np.ndarray, np.ndarray]] = []
    eps0 = np.asarray(eps0, dtype=float).ravel()
    kappa = np.asarray(kappa, dtype=float).ravel()
    for p in plies:
        t = p.t
        zb = z_bot
        zt = z_bot + t
        Q = p.Q_laminate()
        eps_b = eps0 + zb * kappa
        eps_t = eps0 + zt * kappa
        out.append((Q @ eps_b, Q @ eps_t))
        z_bot += t
    return out


def integrate_transverse_shear_stiffness(plies: list[Ply]) -> tuple[float, float]:
    """
    Return thickness-integrated effective transverse shear stiffnesses
    H44 = ∫ Q44(z) dz, H55 = ∫ Q55(z) dz in laminate axes (N/m).
    """
    h = sum(p.t for p in plies)
    z0 = -h / 2.0
    z_bot = z0
    h44 = 0.0
    h55 = 0.0
    for p in plies:
        g23, g13 = p.transverse_shear_moduli()
        q44, q55 = Q_bar_transverse(p.theta_deg, g23, g13)
        h44 += q44 * p.t
        h55 += q55 * p.t
        z_bot += p.t
    return h44, h55


def default_rectangular_plies(E: float, nu: float, thickness: float, n: int = 4) -> list[Ply]:
    """Isotropic plies stacked as [0/90/90/0]-style for a representative strip."""
    tply = thickness / n
    pattern = [0.0, 90.0, 90.0, 0.0]
    G = E / (2.0 * (1.0 + nu))
    plies = []
    for i in range(n):
        th = pattern[i % len(pattern)]
        plies.append(Ply(E1=E, E2=E, G12=G, nu12=nu, theta_deg=th, t=tply))
    return plies


def isotropic_ply_stack(
    E: float,
    nu: float,
    t_each: float,
    n_plies: int,
    theta_pattern_deg: list[float],
) -> list[Ply]:
    """Build plies from isotropic properties (E, nu) repeated with angles."""
    G = E / (2.0 * (1.0 + nu))
    plies = []
    for i in range(n_plies):
        th = theta_pattern_deg[i % len(theta_pattern_deg)]
        plies.append(Ply(E1=E, E2=E, G12=G, nu12=nu, theta_deg=th, t=t_each))
    return plies


def homogenized_axial_modulus(plies: list[Ply]) -> float:
    """Rule-of-mixtures axial stiffness 1/h ∫ E1_eff(z) dz with E1_eff = 1/S11 of rotated compliance."""
    h = sum(p.t for p in plies)
    s = 0.0
    for p in plies:
        Q = p.Q_laminate()
        S = np.linalg.inv(Q)
        E1_eff = 1.0 / S[0, 0]
        s += E1_eff * p.t
    return s / h


def representative_shear_modulus_thickness(plies: list[Ply]) -> tuple[float, float]:
    """Thickness-averaged G for y- and z-direction shear (from H44,H55 / h)."""
    h44, h55 = integrate_transverse_shear_stiffness(plies)
    h = sum(p.t for p in plies)
    if h < 1e-30:
        h = 1e-30
    return h44 / h, h55 / h


def stress_laminate_to_material(
    sigma_lam: np.ndarray, theta_deg: float
) -> np.ndarray:
    """
    Transform plane-stress engineering components from laminate to material axes.

    ``sigma_lam`` = [σx, σy, τxy] with x,y the laminate in-plane axes;
    ``theta_deg`` is the ply fibre angle (same convention as :func:`Q_bar`).
    Returns [σ11, σ22, τ12] in material axes [Pa].
    """
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    sx, sy, txy = float(sigma_lam[0]), float(sigma_lam[1]), float(sigma_lam[2])
    s11 = sx * c * c + sy * s * s + 2.0 * txy * s * c
    s22 = sx * s * s + sy * c * c - 2.0 * txy * s * c
    t12 = -sx * s * c + sy * s * c + txy * (c * c - s * s)
    return np.array([s11, s22, t12], dtype=float)


def tsai_wu_fi(
    sigma_mat: np.ndarray,
    Xt: float,
    Xc: float,
    Yt: float,
    Yc: float,
    S12: float,
) -> float:
    """
    Tsai–Wu failure index (FI = 1 on the failure surface).

    ``sigma_mat`` = [σ11, σ22, τ12] [Pa] in material axes.
    """
    s11, s22, s12 = float(sigma_mat[0]), float(sigma_mat[1]), float(sigma_mat[2])
    Xt = max(Xt, 1e-9)
    Xc = max(Xc, 1e-9)
    Yt = max(Yt, 1e-9)
    Yc = max(Yc, 1e-9)
    S12 = max(S12, 1e-9)
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


def ply_mid_strains(plies: list[Ply], eps0: np.ndarray, kappa: np.ndarray) -> list[np.ndarray]:
    """ε = [εx, εy, γxy] at each ply mid-thickness (laminate axes, engineering shear)."""
    h = sum(p.t for p in plies)
    z0 = -h / 2.0
    z_bot = z0
    eps0 = np.asarray(eps0, dtype=float).ravel()
    kappa = np.asarray(kappa, dtype=float).ravel()
    out: list[np.ndarray] = []
    for p in plies:
        t = p.t
        zm = z_bot + 0.5 * t
        out.append(eps0 + zm * kappa)
        z_bot += t
    return out


def ply_mid_stresses(
    plies: list[Ply], eps0: np.ndarray, kappa: np.ndarray
) -> list[np.ndarray]:
    """σ = [σx, σy, τxy] at each ply **mid-thickness** (laminate axes)."""
    h = sum(p.t for p in plies)
    z0 = -h / 2.0
    z_bot = z0
    eps0 = np.asarray(eps0, dtype=float).ravel()
    kappa = np.asarray(kappa, dtype=float).ravel()
    out: list[np.ndarray] = []
    for p in plies:
        t = p.t
        zm = z_bot + 0.5 * t
        Q = p.Q_laminate()
        eps_m = eps0 + zm * kappa
        out.append(Q @ eps_m)
        z_bot += t
    return out


def membrane_resultants_from_shell_stress(
    sigma_xx: float, sigma_yy: float, tau_xy: float, thickness: float
) -> np.ndarray:
    """[Nx, Ny, Nxy] [N/m] from uniform membrane stresses [Pa] and thickness [m]."""
    h = max(thickness, 1e-30)
    return np.array([sigma_xx * h, sigma_yy * h, tau_xy * h], dtype=float)


def clpt_ply_failure_indices(
    plies: list[Ply],
    N_vec: np.ndarray,
    M_vec: np.ndarray,
    Xt: float,
    Xc: float,
    Yt: float,
    Yc: float,
    S12: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Solve CLPT for (N, M), then Tsai–Wu FI per ply (mid-thickness).

    Returns
    -------
    fi_tw : (n_ply,) Tsai–Wu FI
    eps0, kappa : mid-surface strain and curvature (laminate axes)
    sig_lam : list of [σx, σy, τxy] per ply (laminate axes, mid-thickness)
    """
    A, B, D = abd_stack(plies)
    eps0, kappa = laminate_midstrains_curvatures(A, B, D, N_vec, M_vec)
    sig_lam = ply_mid_stresses(plies, eps0, kappa)
    n = len(plies)
    fi_tw = np.zeros(n, dtype=float)
    for i, p in enumerate(plies):
        sm = stress_laminate_to_material(sig_lam[i], p.theta_deg)
        fi_tw[i] = tsai_wu_fi(sm, Xt, Xc, Yt, Yc, S12)
    return fi_tw, eps0, kappa, sig_lam
