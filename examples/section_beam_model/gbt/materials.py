"""
materials.py — Constitutive models for GBT wall strips.

Provides:
  Lamina              — single ply (E1, E2, G12, nu12, angle, t)
  IsotropicMaterial   — isotropic plate (E, nu, t)
  LaminateMaterial    — classical lamination theory ABD matrix
  SandwichMaterial    — face sheets + foam/honeycomb core with shear stiffness
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Single ply
# ---------------------------------------------------------------------------

@dataclass
class Lamina:
    E1:    float
    E2:    float
    G12:   float
    nu12:  float
    angle: float = 0.0   # fibre angle in degrees
    t:     float = 1e-3

    def Q_global(self) -> NDArray:
        """3x3 reduced stiffness matrix in the global (x,s) coordinate system."""
        E1, E2, G12, nu12 = self.E1, self.E2, self.G12, self.nu12
        nu21 = nu12 * E2 / E1
        denom = 1.0 - nu12 * nu21
        Q11 = E1 / denom
        Q22 = E2 / denom
        Q12 = nu12 * E2 / denom
        Q66 = G12

        th = np.radians(self.angle)
        c, s = np.cos(th), np.sin(th)
        c2, s2, cs = c**2, s**2, c * s

        Q_bar = np.array([
            [Q11*c2**2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*s2**2,
             (Q11 + Q22 - 4*Q66)*c2*s2 + Q12*(c2**2 + s2**2),
             (Q11 - Q12 - 2*Q66)*c*s*c2 - (Q22 - Q12 - 2*Q66)*c*s*s2],
            [(Q11 + Q22 - 4*Q66)*c2*s2 + Q12*(c2**2 + s2**2),
             Q11*s2**2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*c2**2,
             (Q11 - Q12 - 2*Q66)*c*s*s2 - (Q22 - Q12 - 2*Q66)*c*s*c2],
            [(Q11 - Q12 - 2*Q66)*c*s*c2 - (Q22 - Q12 - 2*Q66)*c*s*s2,
             (Q11 - Q12 - 2*Q66)*c*s*s2 - (Q22 - Q12 - 2*Q66)*c*s*c2,
             (Q11 + Q22 - 2*Q12 - 2*Q66)*c2*s2 + Q66*(c2**2 + s2**2)],
        ])
        return Q_bar


# ---------------------------------------------------------------------------
# Isotropic material
# ---------------------------------------------------------------------------

class IsotropicMaterial:
    def __init__(self, E: float, nu: float, t: float):
        self.E = E; self.nu = nu; self.t = t

    def abd_matrix(self) -> NDArray:
        E, nu, t = self.E, self.nu, self.t
        factor = E / (1 - nu**2)
        Q = np.array([[factor,      nu*factor, 0.0              ],
                      [nu*factor,   factor,    0.0              ],
                      [0.0,         0.0,       factor*(1-nu)/2.0]])
        A = Q * t
        D = Q * t**3 / 12.0
        B = np.zeros((3, 3))
        abd = np.zeros((6, 6))
        abd[:3, :3] = A
        abd[:3, 3:] = B
        abd[3:, :3] = B
        abd[3:, 3:] = D
        return abd

    def shear_stiffness(self) -> NDArray:
        kappa = 5.0 / 6.0
        Gxz = self.E / (2 * (1 + self.nu))
        F = kappa * Gxz * self.t
        return np.array([[F, 0.0], [0.0, F]])

    @property
    def total_thickness(self) -> float:
        return self.t


# ---------------------------------------------------------------------------
# Laminate (CLT)
# ---------------------------------------------------------------------------

class LaminateMaterial:
    def __init__(self, plies: list[Lamina]):
        self.plies = plies

    @property
    def total_thickness(self) -> float:
        return sum(p.t for p in self.plies)

    def abd_matrix(self) -> NDArray:
        t_total = self.total_thickness
        z_mid = -t_total / 2.0
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        for ply in self.plies:
            Q = ply.Q_global()
            z1 = z_mid
            z2 = z_mid + ply.t
            A += Q * (z2 - z1)
            B += Q * (z2**2 - z1**2) / 2.0
            D += Q * (z2**3 - z1**3) / 3.0
            z_mid = z2
        abd = np.zeros((6, 6))
        abd[:3, :3] = A
        abd[:3, 3:] = B
        abd[3:, :3] = B
        abd[3:, 3:] = D
        return abd

    def shear_stiffness(self) -> NDArray:
        kappa = 5.0 / 6.0
        F = 0.0
        for ply in self.plies:
            G13 = ply.G12
            F  += kappa * G13 * ply.t
        return np.array([[F, 0.0], [0.0, F]])


# ---------------------------------------------------------------------------
# Sandwich material
# ---------------------------------------------------------------------------

class SandwichMaterial:
    def __init__(self, face_top: LaminateMaterial | IsotropicMaterial,
                 face_bot: LaminateMaterial | IsotropicMaterial,
                 core_thickness: float, core_G13: float, core_G23: float):
        self.face_top = face_top
        self.face_bot = face_bot
        self.core_thickness = core_thickness
        self.core_G13 = core_G13
        self.core_G23 = core_G23

    @property
    def total_thickness(self) -> float:
        return (self.face_top.total_thickness
                + self.core_thickness
                + self.face_bot.total_thickness)

    def abd_matrix(self) -> NDArray:
        # Assemble faces as offset laminates about sandwich mid-plane
        t_top = self.face_top.total_thickness
        t_bot = self.face_bot.total_thickness
        t_c   = self.core_thickness
        t_tot = self.total_thickness

        A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))

        def _add_face(face, z_bot, z_top):
            nonlocal A, B, D
            abd_f = face.abd_matrix()
            Af = abd_f[:3, :3]
            z_mid_f = (z_bot + z_top) / 2.0
            A += Af
            B += Af * z_mid_f
            D += Af * (z_mid_f**2) + abd_f[3:, 3:]

        z0 = -t_tot / 2.0
        _add_face(self.face_bot, z0, z0 + t_bot)
        _add_face(self.face_top, z0 + t_bot + t_c, z0 + t_tot)

        abd = np.zeros((6, 6))
        abd[:3, :3] = A; abd[:3, 3:] = B
        abd[3:, :3] = B; abd[3:, 3:] = D
        return abd

    def shear_stiffness(self) -> NDArray:
        # TODO: Face-sheet shear stiffness contribution is neglected here.
        # For thick composite face-sheets (t_face/t_core > ~0.2), add:
        #   F44 += face_top.shear_stiffness()[0,0] + face_bot.shear_stiffness()[0,0]
        # Currently valid only when core dominates through-thickness shear.
        # Core dominates through-thickness shear
        F44 = self.core_G13 * self.core_thickness
        F55 = self.core_G23 * self.core_thickness
        return np.array([[F44, 0.0], [0.0, F55]])
