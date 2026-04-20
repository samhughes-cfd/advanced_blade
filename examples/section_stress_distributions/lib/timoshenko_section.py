"""
Timoshenko-style section shear stiffness and transverse shear stress recovery.

For a thin-walled strip with CLPT thickness-integrated shear stiffnesses H44, H55
(N/m), an effective shear strain in laminate axes (1 along midline tangent, 2 normal
in section plane) is γ = V_strip / H_eff for a strip carrying resultant V_strip.

At the section level we combine wall contributions into global (y,z) shear stiffnesses
GA_y, GA_z using direction cosines of each panel midline, then γ_yz ≈ V_y / (GA_y),
γ_xz ≈ V_z / (GA_z) for a diagonal effective stiffness (approximate decoupling).

Shear correction κ (default 5/6 for isotropic rectangular section) scales the
effective area used in GA = κ G A_s.
"""

from __future__ import annotations

import numpy as np

from lib.laminate_clpt import (
    Ply,
    default_rectangular_plies,
    integrate_transverse_shear_stiffness,
    representative_shear_modulus_thickness,
)


def panel_shear_stiffness_laminate_axes(
    plies: list[Ply],
) -> tuple[float, float, float]:
    """
    Returns H44, H55 (N/m) and h (m) for the laminate strip.
    Laminate 1-axis is taken aligned with panel tangent (caller rotates V into panel frame).
    """
    h44, h55 = integrate_transverse_shear_stiffness(plies)
    h = sum(p.t for p in plies)
    return h44, h55, h


def global_shear_stiffness_from_panels(
    panel_tangent_yz: list[tuple[float, float]],
    plies_per_panel: list[list[Ply]],
    kappa_y: float = 5.0 / 6.0,
    kappa_z: float = 5.0 / 6.0,
    panel_lengths: list[float] | None = None,
) -> tuple[float, float]:
    """
    Accumulate GA_y, GA_z (N) from discrete thin-wall panels.

    Each wall strip contributes shear stiffness κ G_avg (t L) weighted by the
    squared in-plane normal component (n_y, n_z) that couples global shear V_y, V_z
    to transverse shear in the wall (first-order smear).
    """
    GA_y = 0.0
    GA_z = 0.0
    for i, (ty, tz) in enumerate(panel_tangent_yz):
        plies = plies_per_panel[i]
        gy_eff, gz_eff = representative_shear_modulus_thickness(plies)
        Gavg = 0.5 * (gy_eff + gz_eff)
        h = sum(p.t for p in plies)
        length = panel_lengths[i] if panel_lengths is not None else 1.0
        tn = np.hypot(ty, tz)
        if tn < 1e-15:
            continue
        ty /= tn
        tz /= tn
        ny, nz = -tz, ty
        A_strip = length * h
        GA_y += kappa_y * Gavg * A_strip * (ny**2)
        GA_z += kappa_z * Gavg * A_strip * (nz**2)
    return GA_y, GA_z


def timoshenko_shear_strains(Vy: float, Vz: float, GA_y: float, GA_z: float) -> tuple[float, float]:
    """γ_y ≈ Vy/(GA_y), γ_z ≈ Vz/(GA_z) (decoupled diagonal approximation)."""
    if GA_y < 1e-30:
        GA_y = 1e-30
    if GA_z < 1e-30:
        GA_z = 1e-30
    return Vy / GA_y, Vz / GA_z


def transverse_shear_stress_zavg_from_resultants(
    Vy: float, Vz: float, h44: float, h55: float, cos_align: float, sin_align: float
) -> tuple[float, float]:
    """
    Average τ13, τ23 in laminate axes for a unit-width strip (approximate):
      τ ≈ V / H  with V the shear resultant in the 2-3 plane resolved along 2 and 3.
    Here we return representative (τ_lam_13, τ_lam_23) from global Vy,Vz projected
    onto panel axes using cos_align = panel tangent · y_hat.
    """
    # Global V in section y,z; panel tangent t = (cos_align, sin_align) in y,z.
    # Shear resultant magnitude in plane of section for St. Venant type: |V|.
    V_panel = Vy * sin_align - Vz * cos_align  # shear along normal-ish component (rough)
    H = max(h44, h55)
    if H < 1e-30:
        H = 1e-30
    tau_avg = V_panel / H
    return tau_avg * 0.5, tau_avg * 0.5
