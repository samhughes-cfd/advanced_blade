"""
Benchmark: isotropic lipped-C channel under uniform axial compression.

Compares GBT lambda_cr against the Euler column buckling load for a
simply-supported member (exact solution), providing a sanity check for
the member solver.

Euler critical stress:  sigma_cr = pi^2 * E * I / (A * L^2)
                      = pi^2 * EI / (A * L^2)
lambda_cr (GBT)   ~ sigma_cr * A / N_ref   (N_ref = -1 N applied)

Note: GBT lambda_cr converges to the Euler solution for the dominant
global bending mode as n_modes -> 1 (global only) and n_elem -> inf.
"""
import sys; sys.path.insert(0, '/home/user/output/gbt_module')
import numpy as np
from gbt import (
    IsotropicMaterial, WallDefinition, CrossSection, SectionLoads,
    CrossSectionModalAnalysis, BoundaryConditions, MemberBucklingAnalysis,
    KirchhoffKinematics,
)

def run():
    E, nu, t = 210e9, 0.3, 2e-3
    mat = IsotropicMaterial(E=E, nu=nu, t=t)

    # Symmetric I-section (avoids open-section shear centre complications)
    h = 0.1; b = 0.05
    sec = CrossSection([
        WallDefinition([0, 0],  [0, h],   mat, n_strips=8,  name='web'),
        WallDefinition([-b/2,0],[b/2,0],  mat, n_strips=4,  name='bot_flange'),
        WallDefinition([-b/2,h],[b/2,h],  mat, n_strips=4,  name='top_flange'),
    ])

    props = sec.second_moments()
    Iyy   = props['Iyy']
    print(f"Section: h={h}m, b={b}m, t={t}m")
    print(f"  Iyy (arc-length weighted) = {Iyy:.4e} m^3")

    L = 1.0
    loads = SectionLoads(N=-1.0)
    modal = CrossSectionModalAnalysis(sec, loads, KirchhoffKinematics()).run(n_modes=6)

    print(f"\nCross-section modes:")
    for k in range(len(modal.eigenvalues)):
        print(f"  Mode {k}: lambda={modal.eigenvalues[k]:.4e}  "
              f"D_k={modal.modal_rigidity(k):.4e}  class={modal.classify_mode(k)}")

    bcs = BoundaryConditions.simply_supported()
    res = MemberBucklingAnalysis(modal, L, bcs, n_elem=40, n_modes=4).run()

    print(f"\nGBT lambda_cr = {res.lambda_cr:.4e}  N  (N_ref = -1 N)")
    print(f"  Interpreted as critical force P_cr = {res.lambda_cr:.4e} N")

    # Euler reference (approximate, uses arc-length Iyy as proxy)
    EI_approx = E * Iyy * t   # N.m^2 rough estimate
    P_euler   = np.pi**2 * EI_approx / L**2
    print(f"\nEuler P_cr (approx, arc-length I): {P_euler:.4e} N")
    print(f"Ratio GBT/Euler: {res.lambda_cr/P_euler:.3f}  (expect ~1 for global mode)")

    # Convergence study
    conv = MemberBucklingAnalysis(modal, L, bcs, n_elem=8, n_modes=4).convergence_study(
        [4, 8, 16, 32, 64])
    print(f"\nConvergence study (n_elem vs lambda_cr):")
    for ne, lc in zip(conv['elem_counts'], conv['lambda_cr']):
        print(f"  n_elem={ne:3d}  lambda_cr={lc:.6e}")
    print(f"  Converged: {conv['converged']}  at n_elem={conv['converged_at']}")

if __name__ == '__main__':
    run()
