"""modal.py - GBT cross-section modal analysis with shared-node assembly."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from .section import CrossSection
from .kinematics import KinematicModel, KirchhoffKinematics
from .prebuckling import PreBucklingAnalysis, SectionLoads


def _strip_elastic(abd, kin, ds, Ks=None):
    ndof = kin.n_dof_per_strip
    K = np.zeros((ndof, ndof))
    A, Bm_, D = abd[:3, :3], abd[:3, 3:], abd[3:, 3:]
    Bm = kin.membrane_bkin(ds)
    Bb = kin.bending_bkin(ds)
    K += Bm.T @ A @ Bm * ds
    K += Bb.T @ D @ Bb * ds
    K += Bm.T @ Bm_ @ Bb * ds
    K += Bb.T @ Bm_.T @ Bm * ds
    if Ks is not None and np.any(Ks != 0):
        Bs = kin.shear_bkin(ds)
        K += Bs.T @ Ks @ Bs * ds
    return K


def _strip_geom(Nx, Ns, Nxs, kin, ds):
    ndof = kin.n_dof_per_strip
    Kg = np.zeros((ndof, ndof))
    L = ds
    dw = np.zeros((1, ndof))
    dv = np.zeros((1, ndof))
    if ndof == 8:
        dw[0, 2] = -1/L; dw[0, 6] = 1/L
        dv[0, 1] = -1/L; dv[0, 5] = 1/L
    else:
        dw[0, 2] = -1/L; dw[0, 7] = 1/L
        dv[0, 1] = -1/L; dv[0, 6] = 1/L
    Kg += Nx * (dw.T @ dw) * ds
    Kg += Ns * (dv.T @ dv) * ds
    Kg += Nxs * (dw.T @ dv + dv.T @ dw) * ds
    return Kg


def _build_inertia_matrix(section, kin):
    """
    Lumped inertia (mass-like) matrix M for the cross-section eigenproblem.
    C phi = lambda M phi  ->  load-independent mode shapes.
    Constructed as thickness-weighted diagonal: half the strip arc-length
    mass distributed to each bounding node.
    """
    ndpn  = kin.n_dof_per_strip // 2
    n_dof = section.n_nodes * ndpn
    M     = np.zeros((n_dof, n_dof))
    for i in range(section.n_strips):
        ds    = section.get_strip(i).length
        t     = section.strip_thickness(i)
        gdofs = section.strip_global_dofs(i, ndpn)
        w     = t * ds / 2.0
        for gd in gdofs:
            M[gd, gd] += w
    # Guarantee strict positive definiteness
    M += np.eye(n_dof) * 1e-14 * max(M.max(), 1.0)
    return M


def assemble_section_matrices(section, stress_resultants, kin):
    ndpn  = kin.n_dof_per_strip // 2
    n_dof = section.n_nodes * ndpn
    C = np.zeros((n_dof, n_dof))
    B = np.zeros((n_dof, n_dof))
    for i in range(section.n_strips):
        abd  = section.strip_abd(i)
        ds   = section.get_strip(i).length
        Ks   = section.strip_shear_stiffness(i)
        Ke   = _strip_elastic(abd, kin, ds, Ks)
        Nx, Ns, Nxs = stress_resultants[i]
        Kg   = _strip_geom(Nx, Ns, Nxs, kin, ds)
        gdofs = section.strip_global_dofs(i, ndpn)
        for ii, gi in enumerate(gdofs):
            for jj, gj in enumerate(gdofs):
                C[gi, gj] += Ke[ii, jj]
                B[gi, gj] += Kg[ii, jj]
    return C, B


@dataclass
class ModalResult:
    eigenvalues: NDArray
    modes:       NDArray
    C:           NDArray
    B_geom:      NDArray
    n_nodes:     int
    n_dof:       int

    def modal_rigidity(self, k):
        phi = self.modes[:, k]; return float(phi @ self.C @ phi)

    def modal_geometric_stiffness(self, k):
        phi = self.modes[:, k]; return float(phi @ self.B_geom @ phi)

    def modal_coupling(self, j, k):
        return float(self.modes[:, j] @ self.C @ self.modes[:, k])

    def modal_geom_coupling(self, j, k):
        return float(self.modes[:, j] @ self.B_geom @ self.modes[:, k])

    def critical_eigenvalue(self):
        pos = self.eigenvalues[self.eigenvalues > 1e-10]
        return float(pos[0]) if len(pos) > 0 else np.inf

    def classify_mode(self, k):
        pos = self.eigenvalues[self.eigenvalues > 1e-12]
        if len(pos) == 0: return "undetermined"
        med = float(np.median(pos))
        lam = self.eigenvalues[k]
        if   lam < 1e-10:       return "rigid_body"
        elif lam < 0.05 * med:  return "local"
        elif lam < 0.5  * med:  return "distortional"
        else:                   return "global"

    def orthogonality_check(self, tol=1e-6):
        n = min(len(self.eigenvalues), 20)
        for j in range(n):
            Djj = self.modal_rigidity(j)
            for k in range(j + 1, n):
                ref = max(abs(Djj), abs(self.modal_rigidity(k)), 1e-30)
                if abs(self.modal_coupling(j, k)) / ref > tol:
                    return False
        return True

    def participation_factors(self):
        n = len(self.eigenvalues)
        factors = np.zeros((n, 3))
        ndof = self.n_dof; half = ndof // 2
        for k in range(n):
            phi   = self.modes[:, k]
            total = self.modal_rigidity(k)
            if abs(total) < 1e-30:
                factors[k] = [1/3, 1/3, 1/3]; continue
            mem = abs(float(phi[:half] @ self.C[:half, :half] @ phi[:half]))
            ben = abs(float(phi[half:] @ self.C[half:, half:] @ phi[half:]))
            s   = max(mem + ben, 1e-30)
            factors[k, 0] = mem / s
            factors[k, 1] = ben / s
        return factors


class CrossSectionModalAnalysis:
    def __init__(self, section, loads=None, kinematic_model=None):
        self.section = section
        self.loads   = loads if loads is not None else SectionLoads(N=-1.0)
        self.kin     = kinematic_model if kinematic_model is not None else KirchhoffKinematics()

    def run(self, n_modes=None):
        stress      = PreBucklingAnalysis(self.section, self.loads).run()
        C, B        = assemble_section_matrices(self.section, stress, self.kin)
        M           = _build_inertia_matrix(self.section, self.kin)
        eigs, vecs  = eigh(C, M)
        pos         = eigs > 1e-10 * max(abs(eigs.max()), 1.0)
        ev, vv      = eigs[pos], vecs[:, pos]
        if n_modes is not None:
            m = min(n_modes, len(ev)); ev = ev[:m]; vv = vv[:, :m]
        return ModalResult(eigenvalues=ev, modes=vv, C=C, B_geom=B,
                           n_nodes=self.section.n_nodes, n_dof=C.shape[0])
