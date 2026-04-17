"""
member.py — GBT member buckling analysis with coupled modal formulation.

Formulation
-----------
Member displacement field:
    u(x, s) = sum_k  phi_k(s) * V_k(x)

Element matrices:
    K_elem[jk]  = D_jk * integral_0^Le  d2V_j/dx2 * d2V_k/dx2  dx
    Kg_elem[jk] = B_jk * integral_0^Le  dV_j/dx   * dV_k/dx    dx

where:
    D_jk = phi_j^T C phi_k
    B_jk = phi_j^T M_sigma phi_k  (stress-weighted inertia, negated for compression)

Modal geometric stiffness per unit length:
    b_jk = integral_section  sigma_x(s) * phi_j(s) * phi_k(s) * t  ds

Approximated as lumped nodal contributions from pre-buckling Nx per strip.

Eigenproblem:  K d = lambda Kg d
Solved via:    A = K^{-1} Kg,  lambda_cr = min positive eigenvalue of A.

References
----------
Schardt (1989); Silvestre & Camotim (2002) Comput. Struct. 80, 2127-2148.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .modal       import ModalResult
from .boundary    import BoundaryConditions
from .prebuckling import PreBucklingAnalysis, SectionLoads
from .kinematics  import KirchhoffKinematics


# ---------------------------------------------------------------------------
# Hermite shape functions
# ---------------------------------------------------------------------------

def hermite_shape_functions(xi: float, L: float) -> NDArray:
    """Cubic Hermite shape functions at xi in [0,1], element length L."""
    N1 =  1 - 3*xi**2 + 2*xi**3
    N2 =  L * xi * (1 - xi)**2
    N3 =  3*xi**2 - 2*xi**3
    N4 =  L * xi**2 * (xi - 1)
    return np.array([N1, N2, N3, N4])


def hermite_d2(xi: float, L: float) -> NDArray:
    """Second derivative d2N/dx2 at xi in [0,1]."""
    d2N1 = (1.0 / L**2) * (-6  + 12*xi)
    d2N2 = (1.0 / L)    * (-4  +  6*xi)
    d2N3 = (1.0 / L**2) * ( 6  - 12*xi)
    d2N4 = (1.0 / L)    * (-2  +  6*xi)
    return np.array([d2N1, d2N2, d2N3, d2N4])


def hermite_d1(xi: float, L: float) -> NDArray:
    """
    First derivative dN/dx of the four cubic Hermite shape functions at xi in [0,1].

    Shape functions and their exact x-derivatives:
        N1 = 1 - 3xi^2 + 2xi^3        =>  dN1/dx = (1/L)*(-6xi + 6xi^2)
        N2 = L*xi*(1-xi)^2             =>  dN2/dx = 1 - 4xi + 3xi^2    (dimensionless)
        N3 = 3xi^2 - 2xi^3             =>  dN3/dx = (1/L)*(6xi - 6xi^2)
        N4 = L*xi^2*(xi-1)             =>  dN4/dx = 3xi^2 - 2xi         (dimensionless)

    N2 and N4 carry an explicit factor of L so their x-derivatives are dimensionless
    (the L in dN/dxi cancels the 1/L from the chain rule dx=L*dxi).
    """
    dN1 = (1.0 / L) * (-6*xi + 6*xi**2)
    dN2 =              (1 - 4*xi + 3*xi**2)
    dN3 = (1.0 / L) * ( 6*xi - 6*xi**2)
    dN4 =              (3*xi**2 - 2*xi)
    return np.array([dN1, dN2, dN3, dN4])


# ---------------------------------------------------------------------------
# Gauss quadrature
# ---------------------------------------------------------------------------

_GAUSS = {
    2: (np.array([0.5 - 0.5/np.sqrt(3), 0.5 + 0.5/np.sqrt(3)]),
        np.array([0.5, 0.5])),
    3: (np.array([0.5 - 0.5*np.sqrt(3/5), 0.5, 0.5 + 0.5*np.sqrt(3/5)]),
        np.array([5/18, 8/18, 5/18])),
    4: (np.array([0.0694318, 0.330009, 0.669991, 0.930568]),
        np.array([0.173927, 0.326073, 0.326073, 0.173927])),
}


def _gauss(n: int = 3):
    return _GAUSS.get(n, _GAUSS[3])


# ---------------------------------------------------------------------------
# Element matrices
# ---------------------------------------------------------------------------

def _element_stiffness(D_jk: float, L: float, n_gauss: int = 3) -> NDArray:
    """4x4 elastic stiffness: D_jk * integral d2N/dx2 (x) d2N/dx2 dx."""
    K = np.zeros((4, 4))
    pts, wts = _gauss(n_gauss)
    for xi, w in zip(pts, wts):
        d2N = hermite_d2(xi, L)
        K  += D_jk * np.outer(d2N, d2N) * w * L
    return K


def _element_geom_stiffness(B_jk: float, L: float, n_gauss: int = 3) -> NDArray:
    """4x4 geometric stiffness: B_jk * integral dN/dx (x) dN/dx dx."""
    Kg = np.zeros((4, 4))
    pts, wts = _gauss(n_gauss)
    for xi, w in zip(pts, wts):
        dN  = hermite_d1(xi, L)
        Kg += B_jk * np.outer(dN, dN) * w * L
    return Kg


# ---------------------------------------------------------------------------
# Global assembly
# ---------------------------------------------------------------------------

def _assemble_global(
    n_elem:   int,
    n_modes:  int,
    L_elem:   float,
    D_matrix: NDArray,
    B_matrix: NDArray,
    n_gauss:  int = 3,
) -> tuple[NDArray, NDArray]:
    """
    Assemble global K and Kg for the full member.

    DOF layout (per mode k):
        [w_k(node0), theta_k(node0), w_k(node1), theta_k(node1), ..., theta_k(nodeN)]
    All modes concatenated: total DOFs = 2 * (n_elem+1) * n_modes.
    """
    n_nodes   = n_elem + 1
    n_dof_tot = 2 * n_nodes * n_modes
    K_global  = np.zeros((n_dof_tot, n_dof_tot))
    Kg_global = np.zeros((n_dof_tot, n_dof_tot))

    for e in range(n_elem):
        for j in range(n_modes):
            for k in range(n_modes):
                Ke  = _element_stiffness(     D_matrix[j, k], L_elem, n_gauss)
                Kge = _element_geom_stiffness(B_matrix[j, k], L_elem, n_gauss)
                base_j = j * 2 * n_nodes
                base_k = k * 2 * n_nodes
                g_j = [base_j + 2*e,     base_j + 2*e + 1,
                       base_j + 2*(e+1), base_j + 2*(e+1) + 1]
                g_k = [base_k + 2*e,     base_k + 2*e + 1,
                       base_k + 2*(e+1), base_k + 2*(e+1) + 1]
                for ii, gi in enumerate(g_j):
                    for jj, gj in enumerate(g_k):
                        K_global [gi, gj] += Ke [ii, jj]
                        Kg_global[gi, gj] += Kge[ii, jj]
    return K_global, Kg_global


def _apply_boundary_conditions(
    K:        NDArray,
    Kg:       NDArray,
    bcs:      BoundaryConditions,
    n_nodes:  int,
    n_modes:  int,
) -> tuple[NDArray, NDArray, list[int]]:
    """Apply BCs: add elastic springs, then eliminate constrained DOFs."""
    K_s, _ = bcs.spring_contributions(n_nodes, n_modes)
    K = K + K_s
    constrained = set(bcs.constrained_dofs(n_nodes, n_modes))
    free = [i for i in range(K.shape[0]) if i not in constrained]
    return K[np.ix_(free, free)], Kg[np.ix_(free, free)], free


# ---------------------------------------------------------------------------
# Stress-weighted modal geometric stiffness
# ---------------------------------------------------------------------------

def _build_stress_weighted_B(
    modal_result: ModalResult,
    section,
    loads:   SectionLoads,
    n_modes: int,
) -> NDArray:
    """
    Build (n_modes x n_modes) modal geometric coupling matrix B_jk.

    Uses a lumped stress-weighted inertia matrix M_sigma assembled from
    per-strip axial stress resultants Nx(s):

        (M_sigma)_dd += Nx_i * ds_i / 2   for each displacement DOF d

    Then:  B_jk = -( phi_j^T M_sigma phi_k )

    The negation makes B_jk positive for compressive Nx < 0, so that Kg
    assembled from B_jk is positive semi-definite and the eigenproblem
    K d = lambda Kg d has positive lambda_cr for compressive buckling.

    Only displacement DOFs (every ndpn-th index) are used for the projection,
    matching the basis of the inertia matrix M built in modal.py.
    """
    kin   = KirchhoffKinematics()
    ndpn  = kin.n_dof_per_strip // 2   # DOFs per node (4 for Kirchhoff)
    n_dof = section.n_nodes * ndpn

    pb     = PreBucklingAnalysis(section, loads)
    Nx_arr = pb.axial_stress_resultants()  # (n_strips,)

    # Build diagonal M_sigma: only accumulate on w-type (displacement) DOFs.
    # In Kirchhoff, the DOF order per node is [u, v, w, theta_s].
    # The out-of-plane displacement w is at local index 2 within each node block.
    # M_sigma_{dd} += Nx_i * ds_i / 2  for each strip node d.
    M_sigma = np.zeros(n_dof)
    for i in range(section.n_strips):
        Nx    = Nx_arr[i]
        ds    = section.get_strip(i).length
        gdofs = section.strip_global_dofs(i, ndpn)
        w     = Nx * ds / 2.0
        for gd in gdofs:
            if gd < n_dof:
                M_sigma[gd] += w

    # Project through mode shapes: B_jk = phi_j^T diag(M_sigma) phi_k
    Phi   = modal_result.modes[:, :n_modes]   # (n_dof, n_modes)
    B_mat = Phi.T @ (M_sigma[:, None] * Phi)  # equivalent to Phi.T @ diag(M_sigma) @ Phi

    # Negate: compressive Nx < 0 => M_sigma < 0 => B_mat < 0.
    # We need B_jk > 0 so Kg is positive and lambda_cr is positive.
    # Physical meaning: lambda_cr = min { pi^2 * D_k / (B_k * L^2) } > 0.
    return -B_mat


# ---------------------------------------------------------------------------
# Member buckling result
# ---------------------------------------------------------------------------

@dataclass
class MemberBucklingResult:
    """
    Result of a GBT member buckling analysis.

    Attributes
    ----------
    lambda_cr     : float    Critical load multiplier (min positive eigenvalue).
    eigenvalues   : NDArray  All positive eigenvalues sorted ascending.
    eigenvectors  : NDArray  Corresponding amplitude vectors (free DOFs).
    n_elem        : int      Number of Hermite elements used.
    n_modes       : int      Number of cross-section modes included.
    member_length : float    Member length [m].
    buckling_mode : NDArray  Full DOF vector for the critical mode.
    """
    lambda_cr:     float
    eigenvalues:   NDArray
    eigenvectors:  NDArray
    n_elem:        int
    n_modes:       int
    member_length: float
    buckling_mode: NDArray

    def n_half_waves(self) -> int:
        """Estimate number of half-wavelengths from displacement DOF sign changes."""
        w_dofs = self.buckling_mode[::2]
        signs  = np.sign(w_dofs[np.abs(w_dofs) > 1e-6 * np.abs(w_dofs).max()])
        if len(signs) < 2:
            return 1
        return int(np.sum(np.diff(signs) != 0)) // 2 + 1

    def modal_participation(self, n_modes: int, n_nodes: int) -> NDArray:
        """L2-norm participation fraction of each cross-section mode."""
        full           = self.buckling_mode
        n_dof_per_mode = 2 * n_nodes
        parts = np.zeros(n_modes)
        for k in range(n_modes):
            s = k * n_dof_per_mode
            e = s + n_dof_per_mode
            if e <= len(full):
                parts[k] = np.linalg.norm(full[s:e])
        total = parts.sum()
        return parts / total if total > 1e-30 else parts

    def summary(self) -> str:
        lines = [
            "MemberBucklingResult",
            f"  lambda_cr     = {self.lambda_cr:.6g}",
            f"  n_elem        = {self.n_elem}",
            f"  n_modes       = {self.n_modes}",
            f"  member_length = {self.member_length:.4f} m",
            f"  n_half_waves  ≈ {self.n_half_waves()}",
            f"  next eigenvalues: {self.eigenvalues[1:min(5,len(self.eigenvalues))]}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class MemberBucklingAnalysis:
    """
    GBT member buckling analysis with coupled cross-section modes.

    Parameters
    ----------
    modal_result : ModalResult
        Output of CrossSectionModalAnalysis.run().
    length       : float
        Member length [m].
    bcs          : BoundaryConditions or None
        End boundary conditions (default: simply supported).
    n_elem       : int
        Number of Hermite beam elements (default 20).
    n_modes      : int or None
        Cross-section modes to include (default: all).
    n_gauss      : int
        Gauss points per element (default 3).
    loads        : SectionLoads or None
        Applied section loads. Required for stress-weighted B matrix.
    section      : CrossSection or None
        Cross-section object. Required for stress-weighted B matrix.
        If both are None, falls back to diagonal eigenvalue-scaled B.
    """

    def __init__(
        self,
        modal_result: ModalResult,
        length:       float,
        bcs:          BoundaryConditions | None = None,
        n_elem:       int  = 20,
        n_modes:      int | None = None,
        n_gauss:      int  = 3,
        loads:        SectionLoads | None = None,
        section=None,
    ):
        self.modal   = modal_result
        self.length  = float(length)
        self.bcs     = bcs if bcs is not None else BoundaryConditions.simply_supported()
        self.n_elem  = n_elem
        self.n_modes = n_modes if n_modes is not None else len(modal_result.eigenvalues)
        self.n_gauss = n_gauss
        self.loads   = loads
        self.section = section

    def _build_D_matrix(self) -> NDArray:
        """(n_modes x n_modes) modal elastic coupling D_jk = phi_j^T C phi_k."""
        m = self.n_modes
        D = np.zeros((m, m))
        for j in range(m):
            for k in range(m):
                D[j, k] = self.modal.modal_coupling(j, k)
        return D

    def _build_B_matrix(self) -> NDArray:
        """
        (n_modes x n_modes) modal geometric coupling B_jk (positive for compression).

        Uses stress-weighted M_sigma when section+loads available.
        Falls back to diagonal B_kk = lambda_k * D_kk otherwise.
        """
        if self.section is not None and self.loads is not None:
            return _build_stress_weighted_B(
                self.modal, self.section, self.loads, self.n_modes
            )
        # Fallback diagonal approximation
        m = self.n_modes
        B = np.zeros((m, m))
        for k in range(m):
            B[k, k] = self.modal.eigenvalues[k] * self.modal.modal_rigidity(k)
        return B

    def run(self, n_eigs: int = 10) -> MemberBucklingResult:
        """
        Assemble and solve the GBT member buckling eigenproblem.

        Solves K d = lambda Kg d via A = K^{-1} Kg using Cholesky factorisation
        of K (always positive definite for a constrained structure).

        Returns
        -------
        MemberBucklingResult
        """
        m       = self.n_modes
        n_nodes = self.n_elem + 1
        L_elem  = self.length / self.n_elem

        D_mat = self._build_D_matrix()
        B_mat = self._build_B_matrix()

        K_global, Kg_global = _assemble_global(
            self.n_elem, m, L_elem, D_mat, B_mat, self.n_gauss
        )

        K_free, Kg_free, free_dofs = _apply_boundary_conditions(
            K_global, Kg_global, self.bcs, n_nodes, m
        )

        n_free = K_free.shape[0]
        if n_free < 2:
            raise ValueError(
                "Fewer than 2 free DOFs after boundary conditions. "
                "Increase n_elem or check BoundaryConditions."
            )

        # Regularise K (symmetric PD for any constrained elastic structure)
        eps = 1e-12 * max(float(np.abs(K_free).max()), 1.0)
        K_reg = K_free + np.eye(n_free) * eps

        # Shift-invert: A = K^{-1} Kg, eigenvalues of A = lambda_cr
        try:
            L_chol = np.linalg.cholesky(K_reg)
            tmp = np.linalg.solve(L_chol, Kg_free)
            A   = np.linalg.solve(L_chol, tmp.T).T
        except np.linalg.LinAlgError:
            A = np.linalg.solve(K_reg, Kg_free)

        A = 0.5 * (A + A.T)  # symmetrise to suppress numerical noise

        lam_all, vecs = np.linalg.eigh(A)

        pos_mask = (lam_all > 1e-10) & np.isfinite(lam_all)
        if not np.any(pos_mask):
            raise RuntimeError(
                "No positive eigenvalues found.\n"
                "Checklist:\n"
                "  1. SectionLoads.N must be negative (compression), e.g. N=-EA*1e-3\n"
                "  2. n_elem >= 2 required\n"
                "  3. Pass section= and loads= for physically correct B matrix\n"
                "  4. Boundary conditions must leave at least 2 free displacement DOFs"
            )

        order    = np.argsort(lam_all[pos_mask])
        pos_eigs = lam_all[pos_mask][order]
        pos_vecs = vecs[:, pos_mask][:, order]

        n_total   = K_global.shape[0]
        full_mode = np.zeros(n_total)
        full_mode[free_dofs] = pos_vecs[:, 0]

        return MemberBucklingResult(
            lambda_cr     = float(pos_eigs[0]),
            eigenvalues   = pos_eigs,
            eigenvectors  = pos_vecs,
            n_elem        = self.n_elem,
            n_modes       = m,
            member_length = self.length,
            buckling_mode = full_mode,
        )

    def convergence_study(
        self,
        elem_counts: list[int] | None = None,
        tol: float = 1e-4,
    ) -> dict:
        """
        Run at increasing element counts and report lambda_cr convergence.

        Returns dict: elem_counts, lambda_cr, converged (bool), converged_at (int|None).
        """
        if elem_counts is None:
            elem_counts = [4, 8, 16, 32, 64]

        results   = []
        converged = False
        conv_at   = None

        for ne in elem_counts:
            ana = MemberBucklingAnalysis(
                self.modal, self.length, self.bcs,
                n_elem=ne, n_modes=self.n_modes, n_gauss=self.n_gauss,
                loads=self.loads, section=self.section,
            )
            try:
                results.append(ana.run().lambda_cr)
            except RuntimeError:
                results.append(float("nan"))
                continue

            if len(results) >= 2 and not converged:
                prev, curr = results[-2], results[-1]
                if prev > 0 and abs(curr - prev) / prev < tol:
                    converged = True
                    conv_at   = ne

        return dict(elem_counts=elem_counts, lambda_cr=results,
                    converged=converged, converged_at=conv_at)

    def signature_curve(
        self,
        L_min: float | None = None,
        L_max: float | None = None,
        n_pts: int = 30,
    ) -> dict:
        """
        Compute GBT signature curve: lambda_cr vs half-wavelength.

        Returns dict: half_wave_lengths (n_pts,), lambda_cr (n_pts,).
        """
        if L_min is None:
            L_min = self.length / 20.0
        if L_max is None:
            L_max = 5.0 * self.length

        hw_values  = np.geomspace(L_min, L_max, n_pts)
        lam_values = np.full(n_pts, np.nan)

        for i, hw in enumerate(hw_values):
            ana = MemberBucklingAnalysis(
                self.modal, hw,
                BoundaryConditions.simply_supported(),
                n_elem=max(8, self.n_elem),
                n_modes=self.n_modes, n_gauss=self.n_gauss,
                loads=self.loads, section=self.section,
            )
            try:
                lam_values[i] = ana.run().lambda_cr
            except RuntimeError:
                pass

        return dict(half_wave_lengths=hw_values, lambda_cr=lam_values)