"""modal.py - GBT cross-section modal analysis with shared-node assembly."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from .section import CrossSection
from .kinematics import KinematicModel, KirchhoffKinematics
from .prebuckling import PreBucklingAnalysis, SectionLoads

# Classical beam modes for spanwise export (distortion modes opt-in via explicit labels).
DEFAULT_BEAM_EXPORT_MODE_LABELS: tuple[str, ...] = (
    "axial",
    "bending_x",
    "bending_y",
    "torsion",
)


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


def _nodal_w_and_coords(phi: NDArray, section: CrossSection, ndpn: int) -> tuple[NDArray, NDArray]:
    """Out-of-plane (w) DOF per node and node (y,z) coordinates; Kirchhoff: w is local DOF index 2."""
    if ndpn < 3:
        return np.zeros(section.n_nodes), section.node_coordinates
    w_idx = 2
    w = np.array([phi[i * ndpn + w_idx] for i in range(section.n_nodes)], dtype=np.float64)
    yz = section.node_coordinates
    return w, yz


def _mode_torsion_raw(phi: NDArray, section: CrossSection, ndpn: int) -> float:
    ts = 0.0
    for si in range(section.n_strips):
        s = section.get_strip(si)
        wi = phi[s.node_i * ndpn + 2]
        wj = phi[s.node_j * ndpn + 2]
        ts += abs(wj - wi) / (s.length + 1e-30)
    return float(ts)


def _corr_yz(phi: NDArray, section: CrossSection, ndpn: int) -> tuple[float, float]:
    w, yz = _nodal_w_and_coords(phi, section, ndpn)
    yc, zc = float(np.mean(yz[:, 0])), float(np.mean(yz[:, 1]))
    yd = yz[:, 0] - yc
    zd = yz[:, 1] - zc
    var_y = float(np.dot(yd, yd)) + 1e-30
    var_z = float(np.dot(zd, zd)) + 1e-30
    wy = float(np.dot(w, yd))
    wz = float(np.dot(w, zd))
    wn = float(np.linalg.norm(w)) + 1e-30
    corr_y = abs(wy) / (wn * np.sqrt(var_y))
    corr_z = abs(wz) / (wn * np.sqrt(var_z))
    return corr_y, corr_z


def _classical_export_indices(result: "ModalResult", section: CrossSection) -> dict[str, int]:
    """Pick distinct mode indices for axial / bending_x / bending_y / torsion."""
    n = len(result.eigenvalues)
    ndpn = result.n_dof // max(section.n_nodes, 1)
    pf = result.participation_factors()
    mem = np.array([float(pf[k, 0]) if k < pf.shape[0] else 0.5 for k in range(n)], dtype=np.float64)
    cy = np.zeros(n, dtype=np.float64)
    cz = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for k in range(n):
        phi = result.modes[:, k]
        cy[k], cz[k] = _corr_yz(phi, section, ndpn)
        tr[k] = _mode_torsion_raw(phi, section, ndpn)

    cand_ax = [i for i in range(n) if mem[i] > 0.52 and cy[i] < 0.2 and cz[i] < 0.2]
    if not cand_ax:
        cand_ax = [i for i in range(n) if mem[i] > 0.52]
    if not cand_ax:
        cand_ax = list(range(n))
    axial_k = int(cand_ax[int(np.argmin(result.eigenvalues[cand_ax]))])

    cand_t = [i for i in range(n) if i != axial_k and mem[i] < 0.58]
    if not cand_t:
        cand_t = [i for i in range(n) if i != axial_k]
    torsion_k = int(cand_t[int(np.argmax(tr[cand_t]))])

    used = {axial_k, torsion_k}
    bend_cand = [i for i in range(n) if i not in used]
    if not bend_cand:
        bend_x_k = bend_y_k = int((axial_k + 1) % max(n, 1))
    else:
        bend_x_k = int(bend_cand[int(np.argmax(cz[bend_cand]))])
        used.add(bend_x_k)
        bend_y_pool = [i for i in range(n) if i not in used]
        if not bend_y_pool:
            bend_y_k = bend_x_k
        else:
            by_sorted = sorted(bend_y_pool, key=lambda i: cy[i], reverse=True)
            top_by = by_sorted[: max(5, min(8, len(by_sorted)))]
            bend_y_k = int(top_by[int(np.argmin(result.eigenvalues[top_by]))])
    return {
        "axial": axial_k,
        "bending_x": bend_x_k,
        "bending_y": bend_y_k,
        "torsion": torsion_k,
    }


def classical_export_indices(result: "ModalResult", section: CrossSection) -> dict[str, int]:
    """Indices of axial / bending_x / bending_y / torsion modes in ``result.modes`` columns."""
    return _classical_export_indices(result, section)


def _export_label_for_mode(result: "ModalResult", section: CrossSection, k: int) -> str:
    """
    Export label: four classical indices from :func:`_classical_export_indices`,
    short-wave ``distortion_*`` plate modes, or generic ``distortion_*`` for others.
    """
    ref = result.classification_eigenvalues
    pos = result.eigenvalues[result.eigenvalues > 1e-12] if ref is None else ref[ref > 1e-12]
    lam = float(result.eigenvalues[k])
    lam_max = float(np.max(pos)) if len(pos) else lam
    if lam < 1e-14 * max(lam_max, 1.0):
        return "rigid_body"

    ndpn = result.n_dof // max(section.n_nodes, 1)
    picks = _classical_export_indices(result, section)
    inv = {v: lab for lab, v in picks.items()}
    if k in inv:
        return inv[k]

    phi = result.modes[:, k]
    cy, cz = _corr_yz(phi, section, ndpn)
    tr = _mode_torsion_raw(phi, section, ndpn)
    phi_norm = float(np.linalg.norm(phi)) + 1e-30
    if lam < 1e-7 * max(lam_max, 1.0) and max(cy, cz) < 0.08 and tr / phi_norm < 0.25:
        idx = sum(
            1
            for j in range(k)
            if float(result.eigenvalues[j]) < 1e-7 * max(lam_max, 1.0)
            and max(_corr_yz(result.modes[:, j], section, ndpn)) < 0.08
            and _mode_torsion_raw(result.modes[:, j], section, ndpn)
            / (float(np.linalg.norm(result.modes[:, j])) + 1e-30)
            < 0.25
        )
        return f"distortion_{idx}"

    return "undetermined_export"


def select_modes(
    result: "ModalResult",
    mode_labels: Sequence[str] | None = None,
    n_modes: int | None = None,
) -> "ModalResult":
    """
    Filter modes by export label or by count of lowest generalized eigenvalues.

    Exactly one of ``mode_labels`` or ``n_modes`` may be set; if both are ``None``,
    returns a copy of ``result`` with the same data. ``C`` and ``B_geom`` are unchanged
    (full ``n_dof``); only ``eigenvalues`` and ``modes`` columns are truncated.
    """
    if mode_labels is not None and n_modes is not None:
        raise ValueError("Pass at most one of mode_labels and n_modes.")
    if mode_labels is None and n_modes is None:
        return ModalResult(
            eigenvalues=np.asarray(result.eigenvalues, dtype=np.float64).copy(),
            modes=np.asarray(result.modes, dtype=np.float64).copy(),
            C=result.C,
            B_geom=result.B_geom,
            n_nodes=result.n_nodes,
            n_dof=result.n_dof,
            section=result.section,
            classification_eigenvalues=result.classification_eigenvalues,
            export_column_labels=result.export_column_labels,
        )

    n_all = len(result.eigenvalues)
    col_labels: tuple[str, ...] | None = None
    if mode_labels is not None:
        if result.section is None:
            raise ValueError("select_modes by mode_labels requires ModalResult.section to be set.")
        labels = list(mode_labels)
        want_all_distortion = "distortion" in labels
        pairs: list[tuple[int, str]] = []
        for k in range(n_all):
            lab = _export_label_for_mode(result, result.section, k)
            if lab in labels:
                pairs.append((k, lab))
            elif want_all_distortion and lab.startswith("distortion"):
                pairs.append((k, lab))
        if not pairs:
            raise ValueError(f"No modes matched labels {labels!r}.")
        idx = np.array([p[0] for p in pairs], dtype=np.int64)
        col_labels = tuple(p[1] for p in pairs)
    else:
        assert n_modes is not None
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1.")
        order = np.argsort(result.eigenvalues)
        idx = order[: min(n_modes, len(order))]

    return ModalResult(
        eigenvalues=result.eigenvalues[idx].copy(),
        modes=result.modes[:, idx].copy(),
        C=result.C,
        B_geom=result.B_geom,
        n_nodes=result.n_nodes,
        n_dof=result.n_dof,
        section=result.section,
        classification_eigenvalues=result.classification_eigenvalues,
        export_column_labels=col_labels,
    )


def truncation_report(result: "ModalResult", selected: "ModalResult") -> str:
    """Human-readable log of which original modes were kept vs dropped."""
    if result.section is None:
        return "truncation_report: ModalResult.section is unset; no per-mode export labels.\n"
    n_orig = len(result.eigenvalues)
    n_sel = len(selected.eigenvalues)
    lines = [
        f"GBT mode truncation: {n_sel} / {n_orig} modes retained.",
        "index  eigenvalue   export_label     retained",
        "-----  -----------  ---------------  --------",
    ]
    for k in range(n_orig):
        lam = float(result.eigenvalues[k])
        lab = _export_label_for_mode(result, result.section, k)
        kept = any(
            abs(lam - float(selected.eigenvalues[j])) < 1e-9 * max(abs(lam), 1.0)
            and np.linalg.norm(result.modes[:, k] - selected.modes[:, j]) < 1e-6 * max(np.linalg.norm(result.modes[:, k]), 1.0)
            for j in range(n_sel)
        )
        lines.append(f"{k:5d}  {lam:11.4e}  {lab:15s}  {str(kept)}")
    lines.append(f"summary: retained={n_sel}, dropped={n_orig - n_sel}")
    return "\n".join(lines) + "\n"


@dataclass
class ModalResult:
    eigenvalues: NDArray
    modes:       NDArray
    C:           NDArray
    B_geom:      NDArray
    n_nodes:     int
    n_dof:       int
    section:     CrossSection | None = None
    #: Full positive spectrum (pre-``n_modes`` trim) for stable :meth:`classify_mode` buckets.
    classification_eigenvalues: NDArray | None = None
    #: When built by :func:`select_modes` with ``mode_labels``, labels per retained column.
    export_column_labels: tuple[str, ...] | None = None

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
        ref = self.classification_eigenvalues
        pos = self.eigenvalues[self.eigenvalues > 1e-12] if ref is None else ref[ref > 1e-12]
        if len(pos) == 0:
            return "undetermined"
        med = float(np.median(pos))
        lam = self.eigenvalues[k]
        if   lam < 1e-10:       return "rigid_body"
        elif lam < 0.05 * med:  return "local"
        elif lam < 0.5  * med:  return "distortional"
        else:                   return "global"

    def classify_export_mode(self, k: int) -> str:
        """
        Physical/export label for spanwise classical stiffness (axial, bending_x/y, torsion,
        distortion_n, ...). Requires :attr:`section` when disambiguation needs geometry.
        """
        if self.export_column_labels is not None and 0 <= k < len(self.export_column_labels):
            return self.export_column_labels[k]
        if self.section is None:
            raise ValueError("classify_export_mode requires ModalResult.section to be set.")
        return _export_label_for_mode(self, self.section, k)

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
        ev_class = ev.copy()
        if n_modes is not None:
            m = min(n_modes, len(ev))
            ev = ev[:m]
            vv = vv[:, :m]
        return ModalResult(
            eigenvalues=ev,
            modes=vv,
            C=C,
            B_geom=B,
            n_nodes=self.section.n_nodes,
            n_dof=C.shape[0],
            section=self.section,
            classification_eigenvalues=ev_class,
            export_column_labels=None,
        )
