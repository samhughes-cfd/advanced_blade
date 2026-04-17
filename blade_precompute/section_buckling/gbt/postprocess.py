"""
postprocess.py — Visualisation and export for GBT analysis results.

Functions
---------
plot_cross_section(section)
    Draw wall geometry, node numbering, and wall labels.

plot_mode_shape(section, modal_result, mode_index)
    Visualise a cross-section deformation mode as an exaggerated displaced shape.

plot_signature_curve(sig_dict)
    Plot lambda_cr vs half-wavelength (GBT signature curve).

plot_member_mode(member_result, n_elem, length)
    Plot the longitudinal buckled shape V_k(x) for the critical member mode.

export_results(modal_result, member_result, filepath)
    Export key results to a JSON file.

export_csv(modal_result, filepath)
    Write per-mode eigenvalues and rigidities to CSV.
"""
from __future__ import annotations
import json
import csv
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False


def _check_mpl():
    if not _MPL:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


# ---------------------------------------------------------------------------
# Cross-section geometry plot
# ---------------------------------------------------------------------------

def plot_cross_section(section, ax=None, show=True, title="Cross-Section Geometry"):
    """
    Draw wall geometry with node markers and wall labels.

    Parameters
    ----------
    section : CrossSection
    ax      : matplotlib Axes (optional — created if None)
    show    : bool  Call plt.show() if True.
    title   : str

    Returns
    -------
    fig, ax
    """
    _check_mpl()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 6))

    coords = section.node_coordinates
    # Draw strips
    for s in section._strips:
        ni = section.get_node(s.node_i)
        nj = section.get_node(s.node_j)
        ax.plot([ni.y, nj.y], [ni.z, nj.z], 'b-', lw=2.0, zorder=2)

    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1],
               c='red', s=40, zorder=5, label='Nodes')

    # Label shared junction nodes (wall_ids count > 1)
    for n in section._nodes:
        if len(n.wall_ids) > 1:
            ax.annotate(f'{n.node_id}',
                        xy=(n.y, n.z), xytext=(4, 4),
                        textcoords='offset points', fontsize=7, color='darkred')

    # Label wall midpoints
    for i, wall in enumerate(section.walls):
        mid = 0.5 * (wall.node_start + wall.node_end)
        ax.text(mid[0], mid[1], wall.name or f'W{i}',
                fontsize=8, ha='center', va='bottom', color='navy', style='italic')

    ax.set_xlabel('y [m]'); ax.set_ylabel('z [m]')
    ax.set_title(title); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Mode shape plot
# ---------------------------------------------------------------------------

def plot_mode_shape(section, modal_result, mode_index=0,
                   scale=None, ax=None, show=True):
    """
    Visualise a cross-section deformation mode as an exaggerated displaced shape.

    The out-of-plane (w) displacement DOFs are extracted per node and overlaid
    on the undeformed section in the normal-to-wall direction.

    Parameters
    ----------
    section      : CrossSection
    modal_result : ModalResult
    mode_index   : int
    scale        : float or None  Exaggeration scale (auto if None).
    """
    _check_mpl()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 6))

    ndpn  = modal_result.n_dof // modal_result.n_nodes
    phi   = modal_result.modes[:, mode_index]
    coords = section.node_coordinates

    # Extract w-DOFs (index 2 in each node's DOF block for Kirchhoff: u,v,w,theta)
    # For 4 DOFs/node: indices 2 per node
    w_dof_local = 2 if ndpn >= 3 else 0
    w_disp = np.array([phi[i * ndpn + w_dof_local] for i in range(section.n_nodes)])

    max_w = np.abs(w_disp).max()
    if max_w < 1e-15:
        max_w = 1.0
    if scale is None:
        ref_length = section.walls[0].length if section.walls else 0.1
        scale = 0.15 * ref_length / max_w

    # Plot undeformed (light)
    for s in section._strips:
        ni = section.get_node(s.node_i)
        nj = section.get_node(s.node_j)
        ax.plot([ni.y, nj.y], [ni.z, nj.z], 'lightblue', lw=1.5, zorder=1)

    # Plot deformed (displaced in wall normal direction)
    for i, wall in enumerate(section.walls):
        n_vec = wall.normal
        y_def = []; z_def = []
        for s in section._strips:
            if s.wall_id != i:
                continue
            for nid in (s.node_i, s.node_j):
                nd = section.get_node(nid)
                w  = w_disp[nid] * scale
                y_def.append(nd.y + w * n_vec[0])
                z_def.append(nd.z + w * n_vec[1])
        if y_def:
            ax.plot(y_def, z_def, 'r-', lw=2.0, zorder=3)

    lam = modal_result.eigenvalues[mode_index]
    cls = modal_result.classify_mode(mode_index)
    ax.set_title(f'Mode {mode_index}  (λ={lam:.3e}, {cls})')
    ax.set_xlabel('y [m]'); ax.set_ylabel('z [m]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color='lightblue', label='Undeformed'),
               mpatches.Patch(color='red',       label='Deformed')]
    ax.legend(handles=handles, fontsize=8)

    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Signature curve
# ---------------------------------------------------------------------------

def plot_signature_curve(sig_dict, ax=None, show=True, title="GBT Signature Curve"):
    """
    Plot lambda_cr vs. half-wavelength.

    Parameters
    ----------
    sig_dict : dict  Output of MemberBucklingAnalysis.signature_curve()
    """
    _check_mpl()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 5))

    hw  = sig_dict['half_wave_lengths']
    lam = sig_dict['lambda_cr']
    valid = ~np.isnan(lam)

    ax.semilogx(hw[valid], lam[valid], 'b-o', ms=5, lw=1.8)
    idx_min = np.nanargmin(lam)
    ax.axvline(hw[idx_min], color='red', ls='--', lw=1.2,
               label=f'Min λ={lam[idx_min]:.3e} at L_hw={hw[idx_min]:.4f} m')

    ax.set_xlabel('Half-wavelength [m]')
    ax.set_ylabel('Critical load multiplier λ_cr')
    ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)

    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Member mode shape
# ---------------------------------------------------------------------------

def plot_member_mode(member_result, ax=None, show=True):
    """
    Plot the longitudinal buckled displacement amplitude V(x) for the dominant
    cross-section mode in the critical member buckling mode.
    """
    _check_mpl()
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4))

    n_elem  = member_result.n_elem
    n_modes = member_result.n_modes
    L       = member_result.member_length
    n_nodes = n_elem + 1
    x       = np.linspace(0, L, n_nodes)

    full = member_result.buckling_mode
    n_dof_per_mode = 2 * n_nodes

    # Find dominant mode (largest L2 norm of displacement DOFs)
    norms = np.zeros(n_modes)
    for k in range(n_modes):
        start = k * n_dof_per_mode
        w_dofs = full[start:start + n_dof_per_mode:2]   # every other = displacement
        norms[k] = np.linalg.norm(w_dofs)

    dom = int(np.argmax(norms))
    start = dom * n_dof_per_mode
    w_dofs = full[start:start + n_dof_per_mode:2]   # n_nodes values

    ax.plot(x, w_dofs, 'b-o', ms=4, lw=2)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('Amplitude V(x)')
    ax.set_title(f'Critical member mode — dominant cross-section mode {dom}')
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout(); plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(modal_result, member_result, filepath: str):
    """
    Export key analysis results to a JSON file.

    Parameters
    ----------
    modal_result  : ModalResult
    member_result : MemberBucklingResult
    filepath      : str  Output path (e.g. 'results.json')
    """
    data = {
        "cross_section_modes": {
            "n_modes":      len(modal_result.eigenvalues),
            "eigenvalues":  modal_result.eigenvalues.tolist(),
            "modal_rigidities": [modal_result.modal_rigidity(k)
                                  for k in range(len(modal_result.eigenvalues))],
            "classifications": [modal_result.classify_mode(k)
                                 for k in range(len(modal_result.eigenvalues))],
        },
        "member_buckling": {
            "lambda_cr":     member_result.lambda_cr,
            "n_elem":        member_result.n_elem,
            "n_modes":       member_result.n_modes,
            "member_length": member_result.member_length,
            "n_half_waves":  member_result.n_half_waves(),
            "eigenvalues":   member_result.eigenvalues[:10].tolist(),
        },
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results exported to {filepath}")


def export_csv(modal_result, filepath: str):
    """
    Write per-mode eigenvalues, modal rigidities, and classifications to CSV.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'eigenvalue', 'modal_rigidity',
                         'modal_geom_stiffness', 'classification'])
        for k in range(len(modal_result.eigenvalues)):
            writer.writerow([
                k,
                modal_result.eigenvalues[k],
                modal_result.modal_rigidity(k),
                modal_result.modal_geometric_stiffness(k),
                modal_result.classify_mode(k),
            ])
    print(f"CSV exported to {filepath}")
