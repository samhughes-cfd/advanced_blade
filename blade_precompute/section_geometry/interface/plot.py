"""
interface.plot
==============
Matplotlib-based visualisation helpers for section geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Colour palette for subcomponents
_COMPONENT_COLORS = {
    "outer_skin":       "#2196F3",   # blue
    "spar_cap_upper":   "#F44336",   # red
    "spar_cap_lower":   "#FF9800",   # orange
    "web_0":            "#4CAF50",   # green
    "web_1":            "#009688",   # teal
    "web_2":            "#8BC34A",   # light green
    "core_0":           "#9C27B0",   # purple
    "core_fore":        "#AB47BC",   # purple-ish
    "core_aft":         "#7B1FA2",   # deep purple
    "te_insert":        "#795548",   # brown
    "le_insert":        "#3F51B5",   # indigo
}
_DEFAULT_COLOR = "#607D8B"
_WEB_COLORS = ["#4CAF50", "#009688", "#8BC34A", "#2E7D32"]
_CORE_COLORS = ["#9C27B0", "#AB47BC", "#7B1FA2", "#6A1B9A"]


def _component_color(label):
    if label in _COMPONENT_COLORS:
        return _COMPONENT_COLORS[label]
    if label.startswith("web_"):
        try:
            idx = int(label.split("_", 1)[1])
            return _WEB_COLORS[idx % len(_WEB_COLORS)]
        except (ValueError, IndexError):
            return _WEB_COLORS[0]
    if label.startswith("core_"):
        suffix = label.split("_", 1)[1]
        try:
            idx = int(suffix)
            return _CORE_COLORS[idx % len(_CORE_COLORS)]
        except ValueError:
            # Named core variants (e.g. core_fore/core_aft) are handled here.
            return _CORE_COLORS[0]
    return _COMPONENT_COLORS.get(label, _DEFAULT_COLOR)


def plot_sdf_field(phi, grid, ax=None, title="SDF field",
                   colorbar=True, contour_zero=True,
                   cmap="RdBu_r", vmax=None):
    """Plot a raw SDF field as a filled contour / imshow.

    Parameters
    ----------
    phi : ndarray, shape (ny, nx)
    grid : SDFGrid
    ax : matplotlib Axes or None
    title : str
    colorbar : bool
    contour_zero : bool
        Overlay the zero-level-set as a solid black line.
    cmap : str
    vmax : float or None
        Colour scale symmetric about zero; defaults to max(|phi|).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if vmax is None:
        vmax = float(np.abs(phi).max())

    im = ax.pcolormesh(grid.X, grid.Y, phi,
                       cmap=cmap, vmin=-vmax, vmax=vmax,
                       shading="auto")
    if colorbar:
        fig.colorbar(im, ax=ax, label="φ (m)")

    if contour_zero:
        ax.contour(grid.X, grid.Y, phi, levels=[0.0],
                   colors="k", linewidths=1.5)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_section(section_geometry, grid, ax=None, alpha=0.45,
                 show_airfoil=True, title="Blade section geometry"):
    """Plot all subcomponents as filled colour patches on one axis.

    Parameters
    ----------
    section_geometry : BladeSectionGeometry
    grid : SDFGrid
    ax : matplotlib Axes or None
    alpha : float
        Fill transparency.
    show_airfoil : bool
        Overlay the airfoil boundary contour.
    title : str

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    for label in section_geometry:
        phi   = grid.eval(section_geometry[label])
        color = _component_color(label)
        # Filled region where phi < 0
        ax.contourf(grid.X, grid.Y, phi,
                    levels=[-np.inf, 0.0],
                    colors=[color], alpha=alpha)
        # Boundary contour
        ax.contour(grid.X, grid.Y, phi,
                   levels=[0.0], colors=[color], linewidths=1.0)

    if show_airfoil:
        phi_af = grid.eval(section_geometry.airfoil)
        ax.contour(grid.X, grid.Y, phi_af,
                   levels=[0.0], colors=["k"], linewidths=2.0)

    # Legend patches
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=_component_color(lbl), alpha=alpha, label=lbl)
        for lbl in section_geometry
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    return fig, ax


def plot_medial_axes(midline_dict, ax=None, grid=None,
                     section_geometry=None, alpha_bg=0.2,
                     title="Medial axes"):
    """Plot medial axis polylines, optionally on top of section geometry.

    Parameters
    ----------
    midline_dict : dict  {label: list of (N, 2) ndarray}
    ax : matplotlib Axes or None
    grid, section_geometry : optional
        If both provided, draw section geometry in the background.
    alpha_bg : float
        Background section alpha.
    title : str

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    if grid is not None and section_geometry is not None:
        plot_section(section_geometry, grid, ax=ax,
                     alpha=alpha_bg, show_airfoil=True,
                     title=title)

    for label, polylines in midline_dict.items():
        color = _component_color(label)
        for poly in polylines:
            if len(poly) < 2:
                continue
            ax.plot(poly[:, 0], poly[:, 1],
                    color=color, linewidth=2.5,
                    label=f"{label} (midline)")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    return fig, ax


def plot_grad_magnitude(phi, grid, ax=None, title="|∇φ|  (Eikonal residual)"):
    """Visualise |∇φ| to diagnose medial-axis locus and SDF quality.

    Parameters
    ----------
    phi : ndarray, shape (ny, nx)
    grid : SDFGrid

    Returns
    -------
    fig, ax
    """
    gx = np.gradient(phi, grid.dx, axis=1)
    gy = np.gradient(phi, grid.dy, axis=0)
    gm = np.sqrt(gx**2 + gy**2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    im = ax.pcolormesh(grid.X, grid.Y, gm,
                       cmap="plasma", vmin=0.0, vmax=1.5,
                       shading="auto")
    fig.colorbar(im, ax=ax, label="|∇φ|")
    # Overlay interior boundary
    ax.contour(grid.X, grid.Y, phi, levels=[0.0],
               colors="white", linewidths=1.5, linestyles="--")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax
