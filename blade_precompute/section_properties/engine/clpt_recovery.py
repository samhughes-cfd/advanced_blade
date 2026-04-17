"""
Tier 2 CLPT ply stress recovery in the strip (section) frame, then ply material frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clpt_ply_stresses_section_frame(
    sub_resultants: NDArray[np.float64],
    ABD_inv: NDArray[np.float64],
    Q_bar: NDArray[np.float64],
    z_ply: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Ply-level stresses in **laminate / section** axes (1=x beam, 2=tangent).

    Parameters
    ----------
    sub_resultants
        ``(..., n_comp, 6)`` = ``[N11, N22, N12, M11, M22, M12]``.
    ABD_inv
        ``(n_comp, 6, 6)`` or ``(..., n_comp, 6, 6)`` with optional leading batch
        (e.g. spanwise station index).
    Q_bar
        ``(n_comp, n_ply, 3, 3)`` or ``(..., n_comp, n_ply, 3, 3)``.
    z_ply
        ``(n_comp, n_ply)`` or ``(..., n_comp, n_ply)`` ply mid-surface z [m].

    Returns
    -------
    sigma_sec
        ``(..., n_comp, n_ply, 3)`` = ``[σ11, σ22, τ12]`` [Pa].
    """
    strain6 = np.einsum("...cij,...cj->...ci", ABD_inv, sub_resultants)
    eps0 = strain6[..., :3]
    kap = strain6[..., 3:6]
    eps_k = eps0[..., :, None, :] + z_ply[..., :, None] * kap[..., :, None, :]
    return np.einsum("...cpjk,...cpj->...cpk", Q_bar, eps_k)


def rotate_plies_to_material(
    sigma_sec: NDArray[np.float64],
    T_ply: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map section-frame ply stress toward material frame (engineering ``T_ply``)."""
    return np.einsum("...cpij,...cpj->...cpi", T_ply, sigma_sec)
