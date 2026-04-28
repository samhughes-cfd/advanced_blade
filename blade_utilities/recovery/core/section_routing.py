"""Subcomponent routing and ply metadata shared by tensor-cache and operator builders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from blade_precompute.section_properties.engine.geometry import SubcomponentGeometry
from blade_precompute.section_properties.engine.laminate import LaminateDefinition


def composite_and_isotropic_indices(
    section0_subcomponents: Sequence[SubcomponentGeometry],
) -> tuple[list[int], list[int]]:
    """Return lists of subcomponent indices for composite and isotropic members."""
    comp: list[int] = []
    iso: list[int] = []
    for i, sub in enumerate(section0_subcomponents):
        if sub.is_composite:
            comp.append(i)
        else:
            iso.append(i)
    return comp, iso


def ply_count_row(
    section0_subcomponents: Sequence[SubcomponentGeometry],
    comp_idx: list[int],
    n_s: int,
) -> NDArray[np.int32]:
    """``(n_s, n_comp)`` ply counts per composite subcomponent."""
    row = np.zeros((1, len(comp_idx)), dtype=np.int32)
    for p, gi in enumerate(comp_idx):
        sub = section0_subcomponents[gi]
        assert isinstance(sub.material, LaminateDefinition)
        row[0, p] = int(len(sub.material.plies))
    return np.tile(row, (n_s, 1))


def ply_strength_pad(
    section0_subcomponents: Sequence[SubcomponentGeometry],
    comp_idx: list[int],
    n_ply_max: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Ply strength grids ``(n_comp, n_ply_max)`` [Pa] for Hashin / allowables from reference section plies."""
    xt = np.zeros((len(comp_idx), n_ply_max), dtype=np.float64)
    xc = np.zeros_like(xt)
    yt = np.zeros_like(xt)
    yc = np.zeros_like(xt)
    s12 = np.zeros_like(xt)
    for row, gi in enumerate(comp_idx):
        sub = section0_subcomponents[gi]
        assert isinstance(sub.material, LaminateDefinition)
        lam = sub.material
        for k, (ply, _) in enumerate(lam.plies):
            xt[row, k] = ply.Xt
            xc[row, k] = ply.Xc
            yt[row, k] = ply.Yt
            yc[row, k] = ply.Yc
            s12[row, k] = ply.S12
    return xt, xc, yt, yc, s12
