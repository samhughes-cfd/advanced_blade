"""
Local CLPT shell subcomponent: full [N; M] solve from :class:`ShellPanelResultants`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import ShellPanelResultants


def _stress_model_root() -> Path:
    return Path(__file__).resolve().parents[2] / "section_stress_model"


def _ensure_stress_imports() -> None:
    s = str(_stress_model_root())
    if s not in sys.path:
        sys.path.insert(0, s)


@dataclass
class StationCLPTShellResult:
    """Ply-level CLPT output at one station."""

    fi_tsai_wu: np.ndarray
    eps0: np.ndarray
    kappa: np.ndarray
    sig_lam_mid: list[np.ndarray]
    eps_lam_mid: list[np.ndarray]
    plies: Any
    N_vec: np.ndarray
    M_vec: np.ndarray


def solve_station_clpt_shell(
    resultants: ShellPanelResultants,
    plies: list,
    *,
    Xt: float,
    Xc: float,
    Yt: float,
    Yc: float,
    S12: float,
) -> StationCLPTShellResult:
    """
    Solve CLPT for ``[N; M] = [[A,B],[B,D]] [eps0; kappa]`` and Tsai–Wu per ply.

    Uses the same :func:`clpt_ply_failure_indices` path as the stress-model skin demo.
    """
    _ensure_stress_imports()
    from lib.laminate_clpt import (  # type: ignore[import-untyped]
        clpt_ply_failure_indices,
        ply_mid_strains,
    )

    N_vec = resultants.to_N_vec()
    M_vec = resultants.to_M_vec()

    fi_tw, eps0, kappa, sig_lam = clpt_ply_failure_indices(
        plies,
        N_vec,
        M_vec,
        Xt,
        Xc,
        Yt,
        Yc,
        S12,
    )
    eps_lam = ply_mid_strains(plies, eps0, kappa)

    return StationCLPTShellResult(
        fi_tsai_wu=fi_tw,
        eps0=eps0,
        kappa=kappa,
        sig_lam_mid=list(sig_lam),
        eps_lam_mid=list(eps_lam),
        plies=plies,
        N_vec=N_vec,
        M_vec=M_vec,
    )


def default_skin_strengths_pa() -> dict[str, float]:
    """Same representative strengths as ``SKIN_STRENGTH`` in multi_cell_blade_section."""
    return {
        "Xt": 600e6,
        "Xc": 500e6,
        "Yt": 50e6,
        "Yc": 140e6,
        "S12": 45e6,
    }
