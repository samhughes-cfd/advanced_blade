"""
Shell handoff DTOs: local panel resultants + per-field provenance for MVP auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .section_vlasov import SectionVlasovResult


class ProvenanceKind(str, Enum):
    """How a resultant value was obtained."""

    DERIVED = "derived"          # from section recovery (thin-wall mapping)
    PLACEHOLDER = "placeholder"  # not yet recovered; set to explicit default
    RESERVED = "reserved"        # field reserved for future FSDT / higher-order recovery
    MITC4 = "mitc4"              # recovered from MITC4 panel shell solve


@dataclass
class FieldProvenance:
    """Audit trail for one scalar resultant."""

    kind: ProvenanceKind
    note: str = ""


@dataclass
class ShellPanelResultants:
    """
    Laminate mid-surface shell resultants [N/m] and moments [N·m/m] in strip axes.

    Convention (matches :func:`examples.section_stress_model.lib.laminate_clpt.membrane_resultants_from_shell_stress`):
    index 0 = x (span / beam axis), 1 = y (contour / tangent in section plane), 2 = in-plane shear xy.

    MVP mapping from thin-wall recovery:
    - ``Nx``, ``Nxy`` derived from ``sigma_xx * t`` and ``q`` (shear flow).
    - ``Ny``, ``Mx``, ``My``, ``Mxy`` placeholders (zeros) until shell curvature / Ny recovery exists.
    """

    Nx: float
    Ny: float
    Nxy: float
    Mx: float
    My: float
    Mxy: float
    Qx: float | None = None  # optional transverse shear resultants [N/m] (FSDT)
    Qy: float | None = None

    provenance: dict[str, FieldProvenance] = field(default_factory=dict)

    # Raw thin-wall diagnostics (SI)
    sigma_xx_pa: float = 0.0
    tau_xy_pa: float = 0.0
    q_n_per_m: float = 0.0
    thickness_m: float = 0.0

    panel_label: str = ""
    panel_index: int = 0
    station_index: int = 0

    def to_N_vec(self) -> np.ndarray:
        """[Nx, Ny, Nxy] for CLPT assembly."""
        return np.array([self.Nx, self.Ny, self.Nxy], dtype=float)

    def to_M_vec(self) -> np.ndarray:
        """[Mx, My, Mxy] for CLPT assembly."""
        return np.array([self.Mx, self.My, self.Mxy], dtype=float)


@dataclass
class SectionShellRecoveryBundle:
    """Section recovery outputs plus shell-mapped stations (adapter output)."""

    panels: Any
    booms: Any
    webs_geom: Any
    q_tot: Any
    sig_p: Any
    sig_b: Any
    q0: Any
    props: Any
    y_sc: float
    z_sc: float
    areas: list[float]
    I_omega: float
    gamma_y: float
    gamma_z: float
    GA_y: float
    GA_z: float
    q_primary: Any
    q_warp: Any
    # Shell handoff for one reference station (optional; filled by adapter helper)
    reference_resultants: ShellPanelResultants | None = None
    # Per-panel MITC4 element resultants (list[list[ShellPanelResultants]])
    all_panel_mitc4_results: list | None = None
    # Per-panel MITC4 diagnostics (residual/reaction/load balance details)
    all_panel_mitc4_diagnostics: list | None = None
    # Vlasov warping result (set by run_section_with_mitc4_shell / run_section_both)
    vlasov_result: SectionVlasovResult | None = None
