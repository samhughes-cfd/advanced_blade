"""Discrete ply-angle allowlists per structural role (0 deg = fibre / ply E1 direction)."""

from __future__ import annotations

from collections.abc import Iterable

from blade_precompute.section_optimisation.core.types import ThicknessRole

# Optimiser / laminate builder must only use angles from these sets per role.
PLY_ANGLE_ALLOWLIST_SKIN_DEG: frozenset[float] = frozenset({0.0, 90.0, 45.0, -45.0})
PLY_ANGLE_ALLOWLIST_WEB_DEG: frozenset[float] = frozenset({45.0, -45.0})
PLY_ANGLE_ALLOWLIST_SPAR_CAP_DEG: frozenset[float] = frozenset({0.0})


def allowlist_for_role(role: ThicknessRole) -> frozenset[float]:
    if role == "skin":
        return PLY_ANGLE_ALLOWLIST_SKIN_DEG
    if role == "web":
        return PLY_ANGLE_ALLOWLIST_WEB_DEG
    if role == "cap":
        return PLY_ANGLE_ALLOWLIST_SPAR_CAP_DEG
    return frozenset()  # "fixed" — no ply-angle product constraint here


def validate_stack_angles_for_role(role: ThicknessRole, angles_deg: Iterable[float], *, subcomponent: str = "") -> None:
    """Raise ``ValueError`` if any ply angle is not allowed for ``role`` (skin / web / cap)."""
    allow = allowlist_for_role(role)
    if not allow:
        return
    label = f" ({subcomponent})" if subcomponent else ""
    for a in angles_deg:
        af = float(a)
        if not any(abs(af - x) <= 1e-9 for x in allow):
            raise ValueError(
                f"Ply angle {af} deg not in allowlist for role {role!r}{label}: {sorted(allow)}."
            )


def composite_thickness_m(*, n_plies: int, t_ply_m: float) -> float:
    """Total solid laminate thickness ``t = n * t_ply`` (optimiser-owned ``n``)."""
    return float(n_plies) * float(t_ply_m)
