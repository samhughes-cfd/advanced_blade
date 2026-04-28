"""
Discrete ply-orientation design variable (Group L).

An ``OrientationMix`` describes the half-stack ply composition for one subcomponent role.
The full symmetric balanced laminate is constructed as::

    half-stack: [0°]×n_0  [+45°,-45°]×n_biax  [90°]×n_90
    full stack:  half-stack  |  mirror(half-stack)

Constraints enforced by construction:
- **Symmetric** (B matrix = 0): full stack = half-stack mirrored about midplane.
- **Balanced** (A16 = A26 = 0): ±45 always added as pairs.
- **Allowlist** per role (from ``ply_angle_constraints.py``):
    skin → {0, ±45, 90}; web → {±45 only}; cap → {0 only}.

Practical consequence:
- *cap* has no free orientation choice (always 0°, n_biax = n_90 = 0).
- *web* has one free integer ``n_biax`` (always ±45, n_0 = n_90 = 0).
- *skin* is the only subcomponent with a genuine combinatorial space over (n_0, n_biax, n_90).

For N_half_skin in [2, 10], the feasible space has ≈ 50–100 combinations — trivially enumerable.

Outer-inner architecture (L.4)
-------------------------------
::

    for mix in enumerate_feasible_mixes(role, N_half_min, N_half_max):
        build LaminateDefinition.from_orientation_mix(mix, base_ply, t_total)
        run inner SLSQP over thicknesses
        record (mix, opt_cost, opt_dv)
    select global best mix + thickness

Per-material ply rounding (L.10)
---------------------------------
Each angle group uses its own base ``OrthotropicPly`` with its own ``t_ply``. The half-stack
thickness for a fixed mix is::

    t_half(mix, role) = n_0 * ply_UD.t_ply + 2 * n_biax * ply_biax.t_ply + n_90 * ply_90.t_ply

After inner SLSQP yields continuous ``t_role[i]``, snap to the nearest integer multiple::

    N_half[i] = round(t_role[i] / (2 * t_half(mix, role)))

Spanwise monotone post-rounding sweep (L.9)::

    for i in range(1, n_stations):
        N_half[role][i] = min(N_half[role][i], N_half[role][i - 1])

See ``BladeOptimizer.run_with_orientation`` for the full outer-inner orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    from blade_precompute.section_properties.engine.materials import OrthotropicPly

from ..core.types import OrientationBounds, ThicknessRole


@dataclass(frozen=True)
class OrientationMix:
    """Half-stack integer ply counts for one subcomponent role.

    ``N_half = n_0 + 2 * n_biax_pairs + n_90``  (half-stack ply count).
    Full symmetric balanced stack has ``2 * N_half`` plies.

    Allowlist per role is enforced by :func:`enumerate_feasible_mixes`; use
    :class:`OrientationMix` directly only when building from known-valid counts.
    """

    n_0: int
    n_biax_pairs: int
    n_90: int
    role: ThicknessRole

    @property
    def n_half(self) -> int:
        """Total number of plies in the half-stack."""
        return self.n_0 + 2 * self.n_biax_pairs + self.n_90

    @property
    def n_total(self) -> int:
        """Total number of plies in the full symmetric stack."""
        return 2 * self.n_half

    def fraction_0(self) -> float:
        """Fraction of 0° plies in the full stack."""
        n = self.n_total
        return float(2 * self.n_0) / n if n > 0 else 0.0

    def fraction_biax(self) -> float:
        """Fraction of ±45° plies in the full stack."""
        n = self.n_total
        return float(4 * self.n_biax_pairs) / n if n > 0 else 0.0

    def fraction_90(self) -> float:
        """Fraction of 90° plies in the full stack."""
        n = self.n_total
        return float(2 * self.n_90) / n if n > 0 else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "role": str(self.role),
            "n_0": int(self.n_0),
            "n_biax_pairs": int(self.n_biax_pairs),
            "n_90": int(self.n_90),
            "n_half": int(self.n_half),
            "n_total": int(self.n_total),
        }


def enumerate_feasible_mixes(
    role: ThicknessRole,
    n_half_min: int = 2,
    n_half_max: int = 10,
    n_biax_min: int = 1,
    n_0_min: int = 0,
    n_90_min: int = 0,
    *,
    bounds: OrientationBounds | None = None,
) -> Iterator[OrientationMix]:
    """Yield all feasible :class:`OrientationMix` for *role* within the given half-stack bounds.

    The per-role allowlist from ``ply_angle_constraints.py`` is enforced:

    - ``cap`` → only 0° plies (n_biax = n_90 = 0); single combination per N_half.
    - ``web`` → only ±45° pairs (n_0 = n_90 = 0); single combination per N_half.
    - ``skin`` → any mix of {0, ±45, 90} satisfying n_0 + 2*n_biax + n_90 = N_half.

    Parameters
    ----------
    role
        Subcomponent role (``"skin"``, ``"web"``, ``"cap"``).
    n_half_min, n_half_max
        Inclusive range of half-stack ply counts to enumerate.
    n_biax_min
        Minimum ±45 pairs per half-stack (default 1 ensures shear resistance).
    n_0_min, n_90_min
        Minimum 0° / 90° plies per half-stack.
    bounds
        If provided, overrides the individual scalar arguments above.
    """
    if bounds is not None:
        n_half_min = bounds.n_half_min
        n_half_max = bounds.n_half_max
        n_biax_min = bounds.n_biax_min
        n_0_min = bounds.n_0_min
        n_90_min = bounds.n_90_min

    for N in range(max(1, n_half_min), n_half_max + 1):
        if role == "cap":
            # Only 0° plies
            n_0 = N
            if n_0 >= n_0_min:
                yield OrientationMix(n_0=n_0, n_biax_pairs=0, n_90=0, role=role)
        elif role == "web":
            # Only ±45° pairs; N must be even (each pair = 2 plies in half-stack)
            if N % 2 != 0:
                continue
            n_biax = N // 2
            if n_biax >= n_biax_min:
                yield OrientationMix(n_0=0, n_biax_pairs=n_biax, n_90=0, role=role)
        elif role == "skin":
            # All combos: n_0 + 2*n_biax + n_90 = N, n_biax >= n_biax_min
            for nb in range(n_biax_min, N // 2 + 1):
                remain = N - 2 * nb
                for n0 in range(n_0_min, remain + 1):
                    n90 = remain - n0
                    if n90 >= n_90_min:
                        yield OrientationMix(n_0=n0, n_biax_pairs=nb, n_90=n90, role=role)
        # "fixed" role has no orientation design variable — skip


def t_half_for_mix(
    mix: OrientationMix,
    ply_ud: "OrthotropicPly",
    ply_biax: "OrthotropicPly",
    ply_90: "OrthotropicPly | None" = None,
) -> float:
    """Compute the half-stack thickness for a given ``OrientationMix`` and per-material plies.

    The half-stack thickness (L.10)::

        t_half = n_0 * ply_ud.t_ply
               + 2 * n_biax_pairs * ply_biax.t_ply
               + n_90 * (ply_90.t_ply if ply_90 else ply_biax.t_ply)

    The rounding step size for continuous-to-integer conversion is
    ``delta_t = 2 * t_half(mix, ...)`` (one full symmetric ply repeat).

    Parameters
    ----------
    mix
        Orientation mix for this subcomponent role.
    ply_ud
        0° UD ply (used for n_0 plies in the half-stack).
    ply_biax
        ±45° biax ply (used for each ±45 pair in the half-stack).
    ply_90
        90° ply (uses ply_biax when None — common for woven biax that spans ±45 and 90).
    """
    t_90 = ply_90.t_ply if ply_90 is not None else ply_biax.t_ply
    return (
        float(mix.n_0) * float(ply_ud.t_ply)
        + 2.0 * float(mix.n_biax_pairs) * float(ply_biax.t_ply)
        + float(mix.n_90) * t_90
    )


def snap_to_integer_plies(
    t_continuous: Sequence[float],
    mix: OrientationMix,
    ply_ud: "OrthotropicPly",
    ply_biax: "OrthotropicPly",
    ply_90: "OrthotropicPly | None" = None,
    *,
    n_half_min: int = 1,
    n_half_max: int = 100,
    enforce_monotone: bool = True,
) -> list[int]:
    """Round a continuous thickness profile to integer N_half per station.

    After rounding, apply the spanwise monotone sweep (L.9) when
    ``enforce_monotone`` is True: ``N_half[i] = min(N_half[i], N_half[i-1])``.

    Returns
    -------
    list of int
        ``N_half[i]`` per station (half-stack count).
    """
    import math

    t_half = t_half_for_mix(mix, ply_ud, ply_biax, ply_90)
    step = 2.0 * t_half  # one full symmetric repeat
    if step <= 0.0:
        step = 1e-6

    n_half_out: list[int] = []
    for t in t_continuous:
        n_raw = int(round(float(t) / step))
        n_clamped = max(n_half_min, min(n_half_max, n_raw))
        n_half_out.append(n_clamped)

    if enforce_monotone:
        for i in range(1, len(n_half_out)):
            n_half_out[i] = min(n_half_out[i], n_half_out[i - 1])

    return n_half_out
