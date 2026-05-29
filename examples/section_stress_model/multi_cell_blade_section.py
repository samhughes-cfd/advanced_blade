"""
multi_cell_blade_section.py
============================
Structural visualisation for multi-cell thin-walled blade cross-sections
with explicit boom/panel idealisation.

Three distinct laminates:
  - Skin panels  : ±45 GFRP  — carries shear flow q(s), minimal bending
  - Spar caps    : UD CFRP   — carries axial bending stress σ (boom idealisation)
  - Shear webs   : ±45 GFRP  — carries shear flow q(s)

Shear flow junction condition at each spar cap:
  - Cap carries NO shear flow — discrete ΔQy, ΔQz jumps on the open-section integrals
  - Web shear flow = skin shear flow at the junction (equilibrium satisfied)
  - Multi-cell system: shear closing flows + optional St. Venant torsion (unit T) via augmented Bredt solve
  - Shear centre (y_sc, z_sc) from flexural shear-flow torque balance (unit Vy, Vz, T=0).

Physics modules (``lib/``):
  - lib.laminate_clpt — orthotropic ABD, ply stresses, transverse shear stiffness
  - lib.sectorial_warping — ω, shear centre, I_ω
  - lib.timoshenko_section — GA_y, GA_z, shear strains
  - lib.vlasov_thinwall — bimoment B → σ_ω
  - lib.warping_shear — secondary shear flow from dB/dx (non-uniform torsion)

See STRESS_MODEL.md for assumptions and equations.

Usage:
    python multi_cell_blade_section.py

Writes ``outputs/blade_section_distributions.png`` (q + sigma), ``outputs/blade_section_shear_flow.png`` (q only),
``outputs/blade_section_bending_stress.png`` (sigma only), and ``outputs/blade_section_clpt_fi.png``
(CLT ply stresses/strains with Hashin-envelope FI at one skin station), next to this script.

Requirements: numpy, matplotlib
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field

from lib.laminate_clpt import (
    Ply,
    abd_stack,
    clpt_ply_failure_indices,
    homogenized_axial_modulus,
    membrane_resultants_from_shell_stress,
    ply_mid_strains,
    ply_stresses_bottom_top,
    stress_laminate_to_material,
    default_rectangular_plies,
)
from lib.sectorial_warping import (
    normalized_warping,
    open_outline_from_airfoil,
    warping_constant_I_omega,
)
from lib.timoshenko_section import global_shear_stiffness_from_panels, timoshenko_shear_strains
from lib.vlasov_thinwall import sigma_from_bimoment
from lib.warping_shear import (
    q_omega_secondary_open_vertices,
    q_omega_secondary_panels_particular,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION PROPERTY BUNDLE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionProps:
    """Modulus-weighted centroid, axial stiffness, and second moments (SI)."""
    y_c: float
    z_c: float
    EA: float
    Iyy: float
    Izz: float
    Iyz: float


# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL / LAMINATE PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

E_REF = 40e9   # reference modulus for normalisation [Pa]
G_REF = E_REF / 2.6  # reference shear modulus for thin-wall torsion closure [Pa]

# Section resultants [N, Vy, Vz, Mz, My, T] — SI: N, V [N]; M, T [N·m]
SECTION_RESULTANTS = np.ones(6, dtype=float)

# Bimoment for Vlasov warping stress [N·m²] (prescribed at section; spanwise B(x) not solved)
B_BIMOMENT = 0.0

@dataclass
class Laminate:
    """Wall laminate with axial modulus and thickness; optional ν for default ply stack."""
    E: float      # Young's modulus [Pa]
    t: float      # wall thickness [m] (total laminate thickness = n_plies · t_ply when equal gauge)
    name: str = ""
    nu: float = 0.35
    n_plies: int = 4

    def build_plies(self) -> list[Ply]:
        """Representative [0/90/90/0] isotropic stack for CLPT / Timoshenko."""
        return default_rectangular_plies(self.E, self.nu, self.t, n=self.n_plies)


def skin_laminate(
    E: float,
    nu: float,
    t_ply: float,
    n_plies: int,
    *,
    name: str = "",
) -> Laminate:
    """
    Skin laminate with fixed prepreg gauge ``t_ply`` and ``n_plies`` plies.

    Total thickness is ``n_plies * t_ply``; each ply thickness is ``t_ply``.
    """
    t_tot = t_ply * n_plies
    return Laminate(E=E, t=t_tot, name=name, nu=nu, n_plies=n_plies)


# Representative blade laminates — adjust to your actual layup
SKIN_LAM = Laminate(E=20e9, t=0.006, name="Skin ±45 GFRP", n_plies=4)
CAP_LAM = Laminate(E=120e9, t=0.020, name="Spar cap UD CFRP")
WEB_LAM = Laminate(E=12e9, t=0.010, name="Shear web ±45 GFRP")

# Skin prepreg gauge and density (for ply-count / areal-mass reporting)
T_PLY_SKIN = SKIN_LAM.t / SKIN_LAM.n_plies
RHO_SKIN = 1900.0  # [kg/m³] typical GFRP — areal mass = RHO_SKIN * t_skin [kg/m²]

# Representative ply strengths [Pa] for CLPT failure envelopes (tune to your prepreg data)
SKIN_STRENGTH = dict(
    Xt=600e6,
    Xc=500e6,
    Yt=50e6,
    Yc=140e6,
    S12=45e6,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoomNode:
    """
    Lumped spar cap at a web/skin junction (boom idealisation).

    The cap carries axial bending stress only. No shear flow passes through it.
    Its contribution to the Q integrals is a discrete jump in both Qy and Qz:
        ΔQy = A_eff * (y_cap - y_centroid),  ΔQz = A_eff * (z_cap - z_centroid)
    where A_eff = A_cap * E_cap / E_ref (modulus-weighted area).
    """
    y: float          # chord position [m]
    z: float          # thickness position [m]
    A_cap: float      # cap cross-sectional area [m²]
    lam: Laminate = field(default_factory=lambda: CAP_LAM)
    label: str = ""

    @property
    def A_eff(self):
        """Modulus-weighted area."""
        return self.A_cap * self.lam.E / E_REF


@dataclass
class Panel:
    """
    Thin-walled panel between two nodes.

    Parameters
    ----------
    nodes    : (N,2) array  — [y, z] coordinates along the panel
    lam      : Laminate     — panel material (E, t)
    cell_id  : int          — which closed cell this panel belongs to
    end_boom : BoomNode     — boom at the end of this panel (adds ΔQ jump)
    label    : str
    """
    nodes: np.ndarray
    lam: Laminate
    cell_id: int = 0
    end_boom: object = None
    label: str = ""

    def __post_init__(self):
        ds = np.linalg.norm(np.diff(self.nodes, axis=0), axis=1)
        self.s = np.concatenate([[0], np.cumsum(ds)])
        self.E_n = self.lam.E / E_REF   # normalised modulus


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _pin_le_te_chord(x: np.ndarray) -> np.ndarray:
    """Force ``x/c = 0`` at index 0 and ``x/c = 1`` at index -1 (exact LE/TE)."""
    out = np.asarray(x, dtype=float).copy()
    out[0] = 0.0
    out[-1] = 1.0
    return out


def chordwise_stations(n: int, *, spacing: str = "nested_cosine") -> np.ndarray:
    """
    Chord-normalised stations ``x/c`` in ``[0, 1]``.

    For **each** of the ``n`` stations, the first is **exactly** the leading edge
    (``x/c = 0``) and the last **exactly** the trailing edge (``x/c = 1``). The
    same vector is used for **upper and lower** skin rows in :func:`naca_four_digit`
    / :func:`naca_symmetric`.

    ``cosine``
        Single half-cosine map (:math:`t \\in [0,\\pi]` uniform, :math:`x/c=(1-\\cos t)/2`).
        Good LE/TE clustering; lightest of the cosine family below.

    ``nested_cosine`` (default)
        Apply that half-cosine map **twice** (parameter → :math:`[0,1]` → chord).
        **Stronger** LE/TE concentration than ``cosine``.

    ``nested_cosine_3``
        Half-cosine composed **three** times — **strongest** LE/TE clustering (most
        nodes near the ends, fewest mid-chord). Use with moderate ``n`` to avoid an
        overly sparse mid-panel.

    ``le_dense``
        :math:`x = \\sin(\\pi u / 2)` for :math:`u \\in [0,1]` — LE emphasis; TE
        still has exactly one station at ``x/c = 1``.

    ``uniform``
        Equispaced ``x/c`` (legacy).
    """
    if n < 2:
        raise ValueError("chordwise_stations requires n >= 2")
    if spacing == "uniform":
        return np.linspace(0.0, 1.0, n)
    if spacing == "cosine":
        # t uniform on [0, pi]; x/c = (1-cos t)/2 → densest chordwise near t=0,pi (LE/TE)
        t = np.linspace(0.0, np.pi, n)
        x = 0.5 * (1.0 - np.cos(t))
        return _pin_le_te_chord(x)
    if spacing == "nested_cosine":
        u = np.linspace(0.0, np.pi, n)
        v = 0.5 * (1.0 - np.cos(u))
        x = 0.5 * (1.0 - np.cos(np.pi * v))
        return _pin_le_te_chord(x)
    if spacing == "nested_cosine_3":
        u = np.linspace(0.0, np.pi, n)
        v = 0.5 * (1.0 - np.cos(u))
        w = 0.5 * (1.0 - np.cos(np.pi * v))
        x = 0.5 * (1.0 - np.cos(np.pi * w))
        return _pin_le_te_chord(x)
    if spacing == "le_dense":
        u = np.linspace(0.0, 1.0, n)
        x = np.sin(0.5 * np.pi * u)
        return _pin_le_te_chord(x)
    raise ValueError(
        f"unknown chord spacing {spacing!r}; use 'cosine', 'nested_cosine', "
        f"'nested_cosine_3', 'le_dense', or 'uniform'"
    )


def naca_symmetric(t_c=0.18, n=120, *, chord_spacing: str = "nested_cosine"):
    """
    NACA 00xx symmetric airfoil.

    Returns ``(2n, 2)`` array: **upper** surface LE→TE, then **lower** LE→TE using
    the **same** ``n`` cosine (or chosen) chord stations — **one** node at ``x/c=0``
    and **one** at ``x/c=1`` per surface. Trailing edge is a **sharp** point ``(1, 0)``; both
    surfaces meet at the LE ``(0, 0)``. This matches ``sectorial_warping``'s
    ``len(airfoil)//2`` split and removes the blunt open TE from the analytic
    thickness at ``x/c = 1``. Default ``chord_spacing`` is ``nested_cosine``; use
    ``nested_cosine_3`` for even denser LE/TE (see :func:`chordwise_stations`).
    """
    x = chordwise_stations(n, spacing=chord_spacing)
    y = 5 * t_c * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    upper = np.column_stack([x, y])
    lower = np.column_stack([x, -y])
    upper[-1, :] = (1.0, 0.0)
    lower[-1, :] = (1.0, 0.0)
    return np.vstack([upper, lower])


def naca_four_digit(
    m: float,
    p: float,
    t_c: float,
    n: int = 200,
    *,
    chord_spacing: str = "nested_cosine",
):
    """
    NACA 4-digit cambered profile (asymmetric if ``m > 0``).

    The returned array has **equal-length** upper and lower halves (``2n`` rows):
    upper and lower both run **LE→TE** at the **same** chordwise distribution
    (default: ``nested_cosine`` with **one** station at LE and **one** at TE). The
    **trailing edge is sharp**: upper and lower meet at ``(1, y_c(1))`` with zero
    thickness there (the analytic NACA thickness at ``x/c=1`` is non-zero; it is
    overridden so the section closes to a corner). The **leading edge** is the
    single point ``(0, y_c(0))`` with zero thickness at ``x=0``. Dense sampling
    near TE (``nested_cosine`` / ``nested_cosine_3``) yields short final segments so the
    TE reads as a **convex** kink in plots rather than a long chopped facet.

    Parameters
    ----------
    m : float
        Maximum camber / chord (e.g. 0.02 for 2%).
    p : float
        Chordwise fraction (0–1) where maximum camber occurs (e.g. 0.4 for “4” in 2412).
    t_c : float
        Maximum thickness / chord (e.g. 0.12).
    n : int
        Number of chordwise stations on each surface (LE→TE).
    chord_spacing : str
        ``nested_cosine`` (default): strong LE/TE clustering. ``nested_cosine_3`` for
        maximum end density; ``cosine`` / ``le_dense`` / ``uniform``: see :func:`chordwise_stations`.

    Returns
    -------
    (2n, 2) array ``[y, z]``: upper LE→TE, then lower LE→TE (``sectorial_warping``
    and ``len(airfoil)//2`` expect this layout).
    """
    x = chordwise_stations(n, spacing=chord_spacing)
    yt = 5 * t_c * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    yc = np.zeros_like(x)
    if m > 1e-15 and p > 1e-9 and p < 1.0 - 1e-9:
        mask = x < p
        yc[mask] = (m / p**2) * (2 * p * x[mask] - x[mask] ** 2)
        yc[~mask] = (m / (1 - p) ** 2) * (
            (1 - 2 * p) + 2 * p * x[~mask] - x[~mask] ** 2
        )
    z_te = float(yc[-1])
    upper = np.column_stack([x, yc + yt])
    lower = np.column_stack([x, yc - yt])
    upper[-1, :] = (1.0, z_te)
    lower[-1, :] = (1.0, z_te)
    return np.vstack([upper, lower])


def polygon_area_2d(poly: np.ndarray) -> float:
    """Shoelace area for a simple closed polygon (y,z) in order."""
    y = poly[:, 0]
    z = poly[:, 1]
    return 0.5 * float(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))


def cell_enclosed_areas(airfoil, spar_positions) -> list[float]:
    """
    Midline-enclosed area of each cell between adjacent spar webs (chord units²).

    Cell ``i`` lies between ``all_x[i]`` and ``all_x[i+1]`` with vertical webs at the
    interior boundaries (except the outer LE/TE strips).
    """
    all_x = [0.0] + sorted(spar_positions) + [1.0]
    areas = []
    for i in range(len(all_x) - 1):
        xa, xb = all_x[i], all_x[i + 1]
        pau = interp_surface(airfoil, xa, "upper")
        pbu = interp_surface(airfoil, xb, "upper")
        pal = interp_surface(airfoil, xa, "lower")
        pbl = interp_surface(airfoil, xb, "lower")
        n_half = len(airfoil) // 2
        upper_full = airfoil[:n_half]
        lower_full = airfoil[n_half:]
        mask_u = (upper_full[:, 0] > xa + 1e-9) & (upper_full[:, 0] < xb - 1e-9)
        mask_l = (lower_full[:, 0] > xa + 1e-9) & (lower_full[:, 0] < xb - 1e-9)
        mid_u = upper_full[mask_u]
        mid_l = lower_full[mask_l][::-1]
        upper_chain = np.vstack([pau] + ([mid_u] if len(mid_u) else []) + [pbu])
        lower_return = np.vstack([pbl] + ([mid_l] if len(mid_l) else []) + [pal])
        loop = np.vstack([upper_chain, lower_return])
        loop = np.vstack([loop, loop[:1]])
        areas.append(abs(polygon_area_2d(loop)))
    return areas


def interp_surface(airfoil, x_target, surface="upper"):
    """Interpolate [y, z] at a given chord position.

    The lower half (rows ``len(airfoil)//2`` onward) is stored **LE→TE** like the
    upper half; rows are sorted by chord before ``np.interp``.
    """
    n = len(airfoil) // 2
    seg = airfoil[:n] if surface == "upper" else airfoil[n:]
    order = np.argsort(seg[:, 0])
    seg = seg[order]
    z = np.interp(x_target, seg[:, 0], seg[:, 1])
    return np.array([x_target, z])


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SECTION TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────

def build_section(
    airfoil,
    spar_positions,
    cap_width=0.025,
    cap_height=0.008,
    *,
    skin_lam: Laminate | None = None,
):
    """
    Decompose the blade section into Panel and BoomNode objects.

    Integration order:
        upper skin panels L→R  (TE → spar A → spar B → LE)
      + lower skin panels R→L  (LE → spar B → spar A → TE)
      + web panels (top → bottom at each spar)

    At each spar web location:
      - BoomNode placed at upper and lower skin/web junction
      - end_boom on the upstream skin panel adds ΔQ jump

    Parameters
    ----------
    airfoil         : (N,2) array
    spar_positions  : list of float — spar web x/c positions
    cap_width       : float — cap chord extent [m] (for area calculation)
    cap_height      : float — cap thickness [m]

    Returns
    -------
    panels    : list of Panel  — ordered for Q integration
    booms     : list of BoomNode
    webs_geom : list of (upper_pt, lower_pt) tuples for drawing
    n_cells   : int

    skin_lam : Laminate or None
        Laminate used for upper/lower skin panels only (caps and webs unchanged).
    """
    sl = skin_lam if skin_lam is not None else SKIN_LAM
    n_half = len(airfoil) // 2
    upper_full = airfoil[:n_half]
    lower_full = airfoil[n_half:]
    x_le = float(np.min(airfoil[:, 0]))
    x_te = float(np.max(airfoil[:, 0]))
    all_x = [x_le] + sorted(float(x) for x in spar_positions) + [x_te]
    A_cap = cap_width * cap_height

    upper_skins = []; lower_skins = []
    web_panels  = []; booms = []; webs_geom = []

    for i in range(len(all_x) - 1):
        x0, x1 = all_x[i], all_x[i+1]

        # Key node coordinates
        p0u = interp_surface(airfoil, x0, "upper")
        p1u = interp_surface(airfoil, x1, "upper")
        p0l = interp_surface(airfoil, x0, "lower")
        p1l = interp_surface(airfoil, x1, "lower")

        # Intermediate skin nodes between spar positions
        mask_u = (upper_full[:,0] > x0+1e-9) & (upper_full[:,0] < x1-1e-9)
        mask_l = (lower_full[:,0] > x0+1e-9) & (lower_full[:,0] < x1-1e-9)
        mid_u = upper_full[mask_u]
        # lower_full is LE→TE (increasing chord); mask gives interior points in that
        # order — reverse so the panel runs from p1l (x1) toward p0l (x0).
        mid_l = lower_full[mask_l][::-1]

        u_nodes = np.vstack([p0u] + ([mid_u] if len(mid_u) else []) + [p1u])
        l_nodes = np.vstack([p1l] + ([mid_l] if len(mid_l) else []) + [p0l])

        # Boom at right end of upper skin (not at TE)
        boom_u = None
        if x1 < x_te - 1e-9:
            boom_u = BoomNode(y=p1u[0], z=p1u[1], A_cap=A_cap,
                              label=f"Cap U @ {x1:.2f}c")
            booms.append(boom_u)

        # Boom at left end of lower skin (not at LE)
        boom_l = None
        if x0 > x_le + 1e-9:
            boom_l = BoomNode(y=p0l[0], z=p0l[1], A_cap=A_cap,
                              label=f"Cap L @ {x0:.2f}c")
            booms.append(boom_l)

        upper_skins.append(Panel(u_nodes, sl, i, boom_u, f"USkin C{i+1}"))
        lower_skins.append(Panel(l_nodes, sl, i, boom_l, f"LSkin C{i+1}"))

        # Shear web at x1 (top → bottom)
        if x1 < x_te - 1e-9:
            w_nodes = np.column_stack([
                np.full(20, x1),
                np.linspace(p1u[1], p1l[1], 20)
            ])
            web_panels.append(Panel(w_nodes, WEB_LAM, i, None, f"Web @ {x1:.2f}c"))
            webs_geom.append((p1u, p1l))

    n_cells = len(all_x) - 1
    ordered = upper_skins + lower_skins[::-1] + web_panels
    return ordered, booms, webs_geom, n_cells


# ─────────────────────────────────────────────────────────────────────────────
# SECTION PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

def section_properties(panels, booms) -> SectionProps:
    """
    Modulus-weighted centroid, EA, and full second-moment tensor about centroid.

    Iyy = ∫ E_n (z-zc)² t ds + Σ A_eff (z-zb)², etc.; Iyz includes product coupling.
    """
    EA = 0.0; EAy = 0.0; EAz = 0.0

    for p in panels:
        if len(p.s) < 2:
            continue
        EA  += np.trapezoid(p.E_n * p.lam.t * np.ones(len(p.s)), p.s)
        EAy += np.trapezoid(p.E_n * p.lam.t * p.nodes[:, 0], p.s)
        EAz += np.trapezoid(p.E_n * p.lam.t * p.nodes[:, 1], p.s)
    for b in booms:
        EA += b.A_eff; EAy += b.A_eff * b.y; EAz += b.A_eff * b.z

    y_c = EAy / EA
    z_c = EAz / EA

    Iyy = 0.0; Izz = 0.0; Iyz = 0.0
    for p in panels:
        if len(p.s) < 2:
            continue
        yn = p.nodes[:, 0] - y_c
        zn = p.nodes[:, 1] - z_c
        Iyy += np.trapezoid(p.E_n * p.lam.t * zn**2, p.s)
        Izz += np.trapezoid(p.E_n * p.lam.t * yn**2, p.s)
        Iyz += np.trapezoid(p.E_n * p.lam.t * yn * zn, p.s)
    for b in booms:
        yn = b.y - y_c
        zn = b.z - z_c
        Iyy += b.A_eff * zn**2
        Izz += b.A_eff * yn**2
        Iyz += b.A_eff * yn * zn

    return SectionProps(y_c=y_c, z_c=z_c, EA=EA, Iyy=Iyy, Izz=Izz, Iyz=Iyz)


# ─────────────────────────────────────────────────────────────────────────────
# Q INTEGRATION WITH DISCRETE BOOM JUMPS (BIAXIAL SHEAR)
# ─────────────────────────────────────────────────────────────────────────────

def integrate_Q(panels, props: SectionProps, Vy: float, Vz: float):
    """
    Open-section shear flow q_b for transverse shear (Vy, Vz) through the
    modulus-weighted centroid (no shear-centre offset here — use loads at SC if needed).

    Uses coupled thin-wall formula with Qy, Qz first moments and Iyz coupling
    (Megson / standard texts):

        q_b = -(1/D) * [ Vz * (Izz*Qz - Iyz*Qy) + Vy * (Iyy*Qy - Iyz*Qz) ],

    with D = Iyy*Izz - Iyz², and dQy/ds = E_n*(y-yc)*t, dQz/ds = E_n*(z-zc)*t,
    plus boom jumps in both Qy and Qz at caps.
    """
    Iyy, Izz, Iyz = props.Iyy, props.Izz, props.Iyz
    y_c, z_c = props.y_c, props.z_c
    den = Iyy * Izz - Iyz ** 2
    if abs(den) < 1e-40:
        den = 1e-40

    Qy_run = 0.0
    Qz_run = 0.0
    q_b_panels = []
    Qy_panels = []
    Qz_panels = []

    for p in panels:
        if len(p.s) < 2:
            q_flat = -(
                Vz * (Izz * Qz_run - Iyz * Qy_run)
                + Vy * (Iyy * Qy_run - Iyz * Qz_run)
            ) / den
            q_b_panels.append(np.full(len(p.s), q_flat))
            Qy_panels.append(np.full(len(p.s), Qy_run))
            Qz_panels.append(np.full(len(p.s), Qz_run))
            continue

        yn = p.nodes[:, 0] - y_c
        zn = p.nodes[:, 1] - z_c
        dQy = p.E_n * yn * p.lam.t
        dQz = p.E_n * zn * p.lam.t
        Qyl = Qy_run + np.array([
            np.trapezoid(dQy[:i + 1], p.s[:i + 1]) for i in range(len(p.s))
        ])
        Qzl = Qz_run + np.array([
            np.trapezoid(dQz[:i + 1], p.s[:i + 1]) for i in range(len(p.s))
        ])
        Qy_panels.append(Qyl)
        Qz_panels.append(Qzl)
        q_b_panels.append(
            -(
                Vz * (Izz * Qzl - Iyz * Qyl)
                + Vy * (Iyy * Qyl - Iyz * Qzl)
            )
            / den
        )
        Qy_run = Qyl[-1]
        Qz_run = Qzl[-1]

        if p.end_boom is not None:
            b = p.end_boom
            Qy_run += b.A_eff * (b.y - y_c)
            Qz_run += b.A_eff * (b.z - z_c)

    return q_b_panels, Qy_panels, Qz_panels

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-CELL BREDT / TORSION COMPATIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def bredt_flexibility_matrix(panels, n_cells: int) -> np.ndarray:
    """Flexibility matrix A with A_ij ~ ∮_i (1/t) ds coupling for shared webs."""
    A_mat = np.zeros((n_cells, n_cells))
    for p in panels:
        i = p.cell_id
        if i < 0 or i >= n_cells or len(p.s) < 2:
            continue
        t = p.lam.t
        A_mat[i, i] += np.trapezoid(np.ones(len(p.s)) / t, p.s)
        if "Web" in p.label and i < n_cells - 1:
            wl = p.s[-1]
            A_mat[i, i + 1] -= wl / t
            A_mat[i + 1, i] -= wl / t
            A_mat[i + 1, i + 1] += wl / t
    return A_mat


def bredt_rhs_opening(panels, q_b_panels, n_cells: int) -> np.ndarray:
    """rhs_i = -∮_i q_b/t ds (shear-flow opening)."""
    rhs = np.zeros(n_cells)
    for p, qb in zip(panels, q_b_panels):
        i = p.cell_id
        if i < 0 or i >= n_cells or len(p.s) < 2:
            continue
        t = p.lam.t
        rhs[i] -= np.trapezoid(qb / t, p.s)
        if "Web" in p.label and i < n_cells - 1:
            rhs[i + 1] += np.trapezoid(qb / t, p.s)
    return rhs


def solve_bredt_q0(
    panels,
    q_b_panels,
    n_cells: int,
    cell_areas: list[float],
    T: float,
) -> np.ndarray:
    """
    Closed-cell constant flows q0_i such that q = q_b + q0 on each cell wall.

    Shear only (T=0): A q0 = rhs_opening.

    Shear + St. Venant–Bredt torsion: augment with twist rate θ and torque balance
    using ``G_REF`` (see Megson / Kollár thin-wall torsion matrices):

        [ A   | -2 G_REF A_cell ] [ q0   ] = [ rhs_opening ]
        [ 2A^T|  0              ] [ θ'   ] = [ T           ]

    For T=0 the (n+1)×(n+1) system recovers θ'≈0 and the shear-only solution.
    """
    A_mat = bredt_flexibility_matrix(panels, n_cells)
    rhs_s = bredt_rhs_opening(panels, q_b_panels, n_cells)
    ca = np.asarray(cell_areas[:n_cells], dtype=float)
    n = n_cells

    if abs(T) < 1e-30:
        return np.linalg.solve(A_mat, rhs_s)

    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A_mat
    M[:n, n] = -2.0 * G_REF * ca
    M[n, :n] = 2.0 * ca
    M[n, n] = 0.0
    rvec = np.zeros(n + 1)
    rvec[:n] = rhs_s
    rvec[n] = T
    x = np.linalg.solve(M, rvec)
    return x[:n]


def torque_from_shear_flow(panels, q_panels) -> float:
    """Resultant moment about +x from thin-wall shear flow: T_x ~ ∮ q (y dz - z dy)."""
    tot = 0.0
    for p, q in zip(panels, q_panels):
        y = p.nodes[:, 0]
        z = p.nodes[:, 1]
        qv = np.asarray(q, dtype=float)
        for i in range(len(y) - 1):
            dy = y[i + 1] - y[i]
            dz = z[i + 1] - z[i]
            qs = 0.5 * (qv[i] + qv[i + 1])
            tot += qs * (y[i] * dz - z[i] * dy)
    return float(tot)


def shear_center_flexural_equilibrium(
    panels,
    props: SectionProps,
    n_cells: int,
    cell_areas: list[float],
) -> tuple[float, float]:
    """
    Shear centre from flexural shear flows (Vy,Vz) with T=0: torque from q about
    centroid vanishes when transverse load acts at (y_sc, z_sc). Standard for
    closed thin-walled cells (Megson); robust for multi-cell airfoils.
    """
    y_c, z_c = props.y_c, props.z_c

    def total_q(Vy_u: float, Vz_u: float):
        qb, _, _ = integrate_Q(panels, props, Vy_u, Vz_u)
        q0 = solve_bredt_q0(panels, qb, n_cells, cell_areas, 0.0)
        out = []
        for p, qv in zip(panels, qb):
            if 0 <= p.cell_id < n_cells:
                out.append(qv + q0[p.cell_id])
            else:
                out.append(qv)
        return out

    q_vz = total_q(0.0, 1.0)
    q_vy = total_q(1.0, 0.0)
    t_vz = torque_from_shear_flow(panels, q_vz)
    t_vy = torque_from_shear_flow(panels, q_vy)
    y_sc = y_c + t_vz / 1.0
    z_sc = z_c - t_vy / 1.0
    return float(y_sc), float(z_sc)


def _interp_outline_stress_to_points(
    verts: np.ndarray,
    sigma_vertex: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of σ on an **open** polyline (no wrap)."""
    n = len(verts)
    out = np.zeros(len(points), dtype=float)
    for iq, q in enumerate(points):
        dmin = 1e300
        best = 0.0
        for i in range(n - 1):
            p0 = verts[i]
            p1 = verts[i + 1]
            e = p1 - p0
            el2 = float(np.dot(e, e))
            if el2 < 1e-30:
                continue
            t = float(np.dot(q - p0, e) / el2)
            t = max(0.0, min(1.0, t))
            proj = p0 + t * e
            d = float(np.linalg.norm(q - proj))
            if d < dmin:
                dmin = d
                best = (1.0 - t) * sigma_vertex[i] + t * sigma_vertex[i + 1]
        out[iq] = best
    return out


# ─────────────────────────────────────────────────────────────────────────────
# AXIAL STRESS (N + BIAXIAL BENDING)
# ─────────────────────────────────────────────────────────────────────────────

def compute_axial_stress(panels, booms, props: SectionProps, N: float, My: float, Mz: float):
    """
    σ = E_n * ( N/EA + κ_y (z-z_c) + κ_z (y-y_c) ) with [κ_y, κ_z]^T = I^{-1} [My, Mz]^T.
    """
    I_mat = np.array([[props.Iyy, props.Iyz], [props.Iyz, props.Izz]], dtype=float)
    M_vec = np.array([My, Mz], dtype=float)
    kappa = np.linalg.solve(I_mat, M_vec)
    ky, kz = float(kappa[0]), float(kappa[1])

    sigma_panels = []
    for p in panels:
        yn = p.nodes[:, 0] - props.y_c
        zn = p.nodes[:, 1] - props.z_c
        sigma_panels.append(p.E_n * (N / props.EA + ky * zn + kz * yn))

    sigma_booms = []
    for b in booms:
        Enb = b.lam.E / E_REF
        yn = b.y - props.y_c
        zn = b.z - props.z_c
        sigma_booms.append(Enb * (N / props.EA + ky * zn + kz * yn))

    return sigma_panels, sigma_booms


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_section(
    airfoil,
    spars,
    *,
    skin_lam: Laminate | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    B: float = 0.0,
    dB_dx: float = 0.0,
):
    """
    Build section → properties → shear flow (Vy,Vz,T) → axial stress (N,My,Mz)
    plus optional Vlasov warping σ_ω from bimoment ``B``.

    **Secondary warping shear:** if ``dB_dx`` is non-zero, adds shear flow from
    thin-wall equilibrium ``∂q/∂s ≈ −t ∂σ_ω/∂x`` with ``σ_ω = B ω̂ / I_ω``, i.e.
    ``∂σ_ω/∂x = (dB/dx) ω̂ / I_ω``. The open-chain particular integral starts at
    ``q=0`` at the first outline vertex; **multi-cell closure** uses the same
    Bredt linear system as primary flexural shear (``T=0`` for the warping part).
    Primary (Megson/Bredt) and warping contributions are summed into ``q_tot``;
    see also ``q_primary`` and ``q_warp`` in the return tuple.

    Shear centre from flexural shear-flow torque balance (same as before).
    Timoshenko shear strains from smeared GA_y, GA_z over all panels.

    skin_lam
        Optional skin laminate (upper/lower skins only). Defaults to ``SKIN_LAM``.
        Warping / sectorial thickness uses this laminate's ``t``.
    """
    sl = skin_lam if skin_lam is not None else SKIN_LAM
    panels, booms, webs_geom, n_cells = build_section(airfoil, spars, skin_lam=sl)
    props = section_properties(panels, booms)
    areas = cell_enclosed_areas(airfoil, spars)

    loop_open = open_outline_from_airfoil(airfoil)
    y_sc, z_sc = shear_center_flexural_equilibrium(
        panels, props, n_cells, areas
    )
    yco, zco = props.y_c, props.z_c
    omega_hat, _len_ds = normalized_warping(
        loop_open, y_sc, z_sc, yco, zco, sl.t
    )
    I_omega = warping_constant_I_omega(loop_open, omega_hat, sl.t)

    q_b, _, _ = integrate_Q(panels, props, Vy, Vz)
    q0 = solve_bredt_q0(panels, q_b, n_cells, areas, T)

    q_primary = [
        qb + q0[p.cell_id] if 0 <= p.cell_id < n_cells else qb
        for p, qb in zip(panels, q_b)
    ]

    q_open_vert = q_omega_secondary_open_vertices(
        loop_open, omega_hat, sl.t, I_omega, dB_dx
    )
    q_warp_particular = q_omega_secondary_panels_particular(
        loop_open, omega_hat, q_open_vert, panels, dB_dx, I_omega
    )
    q0_warp = solve_bredt_q0(panels, q_warp_particular, n_cells, areas, 0.0)
    q_warp = [
        qw + q0_warp[p.cell_id] if 0 <= p.cell_id < n_cells else qw
        for p, qw in zip(panels, q_warp_particular)
    ]

    q_tot = [qp + qw for qp, qw in zip(q_primary, q_warp)]

    sig_p, sig_b = compute_axial_stress(panels, booms, props, N, My, Mz)
    sig_w_vertex = sigma_from_bimoment(omega_hat, B, I_omega)
    for i, p in enumerate(panels):
        if "Web" in p.label:
            continue
        sig_p[i] = sig_p[i] + _interp_outline_stress_to_points(
            loop_open, sig_w_vertex, p.nodes
        )

    tangents = []
    lengths = []
    plies_pp = []
    for p in panels:
        if len(p.nodes) < 2:
            tangents.append((1.0, 0.0))
            lengths.append(0.0)
        else:
            d = p.nodes[1] - p.nodes[0]
            tangents.append((float(d[0]), float(d[1])))
            lengths.append(float(p.s[-1]))
        plies_pp.append(p.lam.build_plies())
    GA_y, GA_z = global_shear_stiffness_from_panels(
        tangents, plies_pp, panel_lengths=lengths
    )
    gamma_y, gamma_z = timoshenko_shear_strains(Vy, Vz, GA_y, GA_z)

    return (
        panels,
        booms,
        webs_geom,
        q_tot,
        sig_p,
        sig_b,
        q0,
        props,
        y_sc,
        z_sc,
        areas,
        I_omega,
        gamma_y,
        gamma_z,
        GA_y,
        GA_z,
        q_primary,
        q_warp,
    )


def _skin_station_clpt_data(
    panels,
    q_tot,
    sig_p,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict | None = None,
):
    """
    Thin-wall (σ_xx, q) at one skin panel station → membrane CLPT → Hashin 1980 envelope
    failure index per ply (default in :func:`clpt_ply_failure_indices`).

    **Model.** Laminate x = span, y = contour, σ_yy ≈ 0, τ_xy = q/t;
    Nx = σ_xx·t, Nxy = q.
    """
    if strengths is None:
        strengths = SKIN_STRENGTH
    p = panels[panel_index]
    plies = p.lam.build_plies()
    npt = len(p.s)
    if npt < 2:
        raise ValueError("Panel has insufficient stations for CLPT strip plot.")
    j = npt // 2 if station_index is None else int(station_index)
    j = max(0, min(npt - 1, j))

    sig_xx = float(sig_p[panel_index][j])
    q_here = float(q_tot[panel_index][j])
    t_wall = float(p.lam.t)
    tau_xy = q_here / max(t_wall, 1e-30)

    N_vec = membrane_resultants_from_shell_stress(sig_xx, 0.0, tau_xy, t_wall)
    M_vec = np.zeros(3, dtype=float)

    fi, eps0, kappa, sig_lam = clpt_ply_failure_indices(
        plies,
        N_vec,
        M_vec,
        strengths["Xt"],
        strengths["Xc"],
        strengths["Yt"],
        strengths["Yc"],
        strengths["S12"],
    )
    eps_lam = ply_mid_strains(plies, eps0, kappa)
    return {
        "p": p,
        "plies": plies,
        "fi": fi,
        "eps0": eps0,
        "kappa": kappa,
        "sig_lam": sig_lam,
        "eps_lam": eps_lam,
        "j": j,
        "npt": npt,
        "sig_xx": sig_xx,
        "tau_xy": tau_xy,
        "q_here": q_here,
        "t_wall": t_wall,
        "strengths": strengths,
    }


def skin_station_clpt_max_fi(
    panels,
    q_tot,
    sig_p,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict | None = None,
) -> float:
    """Maximum CLPT-ply failure index (Hashin envelope, default) over plies at one skin station."""
    d = _skin_station_clpt_data(
        panels,
        q_tot,
        sig_p,
        panel_index=panel_index,
        station_index=station_index,
        strengths=strengths,
    )
    fi = d["fi"]
    nply = len(fi)
    return float(np.max(fi)) if nply else 0.0


def optimize_skin_n_for_fi(
    airfoil,
    spars,
    *,
    t_ply: float = T_PLY_SKIN,
    E_skin: float | None = None,
    nu_skin: float | None = None,
    n_min: int = 2,
    n_max: int = 64,
    strengths: dict | None = None,
    panel_index: int = 0,
    station_index: int | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    B: float = 0.0,
    dB_dx: float = 0.0,
):
    """
    Scan ply count ``n`` upward: skin thickness ``t_skin = n * t_ply``, re-run
    ``run_section`` each time so σ and q stay coupled to skin stiffness.

    Minimizes areal mass ``ρ_areal = RHO_SKIN * t_skin`` (fixed ``ρ``, ``t_ply``)
    by taking the **smallest** ``n`` with max CLPT Hashin-envelope FI ``< 1`` at the chosen station.

    Returns ``None`` if no ``n`` in ``[n_min, n_max]`` is feasible.
    """
    E = E_skin if E_skin is not None else SKIN_LAM.E
    nu = nu_skin if nu_skin is not None else SKIN_LAM.nu
    st = strengths if strengths is not None else SKIN_STRENGTH

    print(
        f"Ply-count search: t_ply={t_ply*1000:.4f} mm, E={E/1e9:.2f} GPa, "
        f"n in [{n_min}, {n_max}], station panel={panel_index}"
    )
    print(f"{'n':>4}  {'t_skin [mm]':>14}  {'rho_areal [kg/m^2]':>18}  {'max FI':>10}")
    best = None
    for n in range(n_min, n_max + 1):
        skin = skin_laminate(E, nu, t_ply, n, name=f"Skin n={n}")
        out = run_section(
            airfoil,
            spars,
            skin_lam=skin,
            N=N,
            Vy=Vy,
            Vz=Vz,
            My=My,
            Mz=Mz,
            T=T,
            B=B,
            dB_dx=dB_dx,
        )
        panels, q_tot, sig_p = out[0], out[3], out[4]
        fi_max = skin_station_clpt_max_fi(
            panels,
            q_tot,
            sig_p,
            panel_index=panel_index,
            station_index=station_index,
            strengths=st,
        )
        t_skin = n * t_ply
        rho_areal = RHO_SKIN * t_skin
        mark = "  *" if fi_max < 1.0 and best is None else ""
        print(f"{n:4d}  {t_skin*1000:14.4f}  {rho_areal:16.4f}  {fi_max:10.4f}{mark}")
        if fi_max < 1.0 and best is None:
            best = {
                "n": n,
                "t_skin": t_skin,
                "rho_areal": rho_areal,
                "max_fi": fi_max,
                "section_output": out,
            }
            break

    if best is None:
        print("No feasible ply count in range (max Hashin FI >= 1 for all n).")
    else:
        print(
            f"Optimum (minimum n with FI < 1): n={best['n']}, "
            f"t_skin={best['t_skin']*1000:.4f} mm, "
            f"rho_areal={best['rho_areal']:.4f} kg/m^2, max FI={best['max_fi']:.4f}"
        )
    return best


def plot_clpt_laminate_stress_fi(
    panels,
    q_tot,
    sig_p,
    *,
    panel_index: int = 0,
    station_index: int | None = None,
    strengths: dict | None = None,
    outfile: Path | str | None = None,
    title: str | None = None,
):
    """
    Educational figure: one skin station’s thin-wall (σ_xx, q) → membrane CLPT
    (Nx, Nxy, M = 0) → ply σ, ε, Hashin-envelope FI. The right-hand text block explains
    the workflow, why FI is usually tiny under default unit loads, and demo limits.

    **Model.** Laminate x = span, y = contour, σ_yy ≈ 0, τ_xy = q/t;
    Nx = σ_xx·t, Nxy = q.
    """
    d = _skin_station_clpt_data(
        panels,
        q_tot,
        sig_p,
        panel_index=panel_index,
        station_index=station_index,
        strengths=strengths,
    )
    p = d["p"]
    plies = d["plies"]
    fi = d["fi"]
    eps0 = d["eps0"]
    kappa = d["kappa"]
    sig_lam = d["sig_lam"]
    eps_lam = d["eps_lam"]
    j = d["j"]
    npt = d["npt"]
    sig_xx = d["sig_xx"]
    tau_xy = d["tau_xy"]
    q_here = d["q_here"]
    t_wall = d["t_wall"]
    strengths = d["strengths"]

    nply = len(plies)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0f1117")

    ax0, ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    layers = np.arange(nply)

    # Stress / strain components in material axes (mid-ply)
    s11 = np.zeros(nply)
    s22 = np.zeros(nply)
    t12 = np.zeros(nply)
    e11 = np.zeros(nply)
    e22 = np.zeros(nply)
    g12 = np.zeros(nply)
    for k in range(nply):
        sm = stress_laminate_to_material(sig_lam[k], plies[k].theta_deg)
        em = stress_laminate_to_material(eps_lam[k], plies[k].theta_deg)
        s11[k], s22[k], t12[k] = sm[0], sm[1], sm[2]
        e11[k], e22[k], g12[k] = em[0], em[1], em[2]

    bh = 0.22
    ax0.barh(layers - bh, s11 / 1e6, height=bh * 0.92, label="σ11", color="#5dade2")
    ax0.barh(layers, s22 / 1e6, height=bh * 0.92, label="σ22", color="#ec7063")
    ax0.barh(layers + bh, t12 / 1e6, height=bh * 0.92, label="τ12", color="#af7ac5")
    ax0.axvline(0.0, color="#888888", lw=0.6)
    ax0.set_yticks(layers)
    ax0.set_yticklabels([f"P{k+1} ({plies[k].theta_deg:.0f}°)" for k in range(nply)])
    ax0.set_xlabel("Stress [MPa]")
    ax0.set_title("Ply σ — material axes\n(1 = fibre, 2 = transverse, 12 = in-plane shear)")
    ax0.legend(loc="lower right", fontsize=7)

    ax1.barh(layers - bh, e11 * 1e6, height=bh * 0.92, label="ε11 (µε)", color="#82e0aa")
    ax1.barh(layers, e22 * 1e6, height=bh * 0.92, label="ε22 (µε)", color="#f7dc6f")
    ax1.barh(layers + bh, g12 * 1e6, height=bh * 0.92, label="γ12 (µε)", color="#f5b7b1")
    ax1.axvline(0.0, color="#888888", lw=0.6)
    ax1.set_yticks(layers)
    ax1.set_yticklabels([f"P{k+1}" for k in range(nply)])
    ax1.set_xlabel("Strain [microstrain]")
    ax1.set_title("Ply ε — same material axes")
    ax1.legend(loc="lower right", fontsize=7)

    ax2.bar(layers, fi, width=0.65, label="Hashin envelope FI", color="#bb8fce")
    ax2.axhline(1.0, color="#e74c3c", ls="--", lw=1.0, label="FI = 1 (onset)")
    ax2.set_xticks(layers)
    ax2.set_xticklabels([f"P{k+1}" for k in range(nply)])
    ax2.set_ylabel("Failure index")
    ax2.set_title(
        "Hashin envelope per ply\n(FI < 1 ⇒ inside failure surface for this mode mix)"
    )
    ax2.legend(loc="upper right", fontsize=8)

    ax3.axis("off")
    fi_max = float(np.max(fi)) if nply else 0.0
    summary = (
        "WHAT THIS FIGURE SHOWS\n"
        "  • One skin station: the thin-wall section solver gives spanwise σ_xx and\n"
        "    shear flow q along that panel; we map them to laminate membrane loads\n"
        "    Nx = σ_xx·t, Nxy = q with M = 0 (no bending through the thickness).\n"
        "  • ABD → mid-surface ε⁰, κ → ply mid-plane σ and ε in material axes.\n"
        "  • Hashin envelope FI uses SKIN_STRENGTH (Xt, Xc, …); FI = 1 lies on a mode onset (envelope).\n"
        "\n"
        "THIS RUN (numbers)\n"
        f"  • Panel {p.label!r}  [panel_index={panel_index}]  station s-index {j}/{npt-1}\n"
        f"  • Skin thickness t = {t_wall*1000:.3f} mm\n"
        f"  • σ_xx (span) = {sig_xx/1e6:.4f} MPa    τ_xy = q/t = {tau_xy/1e6:.4f} MPa    q = {q_here:.4f} N/m\n"
        f"  • max Hashin envelope FI = {fi_max:.3e}  (expect ≪ 1 for default unit resultants)\n"
        "\n"
        "LIMITATIONS (demo)\n"
        "  • Demo stack is isotropic [0/90/90/0], not a real ±45 skin layup.\n"
        "  • Membrane-only (no thickness-direction bending from shell curvature).\n"
        "\n"
        "MID-SURFACE RESULTANTS\n"
        f"  ε⁰ = [{eps0[0]:.4e}, {eps0[1]:.4e}, {eps0[2]:.4e}]    "
        f"κ = [{kappa[0]:.4e}, {kappa[1]:.4e}, {kappa[2]:.4e}]\n"
        "\n"
        "STRENGTH INPUTS [MPa]\n"
        f"  Xt={strengths['Xt']/1e6:.0f}  Xc={strengths['Xc']/1e6:.0f}  "
        f"Yt={strengths['Yt']/1e6:.0f}  Yc={strengths['Yc']/1e6:.0f}  S12={strengths['S12']/1e6:.0f}"
    )
    ax3.text(
        0.02,
        0.98,
        summary,
        transform=ax3.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        color="#e0e0e0",
        linespacing=1.35,
    )

    ttl = title or (
        "CLT sanity check: one skin station → ply σ, ε, Hashin-envelope FI\n"
        "(membrane Nx, Nxy only; tune loads/strengths to see FI → 1)"
    )
    fig.suptitle(ttl, color="#e0e0e0", fontsize=10, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        print(f"Saved: {outfile}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def outward_normals(nodes):
    """
    Unit normals perpendicular to the polyline tangent (2D, CCW 90° from tangent).

    Uses chord-based tangents (central difference mid-edge; forward/back at ends) so
    the nose/TE are stable on dense meshes; ``np.gradient`` was prone to asymmetric
    end formulas that exaggerated LE normal jumps between adjacent skin panels.
    """
    p = np.asarray(nodes, dtype=float)
    n = len(p)
    if n < 2:
        return np.array([[0.0, 1.0]], dtype=float)
    tvec = np.zeros_like(p)
    tvec[0] = p[1] - p[0]
    tvec[-1] = p[-1] - p[-2]
    if n > 2:
        tvec[1:-1] = p[2:] - p[:-2]
    ln = np.linalg.norm(tvec, axis=1, keepdims=True)
    ln[ln < 1e-10] = 1e-10
    tvec /= ln
    return np.column_stack([-tvec[:, 1], tvec[:, 0]])


def fill_distribution(ax, nodes, values, scale, pos_color, neg_color,
                      alpha=0.45, zorder=4, axial_stress_outward=False,
                      envelope_line_color=None):
    """
    Fill between wall and scaled distribution, coloured by sign.

    If ``axial_stress_outward`` is True, ribbon width uses ``|scalar|`` and the
    offset follows the **outward** normal; colour still distinguishes sign (tension
    vs compression for σ, positive vs negative ``q``). Use this for **both** σ and
    ``q`` so shear ribbons do not flip across the wall at the LE (where upper and
    lower panels meet with different tangent-based normals).
    """
    normals = outward_normals(nodes)
    mag = np.abs(values) if axial_stress_outward else values
    offset = nodes + scale * mag[:, None] * normals
    for mask, color in [(values >= 0, pos_color), (values < 0, neg_color)]:
        if not np.any(mask): continue
        wp = nodes[mask]; op = offset[mask]
        for i in range(len(wp) - 1):
            ax.fill([wp[i,0], op[i,0], op[i+1,0], wp[i+1,0]],
                    [wp[i,1], op[i,1], op[i+1,1], wp[i+1,1]],
                    color=color, alpha=alpha, linewidth=0, zorder=zorder)
    eline = envelope_line_color if envelope_line_color is not None else pos_color
    ax.plot(offset[:, 0], offset[:, 1],
            color=eline, lw=0.9, ls="--", alpha=0.85, zorder=zorder + 1)


def _auto_distribution_scales(
    q_tot, sig_p, q_span=0.07, sigma_span=0.055,
    *,
    use_q: bool = True,
    use_sigma: bool = True,
):
    """
    Map physical q [N/m] and σ [Pa] to plot scale factors so ribbon width
    stays a small fraction of chord (airfoil y,z are in chord units).
    Only computes scales for quantities that will be drawn.
    """
    scale_q = 1.0
    scale_s = 1.0
    if use_q and q_tot:
        q_flat = np.concatenate([np.asarray(q, dtype=float).ravel() for q in q_tot])
        q_max = float(np.max(np.abs(q_flat))) if q_flat.size else 0.0
        scale_q = q_span / max(q_max, 1e-30)
    if use_sigma and sig_p:
        s_flat = np.concatenate([np.asarray(s, dtype=float).ravel() for s in sig_p])
        s_max = float(np.max(np.abs(s_flat))) if s_flat.size else 0.0
        scale_s = sigma_span / max(s_max, 1e-30)
    return scale_q, scale_s


def plot_section(ax, panels, booms, webs_geom, airfoil, spar_positions,
                 q_tot, sig_p, sig_b, title,
                 scale_q=None, scale_s=None,
                 q_span=0.07, sigma_span=0.055,
                 plot_shear: bool = True,
                 plot_bending: bool = True,
                 xlabel: str = "y/c",
                 ylabel: str = "z/c",
                 title_fontsize: float = 10.0):
    """
    Plot σ(s) and/or q(s) distributions projected normal to all panel walls.
    Spar cap booms (orange) and σ labels are drawn only when plot_bending is True.

    If scale_q / scale_s are None, scales are chosen so max |q| and max |σ|
    ribbons use q_span and sigma_span (fractions of chord).
    """
    if scale_q is None or scale_s is None:
        auto_q, auto_s = _auto_distribution_scales(
            q_tot, sig_p, q_span, sigma_span,
            use_q=plot_shear, use_sigma=plot_bending,
        )
        if scale_q is None:
            scale_q = auto_q
        if scale_s is None:
            scale_s = auto_s

    # Raw ``airfoil`` is [upper_LE→TE; lower_LE→TE]; plotting rows in order draws a
    # spurious chord from TE to LE. Use the boundary polyline from sectorial_warping.
    outline = open_outline_from_airfoil(airfoil)
    af = np.vstack([outline, outline[:1]])
    ax.plot(
        af[:, 0], af[:, 1],
        color="#7eb8da", lw=2.4, solid_capstyle="butt", solid_joinstyle="miter",
        zorder=2, alpha=0.95,
    )

    for p, q_p, s_p in zip(panels, q_tot, sig_p):
        if plot_shear:
            fill_distribution(
                ax, p.nodes, q_p, scale_q, "#2ecc71", "#1abc9c",
                axial_stress_outward=True,
                envelope_line_color="#8dd4b8",
            )
        if plot_bending:
            fill_distribution(
                ax, p.nodes, s_p, scale_s, "#3498db", "#e74c3c",
                axial_stress_outward=True,
                envelope_line_color="#c4c8e0",
            )
        ax.plot(
            p.nodes[:, 0], p.nodes[:, 1],
            color="#f0f4f8", lw=1.1, alpha=0.75, zorder=8,
        )

    # Spar webs
    for (u, l) in webs_geom:
        ax.plot([u[0], l[0]], [u[1], l[1]],
                color="#eef6ff", lw=3.0, alpha=0.95, ls="--", zorder=9)

    # Spar cap booms — orange rectangles + σ annotation (bending plots only)
    if plot_bending:
        cap_h_vis = 0.012; cap_w_vis = 0.03
        for b, sb in zip(booms, sig_b):
            rect = plt.Rectangle(
                (b.y - cap_w_vis/2, b.z - cap_h_vis/2),
                cap_w_vis, cap_h_vis,
                lw=1.2, edgecolor="#f39c12", facecolor="#f39c12", alpha=0.6,
                zorder=22,
            )
            ax.add_patch(rect)
            ax.text(b.y, b.z + cap_h_vis * 1.3,
                    f"{sb/1e6:.2f} MPa",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#f39c12", alpha=0.9, zorder=23)

    # Cell labels
    all_x = [0.0] + spar_positions + [1.0]
    for i in range(len(all_x) - 1):
        x_mid = 0.5 * (all_x[i] + all_x[i+1])
        u_mid = interp_surface(airfoil, x_mid, "upper")
        ax.text(x_mid, u_mid[1] * 0.2, f"C{i+1}",
                ha="center", va="center",
                fontsize=8, color="white", alpha=0.4, zorder=12)

    ax.plot(
        af[:, 0], af[:, 1],
        color="#ffffff", lw=2.2, solid_capstyle="butt", solid_joinstyle="miter",
        zorder=18, alpha=0.98,
    )

    ax.set_aspect("equal")
    ax.set_title(title, color="#e0e0e0", fontsize=title_fontsize, pad=5)
    ax.set_xlabel(xlabel, color="#e0e0e0", fontsize=8)
    ax.set_ylabel(ylabel, color="#e0e0e0", fontsize=8)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3a")
    ax.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _save_section_figure(
    cached,
    airfoil,
    *,
    plot_shear: bool,
    plot_bending: bool,
    outfile: str,
    suptitle: str,
    legend_patches,
):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.patch.set_facecolor("#0f1117")

    for idx, (spars, label, out) in enumerate(cached):
        r, c = divmod(idx, 2)
        ax = axes[r, c]
        panels, booms, webs_geom, q_tot, sig_p, sig_b, *_rest = out
        plot_section(
            ax, panels, booms, webs_geom, airfoil, spars,
            q_tot, sig_p, sig_b, label,
            plot_shear=plot_shear, plot_bending=plot_bending,
        )

    fig.legend(
        handles=legend_patches, loc="upper center", ncol=min(3, len(legend_patches)),
        fontsize=9, facecolor="#1a1a2e", edgecolor="#2a2a3a",
        labelcolor="#e0e0e0", bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(suptitle, color="#e0e0e0", fontsize=10, y=1.055)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    _plot_dir = Path(__file__).resolve().parent / "outputs"
    _plot_dir.mkdir(parents=True, exist_ok=True)

    N_ax, Vy, Vz, Mz, My, T_tor = SECTION_RESULTANTS
    B_use = B_BIMOMENT

    configs = [
        ([],                              "0 Webs — 1 Cell"),
        ([0.35],                          "1 Web  — 2 Cells  (@ 35%)"),
        ([0.25, 0.60],                    "2 Webs — 3 Cells  (@ 25%, 60%)"),
        ([0.20, 0.45, 0.70],              "3 Webs — 4 Cells  (@ 20%, 45%, 70%)"),
        ([0.15, 0.35, 0.55, 0.75],        "4 Webs — 5 Cells  (@ 15%, 35%, 55%, 75%)"),
        ([0.12, 0.28, 0.45, 0.62, 0.78], "5 Webs — 6 Cells"),
    ]

    # NACA 2412-style cambered (asymmetric) section; chord normalised to 1 m in plots
    airfoil = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=360)

    cached = []
    for spars, label in configs:
        out = run_section(
            airfoil, spars,
            N=N_ax, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T_tor, B=B_use,
        )
        cached.append((spars, label, out))

    for spars, label, out in cached:
        (
            panels, booms, webs_geom, q_tot, sig_p, sig_b, q0, props,
            y_sc, z_sc, areas, I_omega, gamma_y, gamma_z, GA_y, GA_z,
            _q_primary, _q_warp,
        ) = out
        q_all = np.concatenate(q_tot)
        sig_all = np.concatenate(sig_p)
        print(f"\n{label}")
        print(f"  EA = {props.EA:.4e}   (y_c, z_c) = ({props.y_c:.5f}, {props.z_c:.5f}) m")
        print(f"  Iyy, Izz, Iyz = {props.Iyy:.4e}, {props.Izz:.4e}, {props.Iyz:.4e} m^4")
        print(f"  shear centre (y_sc, z_sc) = ({y_sc:.5f}, {z_sc:.5f}) m  (flexural q, T=0)")
        print(f"  I_omega = {I_omega:.4e}   GA_y, GA_z = {GA_y:.4e}, {GA_z:.4e} N")
        print(f"  Timoshenko gam_y, gam_z = {gamma_y:.4e}, {gamma_z:.4e} rad")
        print(f"  q0   = {np.round(q0, 4)}  (per cell, N/m scale with loads)")
        print(f"  q    in [{q_all.min():.4f}, {q_all.max():.4f}] N/m")
        print(f"  sigma_skin in [{sig_all.min()/1e6:.4f}, {sig_all.max()/1e6:.4f}] MPa")
        if sig_b:
            print(f"  sigma_cap = {[f'{s/1e6:.4f} MPa' for s in sig_b]}")

    plies_demo = SKIN_LAM.build_plies()
    A_clpt, _, _ = abd_stack(plies_demo)
    sig_p_demo = cached[0][2][4]
    sig_line = float(np.mean(np.abs(np.concatenate(sig_p_demo)))) if sig_p_demo else 0.0
    eps_span = sig_line / SKIN_LAM.E if SKIN_LAM.E > 0 else 0.0
    eps0 = np.array([eps_span, 0.0, 0.0])
    kappa0 = np.zeros(3)
    ply_sig = ply_stresses_bottom_top(plies_demo, eps0, kappa0)
    print("\nSkin CLPT (uniaxial strain eps_x ~ mean |sigma|/E, [0/90/90/0] stack):")
    print(f"  A11 = {A_clpt[0,0]:.4e} N/m   homogenized E1_eff ~ {homogenized_axial_modulus(plies_demo)/1e9:.2f} GPa")
    print(f"  ply1 sigma11 bottom/top MPa: {ply_sig[0][0][0]/1e6:.4f}, {ply_sig[0][1][0]/1e6:.4f}")

    print("\n--- Skin ply-count search (min rho_areal s.t. max Hashin envelope FI < 1) ---")
    optimize_skin_n_for_fi(
        airfoil,
        [0.25, 0.60],
        t_ply=T_PLY_SKIN,
        n_min=2,
        n_max=24,
        panel_index=0,
        station_index=None,
        N=N_ax,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T_tor,
        B=B_use,
    )

    legend_full = [
        mpatches.Patch(color="#2ecc71", alpha=0.7, label="q(s) > 0 (|q| outward)"),
        mpatches.Patch(color="#1abc9c", alpha=0.7, label="q(s) < 0 (|q| outward)"),
        mpatches.Patch(color="#3498db", alpha=0.7, label="sigma(s) — tensile bending"),
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="sigma(s) — compressive bending"),
        mpatches.Patch(color="#f39c12", alpha=0.7, label="Spar cap boom"),
        mpatches.Patch(color="white",   alpha=0.8, label="Skin / web wall"),
    ]
    legend_shear = [
        mpatches.Patch(color="#2ecc71", alpha=0.7, label="q(s) > 0 (|q| outward)"),
        mpatches.Patch(color="#1abc9c", alpha=0.7, label="q(s) < 0 (|q| outward)"),
        mpatches.Patch(color="white",   alpha=0.8, label="Skin / web wall"),
    ]
    legend_bend = [
        mpatches.Patch(color="#3498db", alpha=0.7, label="sigma(s) — tensile bending"),
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="sigma(s) — compressive bending"),
        mpatches.Patch(color="#f39c12", alpha=0.7, label="Spar cap boom"),
        mpatches.Patch(color="white",   alpha=0.8, label="Skin / web wall"),
    ]

    _save_section_figure(
        cached, airfoil,
        plot_shear=True, plot_bending=True,
        outfile=str(_plot_dir / "blade_section_distributions.png"),
        suptitle=(
            "NACA 2412 cambered multi-cell section — unit N,Vy,Vz,My,Mz,T\n"
            "Green = q(s)  |  Blue/Red = sigma(s)  |  Orange = spar cap booms"
        ),
        legend_patches=legend_full,
    )
    _save_section_figure(
        cached, airfoil,
        plot_shear=True, plot_bending=False,
        outfile=str(_plot_dir / "blade_section_shear_flow.png"),
        suptitle=(
            "NACA 2412 cambered section — Shear flow q(s) only (unit loads)\n"
            "Green / teal = sign(q); ribbon width |q| on outward normal  |  White = skin / web midline"
        ),
        legend_patches=legend_shear,
    )
    _save_section_figure(
        cached, airfoil,
        plot_shear=False, plot_bending=True,
        outfile=str(_plot_dir / "blade_section_bending_stress.png"),
        suptitle=(
            "NACA 2412 cambered section — Axial stress sigma(s) only (unit N, My, Mz)\n"
            "Blue / red = signed sigma(s) (ribbon width = |sigma|, outward)  |  Orange = spar caps"
        ),
        legend_patches=legend_bend,
    )

    # One skin panel, mid-chord station: CLPT ply σ, ε, Hashin-envelope FI
    _first = cached[0][2]
    plot_clpt_laminate_stress_fi(
        _first[0],
        _first[3],
        _first[4],
        panel_index=0,
        outfile=_plot_dir / "blade_section_clpt_fi.png",
        title=(
            "Cell 1 upper skin, mid-panel — read the text panel for what/why\n"
            "(membrane Nx = σ_xx·t, Nxy = q; strengths = SKIN_STRENGTH)"
        ),
    )