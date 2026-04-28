"""
Shared test/demo fixtures: small midsurface section model → :class:`RecoveryCache`,
and sinusoidal :class:`ResultantHistory` for the fatigue stack.

Not part of the public API; used by :mod:`blade_analysis.fatigue_damage.__main__` and
``examples/section_fatigue_sinusoid`` to keep smoke paths aligned.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from blade_precompute.global_beam_model.engine.kinematics import rotmat_from_small_curvature
from blade_precompute.section_optimisation.core.types import DesignVector, OptimBladeGeometry
from blade_precompute.section_optimisation.engine.section_builder import SectionBuilder
from blade_precompute.section_properties.engine.geometry import SectionDefinition
from blade_precompute.section_properties.engine.laminate import LaminateDefinition
from blade_precompute.section_properties.engine.materials import IsotropicMaterial, OrthotropicPly
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver
from blade_utilities.recovery import RecoveryCache, RecoveryCacheBuilder

from .engine.sn_curves import SNcurve
from .core.loads import ResultantHistory

_LoadComponent = Literal["N", "Vy", "Vz", "My", "Mz", "T", "B"]


def _ply_gfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="gfrp",
        E1=42e9,
        E2=12e9,
        G12=4.5e9,
        nu12=0.28,
        rho=1900.0,
        t_ply=t_ply,
        Xt=900e6,
        Xc=650e6,
        Yt=65e6,
        Yc=120e6,
        S12=75e6,
        Zt=45e6,
        S13=40e6,
        S23=40e6,
    )


def _ply_cfrp(t_ply: float) -> OrthotropicPly:
    return OrthotropicPly(
        name="cfrp",
        E1=135e9,
        E2=9e9,
        G12=4.8e9,
        nu12=0.28,
        rho=1600.0,
        t_ply=t_ply,
        Xt=1800e6,
        Xc=1200e6,
        Yt=60e6,
        Yc=220e6,
        S12=90e6,
        Zt=55e6,
        S13=45e6,
        S23=45e6,
    )


def _build_smoke_sections_and_cache() -> tuple[RecoveryCache, list[SectionDefinition]]:
    """Internal: compact midsurface model used by CLI, tests, and section-fatigue example."""
    n_s = 3
    z = np.linspace(0.0, 4.0, n_s, dtype=np.float64)
    L = float(z[-1])
    r_ref = np.zeros((n_s, 3), dtype=np.float64)
    r_ref[:, 2] = z
    r_ref[:, 1] = 0.012 * (z / max(L, 1e-12)) ** 2
    kappa0 = np.zeros((n_s, 3), dtype=np.float64)
    kappa0[:, 1] = 0.0012
    chord = np.linspace(1.8, 1.2, n_s, dtype=np.float64)
    twist = np.zeros_like(z)
    web_positions = np.array([-0.32, 0.32], dtype=np.float64)
    t0 = 0.0002
    lam_skin = LaminateDefinition(
        plies=[
            (_ply_gfrp(t0), 0.0),
            (_ply_gfrp(t0), 45.0),
            (_ply_gfrp(t0), -45.0),
        ],
        shear_lag_correction=True,
    )
    lam_cap = LaminateDefinition(plies=[(_ply_cfrp(t0), 0.0)] * 4, shear_lag_correction=True)
    lam_web = LaminateDefinition(
        plies=[(_ply_gfrp(t0), 45.0), (_ply_gfrp(t0), -45.0)],
        shear_lag_correction=True,
    )
    al = IsotropicMaterial(
        name="al6082",
        E=70e9,
        nu=0.33,
        rho=2700.0,
        sigma_allow=260e6,
    )
    bg = OptimBladeGeometry(
        z_stations=z,
        r_ref=r_ref,
        kappa0=kappa0,
        chord=chord,
        twist=twist,
        airfoil_profiles=[],
        web_positions=web_positions,
        subcomponent_materials={
            "skin": lam_skin,
            "cap_ps": lam_cap,
            "web": lam_web,
            "leading_edge_insert": al,
        },
        thickness_role={"leading_edge_insert": "fixed"},
        box_height_frac=0.11,
    )
    dv = DesignVector(
        t_skin=np.full(n_s, 0.010),
        t_cap=np.full(n_s, 0.040),
        t_web=np.full(n_s, 0.012),
    )
    sections = SectionBuilder.build(dv, bg)
    solver = MidsurfaceSectionSolver()
    section_results = [solver.solve_one(s) for s in sections]
    nodal_R = np.stack([rotmat_from_small_curvature(bg.kappa0[i]) for i in range(n_s)], axis=0)
    storage = RecoveryCacheBuilder.build(
        section_results,
        sections[0].subcomponents,
        bg.z_stations,
        nodal_R_stack=nodal_R,
    )
    return RecoveryCache(**storage.__dict__), sections


def build_smoke_recovery_cache() -> RecoveryCache:
    """Same compact geometry as the fatigue CLI and recovery smoke examples."""
    cache, _ = _build_smoke_sections_and_cache()
    return cache


def build_smoke_recovery_cache_and_ref_section() -> tuple[RecoveryCache, SectionDefinition]:
    """
    Same cache as :func:`build_smoke_recovery_cache` plus the reference-station section definition.

    Use ``ref_section.subcomponents[*].midsurface_coords`` for section-plane maps. User-supplied
    ``.npz`` caches do not carry this geometry.
    """
    cache, sections = _build_smoke_sections_and_cache()
    return cache, sections[0]


def default_fatigue_sn_curves() -> dict[str, SNcurve]:
    return {
        "GFRP": SNcurve.gfrp_blade(),
        "CFRP": SNcurve.cfrp_blade(),
        "default": SNcurve.gfrp_blade(),
        "aluminium": SNcurve.steel_dnv(),
    }


def smoke_sinusoidal_resultant_history(
    z_stations: NDArray[np.float64],
    *,
    n_t: int = 256,
    t_end: float = 1.0,
    f_hz: float = 7.0,
    amplitude: float = 5.0e3,
    load_component: _LoadComponent = "My",
    spanwise_envelope: bool = True,
) -> ResultantHistory:
    """
    Dummy harmonic resultant history: one component = ``A * sin(2π f t)`` (optional spanwise ramp from root to tip).

    All other beam channels are zero.
    """
    z1 = np.asarray(z_stations, dtype=np.float64)
    n_s = int(z1.shape[0])
    t = np.linspace(0.0, t_end, int(n_t), dtype=np.float64)
    wave = float(amplitude) * np.sin(2.0 * np.pi * float(f_hz) * t)[:, None]
    if spanwise_envelope:
        zf = z1.reshape(1, n_s) / max(float(z1[-1]), 1e-12)
        field = wave * (0.5 + 0.5 * zf)
    else:
        field = np.broadcast_to(wave, (t.shape[0], n_s)).copy()
    z = np.zeros((t.shape[0], n_s), dtype=np.float64)
    channels: dict[_LoadComponent, NDArray[np.float64]] = {
        "N": z.copy(),
        "Vy": z.copy(),
        "Vz": z.copy(),
        "My": z.copy(),
        "Mz": z.copy(),
        "T": z.copy(),
        "B": z.copy(),
    }
    channels[load_component] = field
    return ResultantHistory(
        z_stations=z1,
        time=t,
        N=channels["N"],
        Vy=channels["Vy"],
        Vz=channels["Vz"],
        My=channels["My"],
        Mz=channels["Mz"],
        T=channels["T"],
        B=channels["B"],
    )
