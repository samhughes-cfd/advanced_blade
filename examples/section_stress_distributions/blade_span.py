"""
Spanwise blade: scale chord-normalised 2D section by ``c(x)``, apply geometric twist
about pivot at ``pivot_y_frac * chord`` from LE, map B-frame resultants to S and call
``run_section`` at each station.

Span coordinate ``x`` runs **root → tip** on ``[0, L]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lib.blade_frames import resultants_B_to_S


def pivot_point_normalized(airfoil: np.ndarray, pivot_y_frac: float = 1.0 / 3.0) -> np.ndarray:
    """
    Pivot on the **mid-thickness** chord line: average of upper and lower ``z`` at ``y``.

    Parameters
    ----------
    airfoil : (N,2)
        Chord-normalised profile [y,z], same convention as ``naca_four_digit``.
    pivot_y_frac : float
        Chordwise fraction from LE (y=0) toward TE (y=1).
    """
    from multi_cell_blade_section import interp_surface

    yp = float(pivot_y_frac)
    pu = interp_surface(airfoil, yp, "upper")
    pl = interp_surface(airfoil, yp, "lower")
    zm = 0.5 * (float(pu[1]) + float(pl[1]))
    return np.array([yp, zm], dtype=float)


def scaled_twisted_airfoil_m(
    airfoil_norm: np.ndarray,
    chord_m: float,
    theta_geom_rad: float,
    pivot_y_frac: float = 1.0 / 3.0,
) -> np.ndarray:
    """
    Rigid rotation in the section plane about ``pivot``, then uniform scale by ``chord_m``.

    ``p' = chord_m * ( R(theta) * (p_norm - pivot) + pivot )``.
    """
    pivot = pivot_point_normalized(airfoil_norm, pivot_y_frac)
    t = float(theta_geom_rad)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s], [s, c]], dtype=float)
    out = np.zeros_like(airfoil_norm, dtype=float)
    for i in range(len(airfoil_norm)):
        p = np.array([airfoil_norm[i, 0], airfoil_norm[i, 1]], dtype=float)
        q = R @ (p - pivot) + pivot
        out[i, 0] = chord_m * q[0]
        out[i, 1] = chord_m * q[1]
    return out


@dataclass
class SpanRunConfig:
    """User inputs for a spanwise pass."""
    L: float
    x_grid: np.ndarray
    chord_m: Callable[[float], float] | np.ndarray
    theta_geom_rad: Callable[[float], float] | np.ndarray
    spar_positions: Callable[[float], list[float]] | list[float] | np.ndarray
    airfoil_norm: np.ndarray
    pivot_y_frac: float = 1.0 / 3.0


def eval_theta_geom_rad(cfg: SpanRunConfig, x_grid: np.ndarray) -> np.ndarray:
    """Geometric twist ``theta_geom(x)`` [rad] on ``x_grid`` (same rule as ``run_span_stations``)."""
    xg = np.asarray(x_grid, dtype=float).ravel()
    return _interp1d_callable(xg, cfg.theta_geom_rad)


def _interp1d_callable(xg: np.ndarray, val: Callable[[float], float] | np.ndarray) -> np.ndarray:
    if callable(val):
        return np.array([float(val(float(x))) for x in xg], dtype=float)
    v = np.asarray(val, dtype=float).ravel()
    if v.shape == xg.shape:
        return v
    raise ValueError("callable or array matching x_grid required")


def run_span_stations(
    cfg: SpanRunConfig,
    N_B: np.ndarray,
    V_edge_B: np.ndarray,
    V_flap_B: np.ndarray,
    M_edge_B: np.ndarray,
    M_flap_B: np.ndarray,
    T_B: np.ndarray,
    B_warp: np.ndarray | None = None,
):
    """
    Loop ``x_grid``; at each station build scaled/twisted airfoil, transform resultants
    B→S, call ``run_section``.

    All ``*_B`` arrays match ``cfg.x_grid`` length.

    Returns
    -------
    list of ``run_section`` return tuples in order of stations.
    """
    from multi_cell_blade_section import run_section

    xg = np.asarray(cfg.x_grid, dtype=float).ravel()
    n = len(xg)
    B_warp = np.zeros(n) if B_warp is None else np.asarray(B_warp, dtype=float).ravel()
    cfun = _interp1d_callable(xg, cfg.chord_m)
    thfun = _interp1d_callable(xg, cfg.theta_geom_rad)

    def spar_at(x: float):
        if callable(cfg.spar_positions):
            return cfg.spar_positions(x)
        return list(np.asarray(cfg.spar_positions, dtype=float).ravel())

    outs: list = []
    for i in range(n):
        c_i = float(cfun[i])
        th_i = float(thfun[i])
        af_i = scaled_twisted_airfoil_m(
            cfg.airfoil_norm, c_i, th_i, cfg.pivot_y_frac
        )
        sp = spar_at(float(xg[i]))
        N, Vy, Vz, My, Mz, T = resultants_B_to_S(
            float(N_B[i]),
            float(V_edge_B[i]),
            float(V_flap_B[i]),
            float(M_edge_B[i]),
            float(M_flap_B[i]),
            float(T_B[i]),
            th_i,
        )
        out = run_section(
            af_i,
            sp,
            N=N,
            Vy=Vy,
            Vz=Vz,
            My=My,
            Mz=Mz,
            T=T,
            B=float(B_warp[i]),
        )
        outs.append(out)
    return outs


def run_span_stations_with_airfoils(
    cfg: SpanRunConfig,
    N_B: np.ndarray,
    V_edge_B: np.ndarray,
    V_flap_B: np.ndarray,
    M_edge_B: np.ndarray,
    M_flap_B: np.ndarray,
    T_B: np.ndarray,
    B_warp: np.ndarray | None = None,
) -> list[tuple[np.ndarray, tuple]]:
    """
    Same as ``run_span_stations`` but returns ``(airfoil_m, run_section_out)`` per station
    so callers can plot geometry without recomputing scale/twist.
    """
    from multi_cell_blade_section import run_section

    xg = np.asarray(cfg.x_grid, dtype=float).ravel()
    n = len(xg)
    B_warp = np.zeros(n) if B_warp is None else np.asarray(B_warp, dtype=float).ravel()
    cfun = _interp1d_callable(xg, cfg.chord_m)
    thfun = _interp1d_callable(xg, cfg.theta_geom_rad)

    def spar_at(x: float):
        if callable(cfg.spar_positions):
            return cfg.spar_positions(x)
        return list(np.asarray(cfg.spar_positions, dtype=float).ravel())

    rows: list[tuple[np.ndarray, tuple]] = []
    for i in range(n):
        c_i = float(cfun[i])
        th_i = float(thfun[i])
        af_i = scaled_twisted_airfoil_m(
            cfg.airfoil_norm, c_i, th_i, cfg.pivot_y_frac
        )
        sp = spar_at(float(xg[i]))
        N, Vy, Vz, My, Mz, T = resultants_B_to_S(
            float(N_B[i]),
            float(V_edge_B[i]),
            float(V_flap_B[i]),
            float(M_edge_B[i]),
            float(M_flap_B[i]),
            float(T_B[i]),
            th_i,
        )
        out = run_section(
            af_i,
            sp,
            N=N,
            Vy=Vy,
            Vz=Vz,
            My=My,
            Mz=Mz,
            T=T,
            B=float(B_warp[i]),
        )
        rows.append((af_i, out))
    return rows


if __name__ == "__main__":
    from lib.beam_vlasov_1d import solve_span_equilibrium
    from multi_cell_blade_section import naca_four_digit

    L = 0.6
    x = np.linspace(0.0, L, 21)
    af = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=100)
    n = len(x)
    qx = np.zeros(n)
    p_edge = 50.0 * (1.0 - x / L)
    p_flap = np.zeros(n)
    mx = np.zeros(n)
    EIw = 2e3 * np.ones(n)
    GJ = 8e3 * np.ones(n)
    eq = solve_span_equilibrium(x, qx, p_edge, p_flap, mx, EIw, GJ)
    cfg = SpanRunConfig(
        L=L,
        x_grid=x,
        chord_m=0.25 * (1.0 - 0.4 * x / L),
        theta_geom_rad=0.15 * (x / L),
        spar_positions=[0.35],
        airfoil_norm=af,
    )
    run_span_stations(
        cfg,
        eq["N"],
        eq["V_edge"],
        eq["V_flap"],
        eq["M_edge"],
        eq["M_flap"],
        eq["T"],
        eq["B"],
    )
    print("blade_span demo: OK", n, "stations")
