"""
Parametric NACA section generators (4-, 5-, and 6-series) for ``AirfoilSDF`` construction.

Five-digit camber/thickness conventions follow Abbott & von Doenhoff / NASA summaries
(e.g. aerospaceweb.org NACA overview).  Six-series profiles for families **63** and **64**
use embedded UIUC Airfoil Data Site coordinates (unit chord), with linear thickness scaling
in *y* when ``t_percent`` differs from the reference section thickness.

References: NACA TR 610 (five-digit); UIUC airfoil coordinate database (six-series .dat).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# (chordwise ref r, max camber m, k1) for NACA five-digit simple (S=0) camber — NASA table.
_NACA5_RMK1: NDArray[np.float64] = np.array(
    [
        [0.05, 0.02000, 361.400],
        [0.08, 0.01700, 330.700],
        [0.10, 0.01500, 310.100],
        [0.15, 0.01250, 270.900],
        [0.20, 0.01100, 251.100],
        [0.25, 0.01100, 231.800],
        [0.30, 0.01100, 212.100],
        [0.35, 0.01100, 193.100],
        [0.40, 0.01000, 174.100],
        [0.45, 0.00900, 155.300],
        [0.50, 0.00800, 136.600],
        [0.55, 0.00700, 118.300],
        [0.60, 0.00600, 100.300],
        [0.65, 0.00500, 82.800],
        [0.70, 0.00400, 66.100],
        [0.75, 0.00300, 50.500],
        [0.80, 0.00200, 36.100],
        [0.85, 0.00100, 23.300],
        [0.90, 0.00050, 11.700],
        [0.95, 0.00025, 3.500],
    ],
    dtype=np.float64,
)


def _thickness_yt_4digit(
    xc: NDArray[np.float64], t_frac: float, *, closed_te: bool
) -> NDArray[np.float64]:
    a4 = -0.1015 if closed_te else -0.1036
    a = np.array([0.2969, -0.1260, -0.3516, 0.2843, a4], dtype=np.float64)
    return (t_frac / 0.2) * (
        a[0] * np.sqrt(np.maximum(xc, 0.0))
        + a[1] * xc
        + a[2] * xc**2
        + a[3] * xc**3
        + a[4] * xc**4
    )


def _wrap_upper_lower(
    xc: NDArray[np.float64],
    yc: NDArray[np.float64],
    dyc: NDArray[np.float64],
    yt: NDArray[np.float64],
    chord: float,
) -> NDArray[np.float64]:
    theta = np.arctan(dyc)
    xu = (xc - yt * np.sin(theta)) * chord
    yu = (yc + yt * np.cos(theta)) * chord
    xl = (xc + yt * np.sin(theta)) * chord
    yl = (yc - yt * np.cos(theta)) * chord
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    return np.vstack([upper, lower])


def naca_four_digit_vertices(
    m: float,
    p: float,
    t_frac: float,
    n_points: int,
    chord: float,
    *,
    closed_te: bool = True,
) -> NDArray[np.float64]:
    """Return (N,2) vertices for classic NACA 4-digit section (``m``, ``p`` as fractions)."""
    beta = np.linspace(0.0, math.pi, max(4, n_points // 2 + 1), dtype=np.float64)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = _thickness_yt_4digit(xc, float(t_frac), closed_te=closed_te)
    m = float(m)
    p = float(p)
    if m <= 0.0 or p <= 0.0:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
    else:
        yc = np.where(
            xc <= p,
            (m / p**2) * (2 * p * xc - xc**2),
            (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc - xc**2),
        )
        dyc = np.where(
            xc <= p,
            (2 * m / p**2) * (p - xc),
            (2 * m / (1 - p) ** 2) * (p - xc),
        )
    return _wrap_upper_lower(xc, yc, dyc, yt, float(chord))


def _interp_naca5_table(r: float) -> tuple[float, float]:
    """Return (m_max, k1) for five-digit camber at reference chord fraction ``r``."""
    r = float(np.clip(r, _NACA5_RMK1[0, 0], _NACA5_RMK1[-1, 0]))
    rr = _NACA5_RMK1[:, 0]
    m = float(np.interp(r, rr, _NACA5_RMK1[:, 1]))
    k1 = float(np.interp(r, rr, _NACA5_RMK1[:, 2]))
    return m, k1


def naca_five_digit_vertices(
    L: int,
    PQ: int,
    TT: int,
    n_points: int,
    chord: float,
    *,
    closed_te: bool = True,
    reflex: bool = False,
) -> NDArray[np.float64]:
    """
    NACA five-digit LPQTT simple camber (S=0), same thickness law as 4-digit.

    ``L`` : first digit (design C_L,i = (3/20) * L in absolute lift).
    ``PQ``: two-digit integer (10*P+Q) giving camber reference r = PQ/200.
    ``TT``: thickness percent (e.g. 12 → 12%).
    """
    if reflex:
        raise NotImplementedError("NACA five-digit reflex (S=1) camber is not implemented.")
    if L < 1 or L > 9:
        raise ValueError(f"Five-digit L must be 1..9, got {L}.")
    if PQ < 0 or PQ > 99:
        raise ValueError(f"Five-digit PQ must be 0..99, got {PQ}.")
    if TT < 1 or TT > 99:
        raise ValueError(f"Five-digit TT must be 1..99, got {TT}.")

    r = float(PQ) / 200.0
    m_tab, k1 = _interp_naca5_table(r)
    cli = (3.0 / 20.0) * float(L)
    lift_scale = cli / 0.3

    beta = np.linspace(0.0, math.pi, max(4, n_points // 2 + 1), dtype=np.float64)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = _thickness_yt_4digit(xc, float(TT) / 100.0, closed_te=closed_te)

    yc = np.zeros_like(xc)
    dyc = np.zeros_like(xc)
    mask0 = xc <= r
    mask1 = ~mask0
    yc = np.where(
        mask0,
        k1 / 6.0 * (xc**3 - 3.0 * r * xc**2 + r**2 * (3.0 - r) * xc),
        yc,
    )
    yc = np.where(mask1, k1 * r**3 / 6.0 * (1.0 - xc), yc)
    dyc = np.where(
        mask0,
        k1 / 6.0 * (3.0 * xc**2 - 6.0 * r * xc + r**2 * (3.0 - r)),
        dyc,
    )
    dyc = np.where(mask1, -k1 * r**3 / 6.0, dyc)
    yc = yc * m_tab * lift_scale
    dyc = dyc * m_tab * lift_scale

    return _wrap_upper_lower(xc, yc, dyc, yt, float(chord))


def _parse_uiuc_ads_flat_dat(text: str) -> NDArray[np.float64]:
    """Parse UIUC 'flat' coordinate dump (numbers after header line)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    nums: list[float] = []
    for ln in lines:
        if re.match(r"^NACA", ln, re.I):
            # Header and grid counts (e.g. ``26. 26.'') live on the same line as x,y pairs.
            m = re.search(r"0\.0+\s+0\.0+", ln)
            if m:
                ln = ln[m.start() :]
            else:
                continue
        parts = ln.replace(",", " ").split()
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
    if len(nums) < 8 or len(nums) % 2 != 0:
        raise ValueError("UIUC flat dat: expected an even count of numeric tokens.")
    arr = np.asarray(nums, dtype=np.float64).reshape(-1, 2)
    return arr


def _resample_polyline(xy: NDArray[np.float64], n_target: int) -> NDArray[np.float64]:
    """Evenly resample closed polyline by arc length (``n_target`` total vertices)."""
    if xy.shape[0] < 3 or n_target < 6:
        return xy
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total <= 0.0:
        return xy
    s_new = np.linspace(0.0, total, n_target, endpoint=False, dtype=np.float64)
    x = np.interp(s_new, s, xy[:, 0])
    y = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x, y])


def _embed_n63415() -> str:
    # UIUC ADS: https://m-selig.ae.illinois.edu/ads/coord/n63415.dat
    return """NACA 63-415 AIRFOIL 26. 26. 0.000000 0.000000 0.003000 0.012870 0.005250 0.015850 0.009910 0.020740 0.021980 0.029640 0.046600 0.042640 0.071470 0.052610 0.096470 0.060770 0.146690 0.073480 0.197050 0.082790 0.247500 0.089410 0.298000 0.093620 0.348520 0.095590 0.399050 0.095270 0.449550 0.092890 0.500000 0.088710 0.550390 0.082980 0.600700 0.075950 0.650930 0.067800 0.701060 0.058770 0.751090 0.049070 0.801020 0.039000 0.850850 0.028850 0.900590 0.018840 0.950280 0.009310 1.000000 0.000000 0.000000 0.000000 0.007000 -0.010870 0.009750 -0.013050 0.015090 -0.016460 0.028020 -0.022200 0.053400 -0.030000 0.078530 -0.035650 0.103530 -0.040090 0.153310 -0.046560 0.202950 -0.050950 0.252500 -0.053610 0.302000 -0.054740 0.351480 -0.054390 0.400950 -0.052430 0.450450 -0.049090 0.500000 -0.044590 0.549610 -0.039180 0.599300 -0.033110 0.649070 -0.026600 0.698940 -0.019890 0.748910 -0.013270 0.799890 -0.007160 0.849150 -0.001930 0.899410 0.001840 0.949720 0.003330 1.000000 0.000000"""


def _embed_n64212() -> str:
    # UIUC ADS: https://m-selig.ae.illinois.edu/ads/coord/n64212.dat
    return """NACA 64(1)-212 26.0 26.0 0.0000000 0.0000000 0.0041800 0.0102500 0.0065900 0.0124500 0.0114700 0.0159300 0.0238200 0.0221800 0.0486800 0.0312300 0.0736400 0.0381500 0.0986500 0.0438600 0.1487200 0.0529100 0.1988600 0.0596800 0.2490300 0.0647000 0.2992100 0.0681500 0.3494100 0.0700800 0.3996100 0.0705200 0.4498200 0.0689300 0.5000000 0.0658300 0.5501600 0.0615100 0.6002900 0.0561900 0.6503900 0.0500400 0.7004500 0.0432200 0.7504700 0.0359000 0.8004500 0.0282500 0.8503800 0.0205400 0.9002700 0.0130300 0.9501300 0.0060400 1.0000000 0.0000000 0.0000000 0.0000000 0.0058200 -0.0092500 0.0084100 -0.0110500 0.0135300 -0.0137900 0.0261800 -0.0184600 0.0513200 -0.0249100 0.0763600 -0.0296700 0.1013500 -0.0335200 0.1512800 -0.0394500 0.2011400 -0.0437600 0.2509700 -0.0468000 0.3007900 -0.0487100 0.3505900 -0.0494800 0.4003900 -0.0491000 0.4501800 -0.0470300 0.5000000 -0.0437700 0.5498400 -0.0396100 0.5997100 -0.0347700 0.6496100 -0.0294400 0.6995500 -0.0237800 0.7495300 -0.0180000 0.7995500 -0.0123300 0.8496200 -0.0070800 0.8997300 -0.0026900 0.9498700 0.0002800 1.0000000 0.0000000"""


def _six_series_reference_xy(family: int) -> NDArray[np.float64]:
    if family == 63:
        raw = _parse_uiuc_ads_flat_dat(_embed_n63415())
    elif family == 64:
        raw = _parse_uiuc_ads_flat_dat(_embed_n64212())
    else:
        raise ValueError(f"Unsupported 6-series family (tens digit 6): {family}.")
    n = raw.shape[0] // 2
    upper = raw[:n]
    lower = raw[n:]
    upper_o = upper[np.argsort(upper[:, 0])][::-1]
    lower_o = lower[np.argsort(lower[:, 0])]
    return np.vstack([upper_o, lower_o[1:]])


def naca_six_series_vertices(
    family: int,
    t_percent: float,
    n_points: int,
    chord: float,
    *,
    closed_te: bool = True,
    design_cl_tenth: float | None = None,
) -> NDArray[np.float64]:
    """
    Six-series airfoil from embedded UIUC coordinates (families **63** and **64**).

    ``family`` is the two-digit code (e.g. **63**, **64**).  ``t_percent`` scales
    *y* about ``y=0`` relative to the reference section thickness (~12% for n64212,
    ~15% for n63415).  ``design_cl_tenth`` is accepted for API compatibility with
    spanwise inputs but does not yet remap to a different camber line (reference
    airfoil is fixed).
    """
    _ = design_cl_tenth
    xy = _six_series_reference_xy(int(family)).astype(np.float64)
    t_ref = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
    if t_ref <= 1e-9:
        raise ValueError("Reference six-series airfoil has near-zero thickness.")
    scale_y = float(t_percent) / 100.0 / (t_ref / 1.0)
    xy[:, 1] *= scale_y
    xy *= float(chord)
    if closed_te:
        pass
    out = _resample_polyline(xy, max(32, int(n_points)))
    return out


def airfoil_vertices_from_spanwise(
    series: Literal[4, 5, 6],
    m: float,
    p: float,
    xx: float,
    n_points: int,
    chord: float,
    *,
    closed_te: bool = True,
) -> NDArray[np.float64]:
    """
    Dispatch NACA geometry from precompute ``naca_series`` + ``(naca_m, naca_p, naca_xx)``.

    **Series 4:** ``m`` and ``p`` are 0–9 digits (percent / tenths as in ``naca4``),
    ``xx`` is thickness percent.

    **Series 5:** ``m`` = L (1–9), ``p`` = PQ two-digit integer (0–99), ``xx`` = TT.

    **Series 6:** ``m`` = two-digit family (**63**, **64**, …), ``p`` = design C_L × 10
    (informational; geometry uses embedded reference for that family), ``xx`` = thickness %%.
    """
    s = int(series)
    key = (
        s,
        float(round(float(m), 10)),
        float(round(float(p), 10)),
        float(round(float(xx), 10)),
        int(n_points),
        float(round(float(chord), 10)),
        bool(closed_te),
    )
    return _airfoil_vertices_from_spanwise_cached(*key).copy()


@lru_cache(maxsize=512)
def _airfoil_vertices_from_spanwise_cached(
    series: int,
    m: float,
    p: float,
    xx: float,
    n_points: int,
    chord: float,
    closed_te: bool,
) -> NDArray[np.float64]:
    s = int(series)
    if s == 4:
        mi = int(np.clip(int(round(float(m))), 0, 9))
        pi = int(np.clip(int(round(float(p))), 0, 9))
        xxi = int(np.clip(int(round(float(xx))), 0, 99))
        return naca_four_digit_vertices(
            mi / 100.0, pi / 10.0, xxi / 100.0, n_points, chord, closed_te=closed_te
        )
    if s == 5:
        L = int(np.clip(int(round(float(m))), 1, 9))
        PQ = int(np.clip(int(round(float(p))), 0, 99))
        TT = int(np.clip(int(round(float(xx))), 1, 99))
        return naca_five_digit_vertices(L, PQ, TT, n_points, chord, closed_te=closed_te)
    if s == 6:
        fam = int(round(float(m)))
        if fam not in (63, 64):
            raise ValueError(
                f"Six-series family {fam} is not supported (embed UIUC data for 63 or 64)."
            )
        cli_t = float(p)
        t_pct = float(xx)
        return naca_six_series_vertices(
            fam,
            t_pct,
            n_points,
            chord,
            closed_te=closed_te,
            design_cl_tenth=cli_t,
        )
    raise ValueError(f"Unsupported NACA series {series!r} (expected 4, 5, or 6).")


def spanwise_airfoil_label(series: int, m: float, p: float, xx: float) -> str:
    """Short label for logs / plots."""
    s = int(series)
    if s == 4:
        mi = int(np.clip(int(round(float(m))), 0, 9))
        pi = int(np.clip(int(round(float(p))), 0, 9))
        xxi = int(np.clip(int(round(float(xx))), 0, 99))
        return f"{mi:d}{pi:d}{xxi:02d}"
    if s == 5:
        L = int(np.clip(int(round(float(m))), 1, 9))
        PQ = int(np.clip(int(round(float(p))), 0, 99))
        TT = int(np.clip(int(round(float(xx))), 1, 99))
        return f"{L:d}{PQ:02d}{TT:02d}"
    if s == 6:
        fam = int(round(float(m)))
        clt = int(np.clip(int(round(float(p))), 0, 9))
        t = int(np.clip(int(round(float(xx))), 1, 99))
        return f"{fam}-{clt}{t:02d}"
    return f"series{s}"
