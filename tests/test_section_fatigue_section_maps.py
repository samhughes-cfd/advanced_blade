"""Section-plane fatigue map PNGs for the sinusoid example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from blade_analysis.fatigue_damage import FatigueAnalysis
from blade_analysis.fatigue_damage._smoke_fixtures import (
    build_smoke_recovery_cache_and_ref_section,
    default_fatigue_sn_curves,
    smoke_sinusoidal_resultant_history,
)


def test_fatigue_section_maps_write_png(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    ex = root / "examples" / "section_fatigue_sinusoid"
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(ex))

    from lib.section_maps import save_fatigue_damage_section_map, save_fatigue_life_section_map

    cache, ref_section = build_smoke_recovery_cache_and_ref_section()
    z = np.asarray(cache.z_stations, dtype=np.float64)
    hist = smoke_sinusoidal_resultant_history(
        z,
        n_t=24,
        t_end=0.4,
        f_hz=2.0,
        amplitude=3.0e3,
        load_component="My",
        spanwise_envelope=True,
    )
    res = FatigueAnalysis.from_cache(
        cache, default_fatigue_sn_curves(), design_life_years=25.0
    ).run(hist, memory_limit_mb=64.0)

    s = int(res.worst_composite[0])
    p_dmg = tmp_path / "fatigue_damage_section_map.png"
    p_life = tmp_path / "fatigue_life_section_map.png"
    save_fatigue_damage_section_map(p_dmg, cache, ref_section, res, z, s)
    save_fatigue_life_section_map(p_life, cache, ref_section, res, z, s)

    assert p_dmg.is_file() and p_dmg.stat().st_size > 200
    assert p_life.is_file() and p_life.stat().st_size > 200
