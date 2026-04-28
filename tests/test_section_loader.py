"""Section spec round-trip."""
import json
import tempfile
from pathlib import Path

from blade_precompute.section_properties.io.section_loader import load_section_from_spec
from blade_precompute.section_properties.engine.solver import MidsurfaceSectionSolver


def test_load_section_spec():
    doc = {
        "station_z": 1.5,
        "ply_library": {
            "ud": {
                "E1": 40e9,
                "E2": 10e9,
                "G12": 4e9,
                "nu12": 0.28,
                "rho": 1900.0,
                "t_ply": 0.0002,
                "Xt": 1e9,
                "Xc": 1e9,
                "Yt": 1e9,
                "Yc": 1e9,
                "S12": 1e9,
                "Zt": 50e6,
                "S13": 40e6,
                "S23": 40e6,
            }
        },
        "materials": {
            "GFRP_laminate": {"ply_type": "ud", "layup": [0, 45, -45, 0]},
        },
        "subcomponents": {
            "skin": {
                "midsurface_coords": [[0.0, 0.0], [0.1, 0.0]],
                "thickness": 0.001,
                "strip_width_m": 0.03,
                "material": "GFRP_laminate",
            },
            "metal": {
                "midsurface_coords": [[0.1, 0.0], [0.12, 0.02]],
                "thickness": 0.002,
                "material": "aluminium_6082",
                "E": 70e9,
                "nu": 0.33,
                "rho": 2700.0,
                "sigma_allow": 270e6,
            },
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sec.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        sec = load_section_from_spec(p)
        assert len(sec.subcomponents) == 2
        res = MidsurfaceSectionSolver().solve_one(sec)
        assert res.K7.shape == (7, 7)
