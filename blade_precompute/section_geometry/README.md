# implicit_section_geometry

SDF-based implicit computational geometry for wind/tidal turbine blade cross-sections.

Midlines, midsurfaces, and medial axes of each structural subcomponent are recovered
**by construction** from the SDF skeleton theorem — no heuristic skeletonisation needed.

---

## Package structure

```text
implicit_section_geometry/
├── geometry/
│   ├── primitives.py     # Exact SDF primitives (circle, box, polygon, …)
│   ├── csg.py            # CSG operations (union, intersect, subtract, offset, shell)
│   ├── airfoil.py        # AirfoilSDF: discrete polyline → SDF + NACA generator
│   └── grid.py           # SDFGrid: structured evaluation, gradient, quadrature
│
├── sections/
│   ├── subcomponents.py  # OuterSkin, SparCap, ShearWeb, SandwichCore
│   └── section.py        # BladeSectionGeometry: full assembly
│
├── medial/
│   └── extractor.py      # MedialAxisExtractor + extract_midline()
│
├── interface/
│   ├── export.py         # SectionPropertiesReport, CSV/JSON export
│   └── plot.py           # Matplotlib visualisation helpers
│
├── tests/
│   ├── test_primitives.py
│   ├── test_csg.py
│   └── test_airfoil.py
│
└── examples/
    └── naca2412_section.py   # End-to-end demonstration
```

---

## Installation

```bash
cd blade_precompute/section_geometry
pip install -r requirements.txt
```

---

## Quick start

```python
from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
from blade_precompute.section_geometry.geometry.grid import SDFGrid
from blade_precompute.section_geometry.sections import BladeSectionGeometry
from blade_precompute.section_geometry.medial import extract_midline
from blade_precompute.section_geometry.interface import plot_section, SectionPropertiesReport

# 1. Airfoil
airfoil = AirfoilSDF.from_naca("2412", chord=1.0)

# — or load from a .dat file —
# airfoil = AirfoilSDF.from_dat("my_section.dat")

# 2. Assemble section (uses defaults; pass config dict to customise)
bsg = BladeSectionGeometry(airfoil)

# 3. Evaluation grid
grid = SDFGrid.from_airfoil(airfoil, nx=512, ny=200)

# 4. Medial axis of the outer skin
midlines = extract_midline(bsg["outer_skin"], grid)

# 5. Section properties for all components
report = SectionPropertiesReport(bsg, grid)
print(report.summary())

# 6. Plot
fig, ax = plot_section(bsg, grid)
fig.savefig("section.png", dpi=150)
```

---

## Running the example

```bash
python examples/naca2412_section.py
```

Outputs are written to `examples/output/`.

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Theory

The medial axis is the locus of points equidistant from two or more boundary
features.  For an SDF field φ with |∇φ| = 1 (Eikonal equation), the medial axis
corresponds to points where the gradient magnitude drops below 1 — because at
such points the closest-point map is multi-valued.

Extraction pipeline:

1. Evaluate φ on a structured grid.
2. Compute |∇φ| via central finite differences.
3. Threshold interior points where |∇φ| < ε (default 0.92).
4. Apply morphological thinning (Lee's algorithm, `skimage.morphology.skeletonize`).
5. Prune short branches.
6. Order skeleton pixels into (x, y) polylines.

Optional Eikonal redistancing (`scikit-fmm`) re-normalises φ before extraction,
improving quality for fields that have drifted after repeated CSG operations.

---

## Configuration schema for BladeSectionGeometry

```python
config = {
    "skin_thickness": 0.003,          # metres (or chord fraction for unit chord)

    "spar_cap": {
        "x_start": 0.20,              # chordwise start
        "x_end":   0.45,              # chordwise end
        "height":  0.012,             # cap thickness
        "y_inner_offset": 0.0,        # gap between cap and neutral axis
    },

    "shear_webs": [
        {
            "x_top": 0.20, "y_top":  0.055,
            "x_bot": 0.20, "y_bot": -0.035,
            "thickness": 0.004,
        },
        # add more webs as needed …
    ],

    "core": {
        "enabled": True,
        "x_start": 0.20,              # None → full chord
        "x_end":   0.45,
    },
}
```
