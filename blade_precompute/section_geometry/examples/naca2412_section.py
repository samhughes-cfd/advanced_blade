"""
examples/naca2412_section.py
============================
End-to-end demonstration of the implicit_section_geometry package.

Demonstrates:
  1.  NACA 2412 airfoil SDF
  2.  Classic twin-web box spar (corrected curved spar caps)
  3.  3-web 4-cell torsion box at 15° section twist
      — forward web: chord-normal
      — mid web:     flapwise-aligned
      — aft web:     chord-normal
  4.  Medial axis extraction for all subcomponents
  5.  Section properties report
  6.  Comparison plots: curved vs flat cap construction

Run from the package root:
    python examples/naca2412_section.py

Requirements: numpy, scipy, matplotlib, scikit-image
Optional:     scikit-fmm  (Eikonal redistancing)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from blade_precompute.section_geometry import SectionGeometryAnalysis
from blade_precompute.section_geometry.engine.implicit_section_geometry import (
    AirfoilSDF,
    MedialAxisExtractor,
    MultiCellSection,
    SDFGrid,
)
from blade_precompute.section_geometry.interface import (
    SectionPropertiesReport,
    export_midlines_csv,
    export_section_json,
    plot_medial_axes,
    plot_sdf_field,
    plot_section,
)
from blade_precompute.section_geometry.interface.plot import plot_grad_magnitude

OUTPUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT, exist_ok=True)


# =============================================================================
# 1. Airfoil
# =============================================================================

print("=" * 60)
print("  NACA 2412 Blade Section Geometry Demo")
print("=" * 60)

analysis = SectionGeometryAnalysis()
af = analysis.build_airfoil("2412", chord=1.0)
print(f"\nAirfoil: {af}")

xc, yc = af.camber_line(n_points=100)
xc_t, t_dist = af.thickness_distribution(n_points=100)
print(f"  Max thickness : {t_dist.max():.4f}c at x/c = {xc_t[np.argmax(t_dist)]:.3f}")
print(f"  Max camber    : {yc.max():.4f}c  at x/c = {xc[np.argmax(yc)]:.3f}")


# =============================================================================
# 2. Twin-web box spar (corrected curved spar caps)
# =============================================================================

print("\n─── Twin-web box spar (0 twist) ─────────────────────────")

twin = MultiCellSection.twin_web(
    af,
    x_fore         = 0.20,
    x_aft          = 0.50,
    skin_thickness = 0.003,
    web_thickness  = 0.004,
    cap_height     = (0.014, 0.012),    # (upper, lower)
    web_alignment  = "chord_normal",
    twist_angle    = 0.0,
    te_insert_x    = 0.75,
    le_insert_x    = 0.10,
    core_enabled   = True,
)
print(f"  Components: {twin.labels}")
print(f"  n_webs={twin.n_webs}, n_cells={twin.n_cells}")

grid_twin = SDFGrid.from_airfoil(af, padding=0.08, nx=600, ny=240)

fig, ax = plot_section(twin, grid_twin,
                       title="NACA 2412 — twin-web box spar (corrected curved caps)")
ax.plot(xc, yc, "k--", lw=1.2, label="camber line")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, "twin_web_section.png"), dpi=150)
plt.close(fig)
print("  Saved: output/twin_web_section.png")


# =============================================================================
# 3. Three-web torsion box at 15° twist with mixed web alignment
# =============================================================================

print("\n─── 3-web torsion box (15° twist, mixed web alignment) ──")

TWIST = np.radians(15)

torsion = MultiCellSection(
    airfoil_sdf     = af,
    web_x_positions = [0.15, 0.35, 0.55],
    web_thickness   = [0.005, 0.004, 0.005],
    web_alignment   = ["chord_normal", "flapwise", "chord_normal"],
    cap_height      = (0.014, 0.012),
    skin_thickness  = 0.003,
    twist_angle     = TWIST,
    te_insert_x     = 0.80,
    le_insert_x     = 0.08,
    core_enabled    = True,
)
print(f"  Components : {torsion.labels}")
print(f"  n_webs={torsion.n_webs}, n_cells={torsion.n_cells}")
print(f"  Twist      : {np.degrees(TWIST):.1f}°")

grid_tors = SDFGrid.from_airfoil(af, padding=0.08, nx=700, ny=280)

fig, ax = plot_section(torsion, grid_tors,
                       title=f"NACA 2412 — 3-web torsion box @ {np.degrees(TWIST):.0f}° twist\n"
                             "(web 0: chord-normal | web 1: flapwise | web 2: chord-normal)")
ax.plot(xc, yc, "k--", lw=1.2, label="camber line (analytical)")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, "torsion_box_section.png"), dpi=150)
plt.close(fig)
print("  Saved: output/torsion_box_section.png")


# =============================================================================
# 4. Medial axes
# =============================================================================

print("\n─── Medial axis extraction ───────────────────────────────")

extractor = MedialAxisExtractor(
    grid_tors,
    grad_threshold    = 0.92,
    redistance        = False,
    min_branch_pixels = 8,
)
midlines = extractor.extract_for_section(torsion)

for label, polys in midlines.items():
    pts = sum(len(p) for p in polys)
    arc = sum(extractor.midline_length(p) for p in polys)
    print(f"  {label:25s}: {len(polys)} branch(es), {pts} pts, L={arc:.4f} m")

fig, ax = plot_medial_axes(
    midlines, grid=grid_tors, section_geometry=torsion,
    alpha_bg=0.20,
    title=f"NACA 2412 — medial axes (3-web, {np.degrees(TWIST):.0f}° twist)",
)
ax.plot(xc, yc, "k--", lw=1.5, label="camber line")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, "torsion_box_medial_axes.png"), dpi=150)
plt.close(fig)
print("  Saved: output/torsion_box_medial_axes.png")


# =============================================================================
# 5. Section properties
# =============================================================================

print("\n─── Section properties ───────────────────────────────────")

report_twin   = SectionPropertiesReport(twin,    grid_twin)
report_torsion = SectionPropertiesReport(torsion, grid_tors)

print("\nTwin-web box spar:")
print(report_twin.summary())

print("\n3-web torsion box (15° twist):")
print(report_torsion.summary())


# =============================================================================
# 6. |∇φ| maps — medial axis locus visible as dark band
# =============================================================================

print("\n─── |∇φ| diagnostic plots ────────────────────────────────")

for label in ["outer_skin", "spar_cap_upper", "web_1"]:
    phi = grid_tors.eval(torsion[label])
    fig, ax = plot_grad_magnitude(phi, grid_tors,
                                  title=f"|∇φ|  [{label}]  — medial axis = dark band")
    fig.tight_layout()
    safe_label = label.replace(" ", "_")
    path = os.path.join(OUTPUT, f"grad_magnitude_{safe_label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: output/grad_magnitude_{safe_label}.png")


# =============================================================================
# 7. Per-component SDF grid (torsion box — 6 panel)
# =============================================================================

labels_to_plot = [
    "outer_skin", "spar_cap_upper", "spar_cap_lower",
    "web_0",      "web_1",          "core_0",
]
fig, axes = plt.subplots(2, 3, figsize=(16, 7))
for ax, lbl in zip(axes.ravel(), labels_to_plot):
    phi = grid_tors.eval(torsion[lbl])
    im  = ax.pcolormesh(grid_tors.X, grid_tors.Y, phi,
                        cmap="RdBu_r", vmin=-0.05, vmax=0.05, shading="auto")
    ax.contour(grid_tors.X, grid_tors.Y, phi, levels=[0.0],
               colors="k", linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_title(lbl, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="φ")
fig.suptitle("Per-component SDF fields — NACA 2412 torsion box @ 15° twist", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT, "component_sdfs_torsion.png"), dpi=150)
plt.close(fig)
print("  Saved: output/component_sdfs_torsion.png")


# =============================================================================
# 8. Export
# =============================================================================

print("\n─── Exporting data ───────────────────────────────────────")

report_torsion.to_json(os.path.join(OUTPUT, "section_properties_torsion.json"))
report_torsion.to_csv(os.path.join(OUTPUT, "section_properties_torsion.csv"))
export_midlines_csv(midlines, os.path.join(OUTPUT, "midlines_torsion.csv"))
export_section_json(torsion, grid_tors, midlines,
                    os.path.join(OUTPUT, "section_full_torsion.json"))

print("  Exported: section_properties_torsion.json/csv")
print("  Exported: midlines_torsion.csv")
print("  Exported: section_full_torsion.json")
print("\nDone.")
