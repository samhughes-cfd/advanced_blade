# Shell model design (MVP)

This package implements the **handoff layer** between global/section thin-wall recovery and **local CLPT shell** subcomponent analysis.

## References

- Section physics and equations: [section_stress_model/STRESS_MODEL.md](../section_stress_model/STRESS_MODEL.md)
- Recovery implementation: [section_stress_model/multi_cell_blade_section.py](../section_stress_model/multi_cell_blade_section.py)
- CLPT primitives: [section_stress_model/lib/laminate_clpt.py](../section_stress_model/lib/laminate_clpt.py)

## Handoff contract (`ShellPanelResultants`)

Per panel contour station, mid-surface resultants use laminate axes consistent with `laminate_clpt`:

- **Membrane:** `Nx`, `Ny`, `Nxy` [N/m]
- **Bending:** `Mx`, `My`, `Mxy` [N·m/m]
- **Optional (reserved):** `Qx`, `Qy` transverse shear [N/m] for future FSDT

### MVP mapping (derived vs placeholder)

| Field | MVP source | Provenance |
|-------|------------|------------|
| `Nx` | `sigma_xx * t` from thin-wall axial + bending (+ warping on skins) | `derived` |
| `Nxy` | shear flow `q` [N/m] | `derived` |
| `Ny` | `0` | `placeholder` (not recovered; demo uses `sigma_yy ≈ 0`) |
| `Mx`, `My`, `Mxy` | `0` | `placeholder` (no thickness-direction bending from section solver yet) |
| `Qx`, `Qy` | unset | `reserved` |

Every field carries a `FieldProvenance` entry so downstream code and reports can audit what is real physics vs stub.

## Local solver

`solve_station_clpt_shell` builds `N_vec = [Nx, Ny, Nxy]` and `M_vec = [Mx, My, Mxy]`, then calls `clpt_ply_failure_indices` (full `ABD` coupling). When MVP placeholders are zero, behavior matches the previous membrane-only skin station check (`Ny=0`, `M=0`).

## Limitations

- Coupling to **blade_precompute** orchestration is not wired here; this is an examples-level MVP.
- **Ny** and **moment** resultants are placeholders until shell curvature / secondary effects are mapped from geometry or higher-order theory.
- Direct import of `multi_cell_blade_section` is intentional for phase 1; a future refactor can move recovery behind a stable internal API in `blade_precompute`.

## Post-MVP refactor

1. Replace runtime `sys.path` injection with a proper package dependency or shared `blade_precompute` module.
2. Populate `Ny` and `Mx, My, Mxy` from shell kinematics (e.g. panel curvature) where available.
3. Optional FSDT path for `Qx`, `Qy` and transverse-shear-aware failure.
