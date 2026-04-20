# Section shell model (MVP)

Two-scale workflow for blade sections:

1. **Closed-cell section recovery** — reuses [section_stress_model](../section_stress_model/multi_cell_blade_section.py) (thin-wall `sigma_xx`, shear flow `q`, Bredt closure, warping helpers).
2. **Local CLPT shell handoff** — maps a panel station to shell resultants `Nx, Ny, Nxy, Mx, My, Mxy` with explicit [provenance](docs/SHELL_MODEL.md), then solves full laminate `[N; M]` via classical lamination theory.

## Run the example

From the repository root:

```bash
python examples/section_shell_model/run_example.py
```

This creates `examples/section_shell_model/outputs/` and writes PNG figures (default **150 dpi**):

| File | Content |
|------|---------|
| `mesh_shell_strips.png` | Panel midlines as linear strip elements; nodes; cell labels |
| `section_shell_demo_shear_flow.png` | Shear flow `q(s)` ribbons (thin-wall recovery) |
| `section_shell_demo_axial_stress.png` | Axial stress `σ_xx(s)` ribbons |
| `clpt_ply_tsai_wu.png` | Ply σ, ε, Tsai–Wu FI at reference skin station |
| `reference_panel_q_sigma_vs_s.png` | `q` and `σ_xx` vs contour `s` on reference panel |

## Layout

| Path | Role |
|------|------|
| `lib/types.py` | `ShellPanelResultants`, `SectionShellRecoveryBundle`, provenance enums |
| `lib/recovery_adapter.py` | Wraps `run_section` and builds shell DTOs |
| `lib/local_clpt_shell.py` | `solve_station_clpt_shell` — ply Tsai–Wu FI from resultants |
| `lib/example_plots.py` | PNG helpers (mesh, stress ribbons, CLPT, along-panel) |
| `docs/SHELL_MODEL.md` | Design notes, MVP limits, refactor path |
| `tests/` | Contract + FI regression tests |

## Dependencies

- `numpy`
- `matplotlib` (example PNG outputs and stress-model plots)
- sibling example [section_stress_model](../section_stress_model) (import path injected at runtime)

### Import path note

Add the ``examples`` directory to ``sys.path`` (or run from the repo root as in the command above). When manipulating ``sys.path`` manually, **do not** place ``section_shell_model`` ahead of ``section_stress_model``: both contain a top-level ``lib`` package, and the stress model’s ``lib.laminate_clpt`` must resolve to [section_stress_model/lib](../section_stress_model/lib). The example script and tests insert ``section_stress_model`` first, then ``examples``.
