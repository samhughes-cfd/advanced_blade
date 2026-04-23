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

Ply-level failure uses the **Hashin (1980) four-mode envelope** in material axes via `clpt_ply_failure_indices(..., criterion="hashin")` (default), with the same strength inputs `Xt`, `Xc`, `Yt`, `Yc`, `S12` as before. Tsai–Wu remains available as `criterion="tsai_wu"` in `laminate_clpt` for comparison.

## Limitations

- Coupling to **blade_precompute** orchestration is not wired here; this is an examples-level MVP.
- **Ny** and **moment** resultants are placeholders until shell curvature / secondary effects are mapped from geometry or higher-order theory.
- Direct import of `multi_cell_blade_section` is intentional for phase 1; a future refactor can move recovery behind a stable internal API in `blade_precompute`.

## Post-MVP refactor

1. Replace runtime `sys.path` injection with a proper package dependency or shared `blade_precompute` module.
2. Populate `Ny` and `Mx, My, Mxy` from shell kinematics (e.g. panel curvature) where available.
3. Optional FSDT path for `Qx`, `Qy` and transverse-shear-aware failure.

---

## Interface constraint modes

The global coupled MITC4 assembly (`solve_global_coupled_mitc4`) supports three modes
for enforcing displacement compatibility at panel junction clusters.  Choose the mode
via the `interface_constraint_mode` parameter of `run_section_with_mitc4_shell`.

### `shared` (legacy, collinear-correct)

Panel endpoints at each junction are merged into a **single global node** (the cluster node).
All five DOFs (UX, US, W, BETA_S, BETA_X) are shared as scalars.

- **Correct when**: all panels at a junction are collinear (e.g. single-panel or straight tube).
- **Incorrect for**: non-collinear junctions (T-junctions, L-junctions).  At a 90° skin–web
  joint, `U_S_skin ≠ U_S_web` physically, but `shared` equates them.  This is **Defect K**.
- **Boundary reactions**: `r_full` at the shared node includes contributions from all panels
  meeting there — cannot be split per-panel.

### `shared_rotated` (hybrid: uses 6-DOF cluster basis)

Equivalent to `transformed_basis` internally: allocates cluster master nodes and applies
6-DOF MPCs.  The label is stored in diagnostics to distinguish from an explicit
`transformed_basis` call, but the numerics are identical.

Use this when you want transformed-basis kinematics but prefer the `shared_rotated` label
in output (e.g. to track which runs used the auto-corrected path).

### `transformed_basis` (default, non-collinear-correct)

Each panel gets **unique endpoint nodes**.  Displacement compatibility is enforced via
multi-point constraints (MPCs) that transform between panel-local and cluster-reference frames:

```
u_panel_s  = ts * u_cluster_s + tn * w_cluster
w_panel    = ns * u_cluster_s + nn * w_cluster
beta_s     = ts * beta_cluster_t + tn * beta_cluster_n   (6-DOF basis)
```

where `ts = ŝ·t_ref`, `tn = ŝ·n_ref`, `ns = n̂·t_ref`, `nn = n̂·n_ref`.

- **Correct for**: arbitrary junction geometry including T-junctions and L-junctions.
- **Boundary reactions**: per-panel contribution extracted from edge-integrated tractions
  (not from `r_full` at cluster node, which carries aggregate reaction).
- **Primary metric**: `reaction_dNx_rel`, `reaction_dNxy_rel` < 0.10 (bounded, not machine-zero).
- **Default**: this mode is the default since Plan B (2026-04).

---

## Acceptance metrics

### Primary (authoritative)

| Metric | Source | Description | Acceptance |
|--------|--------|-------------|------------|
| `reaction_dNx_rel` | `r_full` (or edge tractions) | Spanwise force balance at junction | ≤ 0.05 skin, 0.10 web |
| `reaction_dNxy_rel` | same | Contour force balance at junction | ≤ 0.05 skin, 0.10 web |
| `global_force_balance_rel_at_fixed` | `r_full` at fixed UX DOFs | Global UX Newton's 3rd law | < 1e-6 |

### Secondary (diagnostic / informational)

| Metric | Source | Description | Note |
|--------|--------|-------------|------|
| `dTx_rel` | Edge line-integration | Spanwise traction continuity (Newton III) | < 0.05 (2-way) |
| `dT_yz_rel` | Edge line-integration | Contour traction vector continuity | < 0.10 (2-way) |
| `Tx_rel_cluster` | Cluster sum | N-way junction traction equilibrium | < 0.10 |
| `T_yz_rel_cluster` | Cluster sum | N-way junction YZ traction balance | < 0.15 |
| `resultant_dNx_rel` | Centroid resultant | Field Nx mismatch at boundary | informational |

The **cluster-sum metrics** (`Tx_rel_cluster`, `T_yz_rel_cluster`) are the authoritative
N-way equilibrium check; pair-wise secondary residuals are only meaningful for 2-way junctions.

### Known residuals

The secondary pair-wise residuals (`dTx_rel`, `dT_yz_rel`) may not converge to zero under
mesh refinement, especially at non-collinear junctions.  This is **Defect K**: the panel-local
shear gradient grows near the junction as elements shrink, inflating the recovered traction.
This is distinct from the primary reaction residuals which are governed by the MPC constraint
quality.

The old `global_force_balance_rel` panel-sum metric (≈ 0.5 for shared-DOF topology) is an
artefact of double-counting shared junction reactions and must **not** be used for acceptance.
Use `global_force_balance_rel_at_fixed` (< 1e-6 for a correctly solved assembly).

---

## Mesh-sweep comparison (Plan B1)

Run with `SHELL_MODE_SWEEP=1` to produce a side-by-side table for both `shared` and
`transformed_basis` modes, saved to `outputs/mesh_sweep_secondary.csv`.

Example output (representative values, actual numbers vary with geometry):

```
mode                 n_elem  ss_f_dNx  sw_f_dNx    ss_dTx    sw_dTx  ss_dT_yz  sw_dT_yz   cl_Tx  cl_Tyz
transformed_basis         8    ...       ...        0.18      ...      ...        ...        ...    ...
transformed_basis        12    ...       ...        0.20      ...      ...        ...        ...    ...
transformed_basis        16    ...       ...        0.22      ...      ...        ...        ...    ...
transformed_basis        24    ...       ...        0.24      ...      ...        ...        ...    ...
shared                    8    ...       ...        1.28      ...      ...        ...        ...    ...
shared                   24    ...       ...        1.33      ...      ...        ...        ...    ...
```

The `transformed_basis` mode shows lower absolute secondary residuals (< 0.3 vs > 1.0 for `shared`)
because the magnitude-based scaling denominator (`sqrt(Tx^2+Ts^2)`) avoids inflation at
near-orthogonal junctions. The values remain bounded (< 1.0) across mesh refinements.

---

## Gauss integration order

Edge tractions are integrated with `gauss_n=4` (4-point Gauss-Legendre on each interface
edge).  The per-Gauss-point arrays `Tx_gps` and `Ts_gps` are returned by
`mitc4_edge_shear_traction_integrated` for intra-edge variability inspection without
re-solving the element.
