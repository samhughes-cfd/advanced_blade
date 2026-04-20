# Thin-walled section stress model (`multi_cell_blade_section.py`)

This note documents the equations implemented in `multi_cell_blade_section.py`: a **boom / panel** idealisation of a multi-cell blade section with distinct laminates for skins, webs, and spar caps.

It also records the **full set of classical section resultants**—$N$, $V_y$, $V_z$, $M_y$, $M_z$, $T$—and how they enter **axial stress** and **shear flow**, with an explicit box on what the current script evaluates.

Math uses `$...$` (inline) and `$$...$$` (display) so it renders in GitHub, VS Code / Cursor, and common Markdown viewers.

The model separates **kinematic compatibility**, **equilibrium**, and **constitutive** relations; the sections below label which is which.

---

## Coordinate system and section resultants

- Section coordinates: **$y$** chordwise, **$z$** thickness-wise (vertical in the plots). The beam **axis** is $x$ (spanwise, not shown in the 2-D section plot).
- **Internal resultants** at the section (signs depend on your global convention; the forms below use standard “stress resultants work with positive section normals” bookkeeping—align signs with your load path when coding):

| Symbol | Name | Physical role |
|--------|------|-----------------|
| $N$ | Axial force | Uniform (average) axial extension / contraction of the section |
| $V_y$, $V_z$ | Shear forces | Transverse shear in $y$ and $z$ |
| $M_y$, $M_z$ | Bending moments | Bending about $y$ (fibre variation mainly in $z$) and about $z$ (fibre variation mainly in $y$) |
| $T$ | Torque | St. Venant / circulatory shear in closed cells; warping effects omitted here |

**Implemented in `multi_cell_blade_section.py`:** the example uses an **asymmetric** NACA 4-digit cambered profile (`naca_four_digit`, e.g. 2412-style). Default **section resultants** are set at module level as `SECTION_RESULTANTS = [N, V_y, V_z, M_z, M_y, T]$ (each unity in SI in the stock script). The script reports **shear-centre** coordinates $(y_{\mathrm{sc}}, z_{\mathrm{sc}})$ from **flexural shear-flow torque balance** (unit $V_y$, $V_z$ with $T=0$).

---

## Section geometry: centroid, axial stiffness, and second moments

**Normalisation** (code uses one reference modulus $E_{\mathrm{ref}}$):

$$
E_n = \frac{E}{E_{\mathrm{ref}}}, \qquad
A_{\mathrm{eff}} = A_{\mathrm{cap}}\,\frac{E_{\mathrm{cap}}}{E_{\mathrm{ref}}}.
$$

**Modulus-weighted area and centroid** (same as in code):

$$
(EA) = \int_{\mathrm{walls}} E_n\, t \,\mathrm{d}s + \sum_{\mathrm{booms}} A_{\mathrm{eff}},
$$

$$
(EA_y) = \int_{\mathrm{walls}} E_n\, t\, y \,\mathrm{d}s + \sum_{\mathrm{booms}} A_{\mathrm{eff}}\, y_b,
\qquad
(EA_z) = \int_{\mathrm{walls}} E_n\, t\, z \,\mathrm{d}s + \sum_{\mathrm{booms}} A_{\mathrm{eff}}\, z_b,
$$

$$
y_c = \frac{EA_y}{EA}, \qquad z_c = \frac{EA_z}{EA}.
$$

**Second moments about centroidal axes** (definitions used in bending and shear $Q$-integrals):

$$
I_{yy}
  = \int_{\mathrm{walls}} E_n\, t\, (z - z_c)^2 \,\mathrm{d}s
  + \sum_{\mathrm{booms}} A_{\mathrm{eff}}\, (z_b - z_c)^2,
$$

$$
I_{zz}
  = \int_{\mathrm{walls}} E_n\, t\, (y - y_c)^2 \,\mathrm{d}s
  + \sum_{\mathrm{booms}} A_{\mathrm{eff}}\, (y_b - y_c)^2,
$$

$$
I_{yz}
  = \int_{\mathrm{walls}} E_n\, t\, (y - y_c)(z - z_c) \,\mathrm{d}s
  + \sum_{\mathrm{booms}} A_{\mathrm{eff}}\, (y_b - y_c)(z_b - z_c).
$$

The script computes **$I_{yy}$, $I_{zz}$, and $I_{yz}$** about the modulus-weighted centroid (for biaxial bending and coupled shear).

---

## Structural idealisation (model assumptions)

- **Skins / webs**: thin walls; **shear flow** $q(s)$ [N/m] and **average** $\tau_{\mathrm{avg}} = q/t$.
- **Spar caps**: lumped **booms**; **no shear flow** through the boom; discrete **$\Delta Q$** jumps in the shear-flow integrals.

---

## Axial stress: $N$ and biaxial bending $M_y$, $M_z$

### Compatibility (kinematics)

**Euler–Bernoulli:** plane sections remain plane. Longitudinal strain is linear in $(y,z)$ about the centroid:

$$
\varepsilon_x(y,z) = \varepsilon_0 - \kappa_y\,(z - z_c) - \kappa_z\,(y - y_c),
$$

with uniform extension $\varepsilon_0$, curvatures $\kappa_y$, $\kappa_z$ (definitions follow from resultants once constitutive and equilibrium are enforced).

### Constitutive (uniaxial Hooke)

$$
\sigma_x = E(y,z)\,\varepsilon_x.
$$

### Equilibrium and closed-form $\sigma_x$ (general centroidal axes)

**Axial force:**

$$
N = \int_{\mathrm{section}} \sigma_x\,\mathrm{d}A.
$$

For a **uniform** axial resultant $N$ about the modulus-weighted centroid (no bending), the stress is piecewise:

$$
\sigma_{x,N} = N\,\frac{E}{EA_{\mathrm{phys}}}
= N\,\frac{E_n}{(EA)}.
$$

Here $(EA) = \int (E/E_{\mathrm{ref}})\,\mathrm{d}A$ in the code’s normalised sense, so $\sigma_{x,N}$ uses $E_n/(EA)$ consistently with the implementation style.

**Bending—decoupled case ($I_{yz} = 0$), common for symmetric airfoil sections:**

$$
\sigma_{x,\mathrm{bend}}
  = E_n \left(
      M_y\,\frac{z - z_c}{I_{yy}}
      - M_z\,\frac{y - y_c}{I_{zz}}
    \right).
$$

The sign on the $M_z$ term depends on your **sign convention** for positive $M_z$; the form above matches the usual “$M_y$ bends about $y$, $M_z$ about $z$” pair when $(y,z)$ are principal centroidal directions.

**Bending—general ($I_{yz} \neq 0$):** use the inverse of the $2\times 2$ bending stiffness coupling, or transform to **principal centroidal axes** where $I_{yz}'=0$ and apply the decoupled formula in the rotated coordinates.

**Superposition:**

$$
\sigma_x = \sigma_{x,N} + \sigma_{x,\mathrm{bend}}.
$$

**Implemented in code:** full linear combination with $[\kappa_y,\kappa_z]^T = \mathbf{I}^{-1}[M_y,M_z]^T$ and $\sigma = E_n (N/(EA) + \kappa_y (z-z_c) + \kappa_z (y-y_c))$ on walls and booms.

---

## Shear flow: $V_y$, $V_z$, and superposition

### Open-section “basic” flows $q_b$

Accumulate $Q_y(s)$, $Q_z(s)$ from the free edge with $\mathrm{d}Q_y/\mathrm{d}s = E_n (y-y_c) t$, $\mathrm{d}Q_z/\mathrm{d}s = E_n (z-z_c) t$, and **boom jumps** in both $Q_y$ and $Q_z$.

When $I_{yz}=0$, each shear direction decouples:

$$
q_{b,V_z} = -\frac{V_z}{I_{yy}} Q_z, \qquad q_{b,V_y} = -\frac{V_y}{I_{zz}} Q_y.
$$

For **general** $I_{yz}\neq 0$ (asymmetric section), use the coupled form (see Megson, *Aircraft Structures*):

$$
q_b(s) = -\frac{1}{D} \Bigl[
  V_z\,(I_{zz} Q_z - I_{yz} Q_y)
  + V_y\,(I_{yy} Q_y - I_{yz} Q_z)
\Bigr],
\qquad D = I_{yy} I_{zz} - I_{yz}^2.
$$

**Linear superposition** in $V_y,V_z$ is already in this single expression.

### Constitutive (shear stress vs flow)

$$
q = \tau_{\mathrm{avg}}\, t.
$$

### Torque $T$ and multi-cell closure

Total shear flow is $q = q_b + q_{0,i}$ on walls of cell $i$. The **opening** $\oint q_b/t\,\mathrm{d}s$ is balanced by constant cell flows $q_{0,i}$.

When **$T=0$** (shear only), solve $\mathbf{A}\mathbf{q}_0 = \mathbf{r}$ with $r_i = -\oint_{\partial\Omega_i} q_b/t\,\mathrm{d}s$.

When **$T\neq 0$**, the script augments the system with a twist rate $\theta'$ and torque balance using a reference shear modulus $G_{\mathrm{ref}}$ (`G_REF` in code):

$$
\begin{bmatrix}
\mathbf{A} & -2 G_{\mathrm{ref}} \mathbf{A}_{\mathrm{cell}} \\
2 \mathbf{A}_{\mathrm{cell}}^{\mathsf T} & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{q}_0 \\ \theta'
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{r}_{\mathrm{open}} \\ T
\end{bmatrix},
$$

where $\mathbf{A}_{\mathrm{cell}}$ lists polygonal **enclosed cell areas** (shoelace on each cell midline loop). This couples **shear closing** and **St. Venant-type** torsion in one linear solve.

### Secondary warping shear ($\mathrm{d}B/\mathrm{d}x$)

If the **bimoment** varies along the span, $\sigma_\omega = B\,\hat\omega / I_\omega$ has a longitudinal gradient

$$
\frac{\partial \sigma_\omega}{\partial x} = \frac{1}{I_\omega}\frac{\mathrm{d}B}{\mathrm{d}x}\,\hat\omega(s).
$$

Thin-wall longitudinal equilibrium (neglecting in-plane wall bending) gives a source on the shear flow,

$$
\frac{\partial q_\omega}{\partial s} \approx -\,t(s)\,\frac{\partial \sigma_\omega}{\partial x}
= -\,\frac{t(s)}{I_\omega}\frac{\mathrm{d}B}{\mathrm{d}x}\,\hat\omega(s),
$$

integrated along the **open** airfoil outline from a cut with $q=0$ at the first vertex. **Skins** inherit that particular $q_\omega$ mapped to panel nodes; **webs** use the same derivative relation with $\hat\omega$ taken **linear between** the web endpoints (endpoint $\hat\omega$ from the closest outline segment).

For **closed** multi-cell topology, the particular $q_\omega$ generally has non-zero **opening** per cell; the code applies the **same** Bredt compatibility matrix $\mathbf{A}$ with **$T=0$** to add constant cell circulations so $\oint q_\omega/t\,\mathrm{d}s$ balances per cell—analogous to $q_b$ closure.

**Total** shear flow used in plots is $q = q_{\mathrm{primary}} + q_{\mathrm{warp}}$ where $q_{\mathrm{primary}}$ is the Megson/Bredt solution from $V_y,V_z,T$. The return tuple also exposes `q_primary` and `q_warp` separately.

### Shear centre (flexural equilibrium)

For a general **asymmetric** closed thin-walled section, the shear centre is taken as the point where **flexural** shear flow produces **no resultant torque** about the $x$-axis when the transverse shear acts there. The code runs two **auxiliary** cases with **unit** shear and **$T=0$**: $(V_y,V_z)=(1,0)$ and $(0,1)$. For each, it forms the full $q(s)$, then evaluates

$$
T_x \approx \oint q\,(y\,\mathrm{d}z - z\,\mathrm{d}y)
$$

and recovers offsets from the modulus-weighted centroid (`shear_center_flexural_equilibrium`). The main load case may use nonzero $T$; shear-centre estimates still use the $T=0$ auxiliary solves.

---

## Companion modules (`lib/`)

| Module | Role |
|--------|------|
| `laminate_clpt.py` | Orthotropic plane-stress $[Q]$, $\bar Q(\theta)$, **ABD** stacking, ply $\sigma_{11},\sigma_{22},\tau_{12}$, thickness-integrated $H_{44},H_{55}$ for transverse shear stiffness. |
| `sectorial_warping.py` | Sectorial increment $\Delta\omega$, **open** airfoil outline, normalized warping $\hat\omega$, $I_\omega = \int \hat\omega^2 t\,\mathrm{d}s$ (pole at shear centre from equilibrium). |
| `timoshenko_section.py` | Smeared **$GA_y$, $GA_z$** from laminate $G$ and panel geometry; shear strains $\gamma \approx V/(GA)$ (section-level snapshot). |
| `vlasov_thinwall.py` | **Bimoment** $B$: $\sigma_\omega = B\,\hat\omega / I_\omega$ on the open outline, mapped to skin panels for plotting. |
| `warping_shear.py` | Secondary shear $q_\omega$ from $\mathrm{d}B/\mathrm{d}x$ (open particular + Bredt closure); summed into `q_tot` in `run_section`. |

**Warping normal stress** is added to Euler–Bernoulli skin $\sigma$ on non-web panels (webs omitted for $\sigma_\omega$ in this example).

---

## Summary table (resultant → typical use)

| Resultant | Axial stress $\sigma_x$ | Shear flow $q$ (thin wall) |
|-----------|-------------------------|----------------------------|
| $N$ | Uniform extension: $\sigma \propto E_n/(EA)$ | — |
| $M_y$ | Bending about $y$: $\propto E_n (z-z_c)/I_{yy}$ | Feeds $\partial\sigma/\partial x$ → drives $q_{b,V_z}$ |
| $M_z$ | Bending about $z$: $\propto E_n (y-y_c)/I_{zz}$ (if $I_{yz}=0$) | Feeds → $q_{b,V_y}$ |
| $V_z$ | — | Open $q_{b,V_z}$; closed + $q_{0,i}^{(V)}$ |
| $V_y$ | — | Open $q_{b,V_y}$; closed + $q_{0,i}^{(V)}$ |
| $T$ | — | Circulatory $q_T$ (Bredt / multi-cell); couples cells |
| $\mathrm{d}B/\mathrm{d}x$ | — | Secondary $q_\omega$ from $\partial\sigma_\omega/\partial x$; Bredt closure ($T=0$) |

---

## What `multi_cell_blade_section.py` implements

- **Geometry:** `naca_four_digit` asymmetric cambered profile (e.g. 2412-style); chord normalised to 1 in plots.
- **Section:** $EA$, $(y_c,z_c)$, $I_{yy}$, $I_{zz}$, $I_{yz}$; **cell areas** for torsion/shear closure.
- **Loads (example):** unit $N$, $V_y$, $V_z$, $M_y$, $M_z$, $T$ applied together.
- **Stress:** $\sigma = E_n (N/EA + \kappa_y (z-z_c) + \kappa_z (y-y_c))$ with $\boldsymbol{\kappa} = \mathbf{I}^{-1}[M_y,M_z]^{\mathsf T}$.
- **Shear:** coupled $q_b(V_y,V_z)$; **$\mathbf{q}_0$** from `solve_bredt_q0` including optional $T$ via the augmented system above; optional **`dB_dx`** adds warping secondary $q_\omega$ (open particular + Bredt closure for $q_\omega$ with $T=0$).
- **Shear centre:** $(y_{\mathrm{sc}}, z_{\mathrm{sc}})$ from flexural torque balance (`shear_center_flexural_equilibrium`).
- **CLPT demo:** optional ply stress recovery from a representative **[0/90/90/0]** stack in `__main__`.
- **Timoshenko:** $GA_y$, $GA_z$ and $\gamma_y,\gamma_z$ printed; bending stress law remains **EB** on the walls (Timoshenko does not modify $\sigma_x$ from curvature in this script).
- **Vlasov:** prescribed **bimoment** `B_BIMOMENT` adds $\sigma_\omega$ on skins (spanwise $B(x)$ not solved); optional **`dB_dx`** feeds secondary warping shear into `q_tot`.

---

## Explicit omissions (remaining gaps)

- **Timoshenko beam ODE:** `multi_cell_blade_section` still uses **section-level** $GA$ and $\gamma$ only; no Timoshenko **curvature correction** along the span in the bending law.
- **Spanwise beam model:** `beam_vlasov_1d` uses a **scalar** non-uniform torsion equation and **decoupled** cantilever integrals for $N$, $V$, $M$, $T$; full **7-DOF coupled** bending–torsion–warping matrices are **not** implemented.
- **CLPT in the main stress ribbons:** skin $\sigma(s)$ in plots remains the **homogenised** EB formula; the CLPT block is a **demonstration** printout.
- **Interlaminar / zigzag:** no layerwise or FSDT recovery.
- **Sectorial shear-centre iteration:** `sectorial_warping` includes outline-based sectorial helpers; **shear centre** in the orchestrator uses **flexural equilibrium** (robust for multi-cell closed sections). Sectorial $\hat\omega$ and $I_\omega$ use the open airfoil outline with pole at that shear centre.
- **Single-section `multi_cell` script:** still allows a **scalar** `B_BIMOMENT`; spanwise **$B(x)$** comes from `beam_vlasov_1d` when using `blade_span.run_span_stations`.

## Scope and numerical caveats

- Shear flow $q$ [N/m] uses $\tau_{\mathrm{avg}} = q/t$; ply-level $\tau$ from CLPT equilibrium is **not** merged into the ribbon plots.
- Validate design-critical values against test or higher-fidelity models when needed.

## Related files

- Orchestrator: `multi_cell_blade_section.py` (`section_properties`, `integrate_Q`, `solve_bredt_q0`, `compute_axial_stress`, `shear_center_flexural_equilibrium`).
- Physics helpers: `laminate_clpt.py`, `sectorial_warping.py`, `timoshenko_section.py`, `vlasov_thinwall.py`.

---

## Spanwise blade (`blade_frames.py`, `blade_span.py`, `beam_vlasov_1d.py`)

- **Span coordinate:** $x \in [0, L]$ from **root** ($x=0$) to **tip** ($x=L$).
- **Chord:** $c(x)$ scales the **chord-normalised** 2D coordinates of the airfoil (same normalisation as `naca_four_digit`).
- **Geometric twist:** angle $\theta_{\mathrm{geom}}(x)$ about the spanwise axis; **pivot** for rigid rotation in the section plane at **one-third chord from the LE** on the **mid-thickness** line (average of upper and lower surface $z$ at that $y$). After rotation, all coordinates are multiplied by $c(x)$ to obtain **metres** for `build_section` / `run_section`.
- **B frame vs S frame:** distributed loads and **cantilever resultants** are assembled in the **blade body** basis (**edge**, **flap**) in `beam_vlasov_1d.cantilever_resultants_B`. Section resultants for `run_section` must be in **S** $(y,z)$: use `blade_frames.resultants_B_to_S` so that at $\theta_{\mathrm{geom}}=0$, $V_y = V_{\mathrm{edge}}$ and $V_z = V_{\mathrm{flap}}$ (and similarly for $M_y$, $M_z$). **$N$** and **$T$** are unchanged by this 2-D rotation. See `blade_frames.py` docstring for the CCW sign convention.
- **Level 1:** prescribe $N(x), V_{\mathrm{edge}}(x), \ldots$ on a grid and call `blade_span.run_span_stations`.
- **Level 2:** `beam_vlasov_1d.solve_span_equilibrium` integrates distributed $q_x$, $p_{\mathrm{edge}}$, $p_{\mathrm{flap}}$, $m_x$ on a **cantilever** (fixed root, free tip) to get $N(x)$, shears, moments, $T(x)$, and solves the non-uniform torsion FD problem $EI_\omega \phi'''' - GJ \phi'' = m_x$ for **warping** $\phi(x)$ and **bimoment** $B(x) \approx -EI_\omega \phi''$. Pass $B(x)$ into `run_span_stations` for $\sigma_\omega$ at each station.
- **FD note:** the warping ODE solver requires a **uniform** $x$ grid and at least **7** nodes.

---

## Related files (spanwise)

- `blade_frames.py` — B→S rotation for shears and bending moments.
- `blade_span.py` — scale, twist, station loop over `run_section`.
- `beam_vlasov_1d.py` — cantilever statics + non-uniform torsion FD.
- `blade_span_viz.py` — **3D** span figures aligned with the `blade-structure` sweepkit convention (`span3d_figures.py`): matplotlib axes **(x, y, z) = (spanwise Z, edgewise Y_B, flapwise X_B)** with section points mapped by $\mathbf{p}_B=\mathbf{R}(\theta_{\mathrm{geom}})^{\mathsf T}\mathbf{p}_S$, **B-side-forward** camera (elev 25°, azim −50°), proportional `set_box_aspect`, and **Line3DCollection** on panel polylines with a **single** global colour scale for $q$ (kN/m) or $\sigma$ (MPa). Use `save_span3d_outputs` to write `span3d_geometry.png`, `span3d_q.png`, `span3d_sigma.png`.
