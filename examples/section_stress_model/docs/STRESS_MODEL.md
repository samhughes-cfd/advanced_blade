# Thin-Walled Multi-Cell Section Stress Model (`multi_cell_blade_section.py`)

This document provides a thesis-level model overview for the implemented section solver, with explicit progression from continuous equations to discretized algebra and numerical solve.

---

## 1. Nomenclature and Sign Conventions

- Section frame `S`: $(y,z)$ in-plane, beam axis along $x$.
- Contour orientation: CCW positive on each cell loop.
- Shared walls use opposite traversal direction in neighboring cells.

| Symbol | Meaning | Units |
|---|---|---|
| $N,V_y,V_z,M_y,M_z,T$ | Section resultants | N, N, N, N m, N m, N m |
| $E,G,t$ | Young/shear modulus and wall thickness | Pa, Pa, m |
| $E_n=E/E_{\mathrm{ref}}$ | Normalized modulus | - |
| $A_b^\star=A_b(E_b/E_{\mathrm{ref}})$ | Effective boom area | m$^2$ |
| $q,\tau_{\mathrm{avg}}$ | Shear flow and average wall shear stress | N/m, Pa |

---

## 2. Governing Assumptions and Validity Domain

1. Thin-wall behavior on skins/webs ($t/R\ll1$).
2. Euler-Bernoulli axial kinematics for $\sigma_x$ recovery.
3. Boom-panel idealization: booms carry normal force, walls carry shear flow.
4. Linear elastic, small-strain response.
5. Closed-cell torsion/closure through Bredt-type compatibility.

---

## 3. Axial Model ($N,M_y,M_z$)

### Quick Start / Cheat Sheet

- **Point of this model:** recover axial normal stress $\sigma_x$ from extensional and bending resultants.
- **Primary unknowns:** $\varepsilon_0,\kappa_y,\kappa_z$.
- **Key solve:** $\mathbf{I}\boldsymbol{\kappa}=\mathbf{M}$ with $\varepsilon_0=N/EA$.
- **Recovery equation:** $\sigma_x=E_n\left(\varepsilon_0+\kappa_y(z-z_c)+\kappa_z(y-y_c)\right)$.
- **Main outputs:** wall/boom $\sigma_x$ and residual checks for $N,M_y,M_z$ recovery.

### Continuous Governing Equations

$$
EA=\int_{\text{walls}}E_nt\,\mathrm{d}s+\sum_bA_b^\star,\quad
y_c=\frac{EA_y}{EA},\quad z_c=\frac{EA_z}{EA}.
$$
$$
\mathbf{I}
\begin{bmatrix}\kappa_y\\\kappa_z\end{bmatrix}
=
\begin{bmatrix}M_y\\M_z\end{bmatrix},\quad
\mathbf{I}=
\begin{bmatrix}I_{yy}&I_{yz}\\I_{yz}&I_{zz}\end{bmatrix}.
$$
$$
\varepsilon_0=\frac{N}{EA},\qquad
\sigma_x=E_n\left(\varepsilon_0+\kappa_y(z-z_c)+\kappa_z(y-y_c)\right).
$$

### Discretized Form

For wall segments $e$ and booms $b$:
$$
EA\approx \sum_e E_{n,e}t_e\Delta s_e+\sum_bA_b^\star,\quad
I_{yy}\approx\sum_eE_{n,e}t_e(z_e-z_c)^2\Delta s_e+\sum_bA_b^\star(z_b-z_c)^2,
$$
$$
I_{zz}\approx\sum_eE_{n,e}t_e(y_e-y_c)^2\Delta s_e+\sum_bA_b^\star(y_b-y_c)^2,
$$
$$
I_{yz}\approx\sum_eE_{n,e}t_e(y_e-y_c)(z_e-z_c)\Delta s_e+\sum_bA_b^\star(y_b-y_c)(z_b-z_c).
$$

### Linear System and Numerical Solve

$$
\begin{bmatrix}\kappa_y\\\kappa_z\end{bmatrix}
=
\mathrm{solve}\!\left(
\begin{bmatrix}I_{yy}&I_{yz}\\I_{yz}&I_{zz}\end{bmatrix},
\begin{bmatrix}M_y\\M_z\end{bmatrix}
\right),\qquad
\varepsilon_0=N/EA.
$$

### Recovered Fields and Residual Checks

$$
\sigma_{x,p}=E_{n,p}\left(\varepsilon_0+\kappa_y(z_p-z_c)+\kappa_z(y_p-y_c)\right).
$$
$$
r_N=\left|\sum_p\sigma_{x,p}\Delta A_p-N\right|,\quad
r_{M_y}=\left|\sum_p\sigma_{x,p}(z_p-z_c)\Delta A_p-M_y\right|,
$$
$$
r_{M_z}=\left|\sum_p\sigma_{x,p}(y_p-y_c)\Delta A_p-M_z\right|.
$$

---

## 4. Transverse Shear Model ($V_y,V_z$)

### Quick Start / Cheat Sheet

- **Point of this model:** compute basic shear flow $q_b$ and average wall shear stress from transverse shear resultants.
- **Primary unknowns:** running first moments $Q_y,Q_z$ and segmentwise $q_b$.
- **Key equations:** $\mathrm{d}Q_y/\mathrm{d}s=E_n(y-y_c)t$, $\mathrm{d}Q_z/\mathrm{d}s=E_n(z-z_c)t$ with boom jumps.
- **Flow law:** $q_b=-D^{-1}[V_z(I_{zz}Q_z-I_{yz}Q_y)+V_y(I_{yy}Q_y-I_{yz}Q_z)]$.
- **Main outputs:** segment shear flow/stress and residual checks for $V_y,V_z$.

### Continuous Governing Equations

$$
\frac{\mathrm{d}Q_y}{\mathrm{d}s}=E_n(y-y_c)t,\qquad
\frac{\mathrm{d}Q_z}{\mathrm{d}s}=E_n(z-z_c)t.
$$
$$
\Delta Q_y=A_b^\star(y_b-y_c),\qquad \Delta Q_z=A_b^\star(z_b-z_c).
$$
$$
q_b=-\frac{1}{D}\left[
V_z(I_{zz}Q_z-I_{yz}Q_y)+V_y(I_{yy}Q_y-I_{yz}Q_z)
\right],\quad D=I_{yy}I_{zz}-I_{yz}^2.
$$
$$
\tau_{\mathrm{avg}}=q/t.
$$

### Discretized Form

$$
Q_{y,k+1}=Q_{y,k}+E_{n,k}(y_k-y_c)t_k\Delta s_k+\sum_{b\in k}\Delta Q_{y,b},
$$
$$
Q_{z,k+1}=Q_{z,k}+E_{n,k}(z_k-z_c)t_k\Delta s_k+\sum_{b\in k}\Delta Q_{z,b}.
$$
$$
q_{b,k+\frac12}= -\frac{1}{D}\left[
V_z(I_{zz}Q_{z,k+\frac12}-I_{yz}Q_{y,k+\frac12})
+V_y(I_{yy}Q_{y,k+\frac12}-I_{yz}Q_{z,k+\frac12})
\right].
$$

### Linear System and Numerical Solve

Use recursive integration (no global matrix inversion):
$$
Q_{y,0}=Q_{z,0}=0,
$$
march around the open contour from the cut, then evaluate $q_b$ segmentwise.

### Recovered Fields and Residual Checks

$$
\tau_{k+\frac12}=\frac{q_{k+\frac12}}{t_{k+\frac12}}.
$$
$$
r_{V_y}=\left|\sum_k q_k n_{y,k}\Delta s_k-V_y\right|,\qquad
r_{V_z}=\left|\sum_k q_k n_{z,k}\Delta s_k-V_z\right|.
$$

---

## 5. Closed-Cell Closure and Torsion ($T$)

### Quick Start / Cheat Sheet

- **Point of this model:** enforce closed-cell compatibility and torque equilibrium by adding constant cell circulations.
- **Primary unknowns:** $\mathbf{q}_0$ (cell circulatory flows), and $\theta'$ when $T\neq0$.
- **Key solve (closure-only):** $\mathbf{A}\mathbf{q}_0=\mathbf{r}$.
- **Key solve (with torque):** block system in $[\mathbf{q}_0,\theta']^{\mathsf T}$ with $G_{\mathrm{ref}}$ and $\mathbf{A}_{\mathrm{cell}}$.
- **Main outputs:** corrected closed-section $q_e$ and residual checks for compatibility and torque.

### Continuous Governing Equations

$$
q=q_b+q_{0,i}\ \text{on walls of cell }i.
$$
$$
\mathbf{A}\mathbf{q}_0=\mathbf{r},\qquad
r_i=-\oint_{\partial\Omega_i}\frac{q_b}{t}\,\mathrm{d}s
\quad (T=0).
$$
$$
\begin{bmatrix}
\mathbf{A}&-2G_{\mathrm{ref}}\mathbf{A}_{\mathrm{cell}}\\
2\mathbf{A}_{\mathrm{cell}}^{\mathsf T}&0
\end{bmatrix}
\begin{bmatrix}\mathbf{q}_0\\\theta'\end{bmatrix}
=
\begin{bmatrix}\mathbf{r}_{\mathrm{open}}\\T\end{bmatrix}.
$$

### Discretized Form

$$
A_{ij}=\sum_{e\in(\partial\Omega_i\cap\partial\Omega_j)}\frac{\Delta s_e}{t_e},
\qquad
r_i=-\sum_{e\in\partial\Omega_i}\frac{q_{b,e}}{t_e}\Delta s_e.
$$
$$
\mathbf{A}_{\mathrm{cell}}=[A_{\mathrm{cell},1},\dots,A_{\mathrm{cell},m}]^{\mathsf T},
\quad
A_{\mathrm{cell},i}=\frac12\sum_j(y_jz_{j+1}-y_{j+1}z_j).
$$

### Linear System and Numerical Solve

$$
\mathbf{q}_0=\mathrm{solve}(\mathbf{A},\mathbf{r})
$$
for closure-only, and the augmented block solve above when $T\neq0$.

### Recovered Fields and Residual Checks

$$
q_e=q_{b,e}+\sum_i\chi_{e,i}q_{0,i},\quad \chi_{e,i}\in\{-1,0,1\}.
$$
$$
r_{\mathrm{comp},i}=\left|\sum_{e\in\partial\Omega_i}\frac{q_e}{t_e}\Delta s_e\right|,
\qquad
r_T=\left|2\sum_iA_{\mathrm{cell},i}q_{0,i}-T\right|.
$$

---

## 6. Warping-Secondary Shear ($\mathrm{d}B/\mathrm{d}x$)

### Quick Start / Cheat Sheet

- **Point of this model:** add warping-induced secondary shear flow when bimoment varies spanwise.
- **Primary unknowns:** particular flow $q_\omega^{\mathrm{part}}$ and closure correction $\mathbf{q}_{0,\omega}$.
- **Source equation:** $\partial q_\omega/\partial s\approx-(t/I_\omega)(\mathrm{d}B/\mathrm{d}x)\hat\omega$.
- **Key solve:** reuse closure matrix with $\mathbf{q}_{0,\omega}=\mathrm{solve}(\mathbf{A},\mathbf{r}_\omega^{\mathrm{part}})$.
- **Main outputs:** $q_{\mathrm{warp}}$, $q_{\mathrm{tot}}$, and per-cell warping-closure residuals.

### Continuous Governing Equations

$$
\sigma_\omega=\frac{B\hat\omega}{I_\omega},\qquad
\frac{\partial q_\omega}{\partial s}\approx
-\frac{t}{I_\omega}\frac{\mathrm{d}B}{\mathrm{d}x}\hat\omega.
$$

### Discretized Form

$$
q_{\omega,k+1}^{\mathrm{part}}=
q_{\omega,k}^{\mathrm{part}}
-\frac{t_k}{I_\omega}\frac{\mathrm{d}B}{\mathrm{d}x}\hat\omega_k\Delta s_k,\qquad
q_{\omega,0}^{\mathrm{part}}=0.
$$
$$
r_{\omega,i}^{\mathrm{part}}=
-\sum_{e\in\partial\Omega_i}\frac{q_{\omega,e}^{\mathrm{part}}}{t_e}\Delta s_e.
$$

### Linear System and Numerical Solve

$$
\mathbf{q}_{0,\omega}=\mathrm{solve}(\mathbf{A},\mathbf{r}_{\omega}^{\mathrm{part}}),\qquad
q_{\omega,e}=q_{\omega,e}^{\mathrm{part}}+\sum_i\chi_{e,i}q_{0,\omega,i}.
$$

### Recovered Fields and Residual Checks

$$
q_{\mathrm{tot}}=q_{\mathrm{primary}}+q_{\mathrm{warp}},\qquad q_{\mathrm{warp}}=q_\omega.
$$
$$
r_{\omega,\mathrm{comp},i}=
\left|\sum_{e\in\partial\Omega_i}\frac{q_{\omega,e}}{t_e}\Delta s_e\right|.
$$

---

## 7. Shear-Centre Extraction

### Quick Start / Cheat Sheet

- **Point of this model:** locate $(y_{\mathrm{sc}},z_{\mathrm{sc}})$ from flexural-shear torque balance.
- **Primary unknowns:** centroid offsets $\Delta y_{\mathrm{sc}},\Delta z_{\mathrm{sc}}$.
- **Load cases:** two auxiliary solves at $T=0$: $(V_y,V_z)=(1,0)$ and $(0,1)$.
- **Key torque operator:** $T_x\approx\sum_e q_e(y_e\Delta z_e-z_e\Delta y_e)$.
- **Main outputs:** $(y_{\mathrm{sc}},z_{\mathrm{sc}})$ and verification residual $r_{\mathrm{sc}}$.

### Continuous Governing Equations

Use auxiliary solves at $T=0$:
$$
(V_y,V_z)=(1,0),\qquad (V_y,V_z)=(0,1),
$$
with torque functional
$$
T_x=\oint q(y\,\mathrm{d}z-z\,\mathrm{d}y).
$$

### Discretized Form

$$
T_x\approx\sum_e q_e(y_e\Delta z_e-z_e\Delta y_e).
$$
Let outputs be $T_x^{(V_y=1)}$ and $T_x^{(V_z=1)}$.

### Linear System and Numerical Solve

$$
\Delta z_{\mathrm{sc}}=-T_x^{(V_y=1)},\qquad
\Delta y_{\mathrm{sc}}=T_x^{(V_z=1)},
$$
$$
y_{\mathrm{sc}}=y_c+\Delta y_{\mathrm{sc}},\qquad
z_{\mathrm{sc}}=z_c+\Delta z_{\mathrm{sc}}.
$$

### Recovered Fields and Residual Checks

For test shear $(\bar V_y,\bar V_z)$:
$$
r_{\mathrm{sc}}=
\left|T_x^{\mathrm{recovered}}-(\bar V_y\Delta z_{\mathrm{sc}}-\bar V_z\Delta y_{\mathrm{sc}})\right|.
$$

---

## 8. Implementation Map (Input -> Operator -> Output)

| Module | Input | Operator | Output |
|---|---|---|---|
| `multi_cell_blade_section.py` | geometry + booms + resultants | EB stress, basic flow, Bredt closure, shear-centre solve | $\sigma_x$, $q$, $(y_{\mathrm{sc}},z_{\mathrm{sc}})$ |
| `laminate_clpt.py` | laminate stacks | CLPT constitutive assembly | ABD, ply stress auxiliaries |
| `sectorial_warping.py` | open contour + pole | sectorial integration | $\hat\omega$, $I_\omega$ |
| `vlasov_thinwall.py` | $B,\hat\omega,I_\omega$ | warping stress recovery | $\sigma_\omega$ |
| `warping_shear.py` | $\mathrm{d}B/\mathrm{d}x$ + topology | particular flow + closure correction | $q_{\mathrm{warp}}$ |

---

## 9. Verification Protocol and Residual Targets

Required checks:

1. Single-cell Bredt benchmark.
2. Symmetric section ($I_{yz}=0$) decoupling.
3. Asymmetric section ($I_{yz}\neq0$) coupled response.
4. Torque-only closure.
5. Warping-secondary activation with opening cancellation.

Residual targets:
$$
\left|\int\sigma_x\,\mathrm{d}A-N\right|/|N|<10^{-6},
\quad
\left|T_{\mathrm{recovered}}-T\right|/|T|<10^{-5},
$$
plus per-cell compatibility closure residuals below discretization-scaled tolerance.
