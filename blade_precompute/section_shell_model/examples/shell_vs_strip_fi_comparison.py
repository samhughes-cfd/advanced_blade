"""
Shell (MITC4) vs strip-theory comparison for a single blade section.

Runs a NACA63-415 two-cell section under combined flapwise + edgewise + torsion loading
using both the thin-wall strip model (MVP path) and MITC4, then compares:

  1. Shear-center location — strip vs Vlasov (MITC4).
  2. Panel-level shear stress τ_xy: thin-wall estimate (q/t) vs MITC4 element mean.

Usage (from repo root)::

    python blade_precompute/section_shell_model/examples/shell_vs_strip_fi_comparison.py

Output: ``blade_precompute/section_shell_model/examples/output/shell_vs_strip_fi_comparison.png``
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def main() -> None:
    _bootstrap_path()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blade_precompute.section_geometry.geometry.airfoil import AirfoilSDF
    from blade_precompute.section_shell_model.lib.recovery_adapter import run_section_both

    # ── section geometry ─────────────────────────────────────────────────────
    chord_m = 1.0          # normalised to 1 m for clarity
    naca_series = 6        # NACA 6-series
    naca_m, naca_p, naca_xx = 63.0, 4.0, 15.0   # NACA63-415
    spar_fracs = [0.20, 0.55]                     # two shear webs at 20 % and 55 % chord
    n_elem = 10                                   # MITC4 elements per panel

    airfoil_sdf = AirfoilSDF.from_naca_series(
        naca_series, naca_m, naca_p, naca_xx,
        n_points=200, chord=chord_m, closed_te=True,
    )
    airfoil = np.asarray(airfoil_sdf.vertices, dtype=np.float64)   # (n_pts, 2) in metres
    spars_m = [f * chord_m for f in spar_fracs]

    # ── loading ──────────────────────────────────────────────────────────────
    loads = dict(
        N=0.0,
        Vy=5_000.0,     # edgewise shear [N]
        Vz=20_000.0,    # flapwise shear [N]
        My=0.0,
        Mz=0.0,
        T=2_000.0,      # torsion [N·m]
        B=0.0,
        dB_dx=0.0,
    )

    # ── run both models ──────────────────────────────────────────────────────
    print("Running strip-theory (thin-wall MVP) and MITC4 shell models …")
    mvp_bundle, mitc4_bundle = run_section_both(
        airfoil,
        spars_m,
        n_elements_per_panel=n_elem,
        **loads,
    )
    print("  Done.")

    # ── shear center ─────────────────────────────────────────────────────────
    ysc_strip = float(mvp_bundle.y_sc)
    zsc_strip = float(mvp_bundle.z_sc)
    ysc_mitc4 = float(mitc4_bundle.vlasov_result.y_sc) if mitc4_bundle.vlasov_result else float("nan")
    zsc_mitc4 = float(mitc4_bundle.vlasov_result.z_sc) if mitc4_bundle.vlasov_result else float("nan")

    print(f"\nShear centre comparison (chord = {chord_m} m):")
    print(f"  Strip (thin-wall):  y = {ysc_strip:.4f} m,  z = {zsc_strip:.4f} m")
    print(f"  MITC4 (Vlasov):     y = {ysc_mitc4:.4f} m,  z = {zsc_mitc4:.4f} m")
    print(f"  dy = {abs(ysc_mitc4 - ysc_strip):.4f} m  ({100*abs(ysc_mitc4 - ysc_strip)/chord_m:.1f} % chord)")

    # ── per-panel shear stress comparison ────────────────────────────────────
    # Strip estimate: τ = q / thickness (using panel nominal thickness from MITC4 panel data).
    # MITC4: mean |τ_xy| over elements within each panel.
    panels = mvp_bundle.panels
    n_panels = len(panels)

    labels = [p.label if hasattr(p, "label") else f"P{i}" for i, p in enumerate(panels)]
    tau_strip = []
    tau_mitc4 = []
    x_mid_norm = []

    apr = mitc4_bundle.all_panel_mitc4_results or []
    _t = mitc4_bundle.mitc4_panel_thickness_m
    thicknesses = list(_t) if _t is not None else []

    for i, p in enumerate(panels):
        q_panel = float(np.asarray(mvp_bundle.q_tot[i], dtype=np.float64).mean())
        t = thicknesses[i] if i < len(thicknesses) and thicknesses[i] else 0.01
        tau_strip.append(abs(q_panel) / max(float(t), 1e-6) / 1e6)   # MPa

        # MITC4 element mean |N_xy| [N/m] converted to shear stress [MPa] via t.
        # N_xy is the in-plane membrane shear force per unit width (equiv. to thin-wall q).
        prs = apr[i] if i < len(apr) else []
        nxy_els = [abs(float(r.Nxy)) / max(float(t), 1e-6) / 1e6
                   for r in prs if hasattr(r, "Nxy")]
        tau_mitc4.append(float(np.mean(nxy_els)) if nxy_els else float("nan"))

        # mid x-position (chord-normalised)
        nds = np.asarray(p.nodes, dtype=np.float64)
        xm = float(nds[:, 0].mean()) / chord_m if nds.ndim == 2 else float(i) / n_panels
        x_mid_norm.append(xm)

    print(f"\nPanel shear stress comparison [MPa] (strip |q|/t vs MITC4 mean |Nxy|/t):")
    print(f"  {'Panel':<16} {'x/c':>6} {'Strip':>10} {'MITC4':>10} {'ratio':>8}")
    for i in range(n_panels):
        ratio = tau_mitc4[i] / tau_strip[i] if tau_strip[i] > 1e-6 else float("nan")
        print(f"  {labels[i]:<16} {x_mid_norm[i]:>6.3f} {tau_strip[i]:>10.3f} {tau_mitc4[i]:>10.3f} {ratio:>8.3f}")

    # ── plot ─────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "shell_vs_strip_fi_comparison.png"

    dark_bg = "#0f1117"
    col_strip = "#5dade2"
    col_mitc4 = "#f5b041"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=dark_bg)

    # ── left: airfoil + shear centre markers ─────────────────────────────────
    ax0 = axes[0]
    ax0.set_facecolor(dark_bg)
    af_closed = np.vstack([airfoil, airfoil[:1]])
    ax0.plot(af_closed[:, 0] / chord_m, af_closed[:, 1] / chord_m,
             color="#aaaaaa", lw=1.2, label="NACA63-415")
    for fi, sf in enumerate(spar_fracs):
        ax0.axvline(sf, color="#e74c3c", lw=1.0, ls="--",
                    label=f"spar @ {sf:.0%}c" if fi == 0 else None)
    ax0.scatter([ysc_strip], [zsc_strip], s=80, color=col_strip,
                marker="*", zorder=5, label=f"SC strip ({ysc_strip:.3f}, {zsc_strip:.3f}) m")
    ax0.scatter([ysc_mitc4], [zsc_mitc4], s=80, color=col_mitc4,
                marker="D", zorder=5, label=f"SC MITC4 ({ysc_mitc4:.3f}, {zsc_mitc4:.3f}) m")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x / chord [—]", color="#e0e0e0")
    ax0.set_ylabel("y / chord [—]", color="#e0e0e0")
    ax0.set_title("Section geometry + shear-centre comparison", color="#e0e0e0")
    ax0.tick_params(colors="#bbbbbb")
    for sp in ax0.spines.values():
        sp.set_edgecolor("#2a2a3a")
    ax0.grid(True, color="#2a2a3a", lw=0.4, alpha=0.5)
    ax0.legend(fontsize=7, labelcolor="#e0e0e0",
               facecolor="#1a1a2a", edgecolor="#2a2a3a")

    # ── right: shear stress per panel ────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_facecolor(dark_bg)
    x_idx = np.arange(n_panels)
    bar_w = 0.38
    ax1.bar(x_idx - bar_w / 2, tau_strip, width=bar_w, color=col_strip,
            alpha=0.85, label="Strip |q|/t")
    ax1.bar(x_idx + bar_w / 2, tau_mitc4, width=bar_w, color=col_mitc4,
            alpha=0.85, label="MITC4 mean |Nxy|/t")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=7, color="#bbbbbb")
    ax1.set_xlabel("Panel", color="#e0e0e0")
    ax1.set_ylabel("|Nxy|/t or |q|/t  [MPa]", color="#e0e0e0")
    ax1.set_title(
        f"Panel shear stress: strip vs MITC4\n"
        f"Vy={loads['Vy']/1e3:.0f} kN  Vz={loads['Vz']/1e3:.0f} kN  T={loads['T']/1e3:.1f} kN·m",
        color="#e0e0e0",
    )
    ax1.tick_params(colors="#bbbbbb")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#2a2a3a")
    ax1.grid(True, axis="y", color="#2a2a3a", lw=0.4, alpha=0.5)
    ax1.legend(fontsize=9, labelcolor="#e0e0e0",
               facecolor="#1a1a2a", edgecolor="#2a2a3a")

    fig.suptitle(
        f"NACA63-415 · chord={chord_m} m · spars @ {', '.join(f'{f:.0%}' for f in spar_fracs)} chord"
        f" · n_elem/panel={n_elem}",
        color="#e0e0e0", fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=dark_bg)
    plt.close(fig)

    print(f"\nPNG saved: {out_png}")


if __name__ == "__main__":
    main()
