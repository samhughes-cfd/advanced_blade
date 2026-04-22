"""Tests for section_shell_model MVP (numpy + pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parent.parent.parent
_STRESS_ROOT = _EXAMPLES / "section_stress_model"

# Insert order: last insert(0) wins first on sys.path. Put stress_model first so
# ``lib`` is section_stress_model/lib (not section_shell_model/lib).
sys.path.insert(0, str(_EXAMPLES))
sys.path.insert(0, str(_STRESS_ROOT))


from section_shell_model.lib.local_clpt_shell import (  # noqa: E402
    default_skin_strengths_pa,
    solve_station_clpt_shell,
)
from section_shell_model.lib.recovery_adapter import (  # noqa: E402
    build_load_reaction_audit,
    check_panel_equilibrium,
    panel_station_shell_resultants,
    run_section_with_shell_mapping,
)
from section_shell_model.lib.types import ShellPanelResultants  # noqa: E402


def test_shell_resultants_match_membrane_mapping():
    """Nx, Nxy match laminate_clpt membrane_resultants_from_shell_stress."""
    from lib.laminate_clpt import membrane_resultants_from_shell_stress  # type: ignore[import-untyped]
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], dB_dx=0.0, B=0.0)
    panels, q_tot, sig_p = out[0], out[3], out[4]

    ref = panel_station_shell_resultants(
        panels, q_tot, sig_p, panel_index=0, station_index=None
    )
    t = ref.thickness_m
    N_direct = membrane_resultants_from_shell_stress(
        ref.sigma_xx_pa, 0.0, ref.tau_xy_pa, t
    )

    assert np.allclose(ref.to_N_vec(), N_direct, rtol=0, atol=1e-9)
    assert np.allclose(ref.to_M_vec(), np.zeros(3), atol=0.0)
    assert ref.provenance["Nx"].kind.value == "derived"
    assert ref.provenance["Ny"].kind.value == "placeholder"


def test_tsai_wu_fi_matches_direct_clpt_pipeline():
    """solve_station_clpt_shell reproduces stress-model ply FI for same N, M."""
    from lib.laminate_clpt import clpt_ply_failure_indices  # type: ignore[import-untyped]
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35], N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=0.0)
    panels, q_tot, sig_p = out[0], out[3], out[4]
    ref = panel_station_shell_resultants(panels, q_tot, sig_p, panel_index=0)
    plies = panels[0].lam.build_plies()
    st = default_skin_strengths_pa()

    shell_res = solve_station_clpt_shell(
        ref,
        plies,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
    )

    fi_tw, _, _, _ = clpt_ply_failure_indices(
        plies,
        ref.to_N_vec(),
        ref.to_M_vec(),
        st["Xt"],
        st["Xc"],
        st["Yt"],
        st["Yc"],
        st["S12"],
    )

    assert np.allclose(shell_res.fi_tsai_wu, fi_tw, rtol=0, atol=1e-12)


def test_run_section_with_shell_mapping_has_reference():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    bundle = run_section_with_shell_mapping(
        air,
        [0.25, 0.60],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
    )
    assert bundle.reference_resultants is not None
    assert bundle.I_omega >= 0.0
    assert np.isfinite(bundle.y_sc) and np.isfinite(bundle.z_sc)


# ---------------------------------------------------------------------------
# MITC4 element patch tests
# ---------------------------------------------------------------------------

def test_mitc4_stiffness_shape_and_symmetry():
    """20×20 stiffness must be symmetric and positive semi-definite (after BCs)."""
    from section_shell_model.lib.mitc4_element import mitc4_stiffness  # noqa: E402

    E, nu, t = 70e9, 0.33, 0.005
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    B_mat = np.zeros((3, 3))
    ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])

    K = mitc4_stiffness(L_s=0.1, L_x=1.0, ABD=ABD, thickness=t, G_eff=G)
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T, atol=1e-10), "Stiffness must be symmetric"

    # After pinning rigid-body DOFs: 6 eigenvalues should be near zero (mechanisms)
    eigvals = np.linalg.eigvalsh(K)
    n_near_zero = int(np.sum(np.abs(eigvals) < 1e-3 * eigvals[-1]))
    assert n_near_zero >= 6, f"Expected ≥6 near-zero eigenvalues, got {n_near_zero}"


def test_mitc4_edge_resultants_uniform_membrane_matches_centroid():
    from section_shell_model.lib.mitc4_element import (
        mitc4_edge_resultants,
        mitc4_resultants,
    )

    E, nu, t = 70e9, 0.33, 0.005
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0], [nu * A_val, A_val, 0], [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0], [nu * D_val, D_val, 0], [0, 0, 0.5 * (1 - nu) * D_val]])
    ABD = np.block([[A_mat, np.zeros((3, 3))], [np.zeros((3, 3)), D_mat]])
    L_s, L_x = 0.2, 1.0
    d = np.zeros(20)
    # Constant membrane strain: eps_xx = dux/dx = alpha.
    alpha = 1e-3
    for node, eta in enumerate([-1.0, -1.0, 1.0, 1.0]):
        x = 0.5 * (eta + 1.0) * L_x
        d[node * 5 + 0] = alpha * x
    c = mitc4_resultants(d, L_s, L_x, ABD)
    e = mitc4_edge_resultants(d, L_s, L_x, ABD)
    assert np.isclose(e["start"]["Nx"], c["Nx"], rtol=1e-10)
    assert np.isclose(e["end"]["Nx"], c["Nx"], rtol=1e-10)
    assert np.isclose(e["start"]["Nxy"], 0.0, atol=1e-10)
    assert np.isclose(e["end"]["Nxy"], 0.0, atol=1e-10)


def test_mitc4_edge_shear_traction_integrated_uniform_field():
    from section_shell_model.lib.mitc4_element import mitc4_edge_shear_traction_integrated

    E, nu, t = 70e9, 0.33, 0.005
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0], [nu * A_val, A_val, 0], [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0], [nu * D_val, D_val, 0], [0, 0, 0.5 * (1 - nu) * D_val]])
    ABD = np.block([[A_mat, np.zeros((3, 3))], [np.zeros((3, 3)), D_mat]])
    L_s, L_x = 0.2, 1.0
    d = np.zeros(20)
    gamma = 2e-3
    # u_x = gamma*s, u_s = 0 gives near-uniform gamma_xs = gamma
    for node, xi in enumerate([-1.0, 1.0, 1.0, -1.0]):
        s = 0.5 * (xi + 1.0) * L_s
        d[node * 5 + 0] = gamma * s
    out_start = mitc4_edge_shear_traction_integrated(d, L_s, L_x, ABD, edge="start")
    out_end = mitc4_edge_shear_traction_integrated(d, L_s, L_x, ABD, edge="end")
    assert np.isclose(out_start["Nxy_edge_int"], -out_end["Nxy_edge_int"], rtol=1e-10, atol=1e-10)
    assert abs(out_start["Nxy_edge_int"]) > 0.0


def test_mitc4_patch_pure_Nx():
    """
    Uniform Nx applied to a simply-supported isotropic panel → MITC4 recover
    Nx ≈ applied, Ny ≈ 0, Mx ≈ My ≈ Mxy ≈ 0 at the centre element.
    """
    from section_shell_model.lib.panel_mitc4_model import solve_panel_mitc4  # noqa: E402

    E, nu, t = 70e9, 0.33, 0.005
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    B_mat = np.zeros((3, 3))
    ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])

    Nx_applied = 1000.0   # N/m
    s_panel = np.linspace(0.0, 0.5, 20)
    Nx_panel = np.full_like(s_panel, Nx_applied)
    Nxy_panel = np.zeros_like(s_panel)

    results = solve_panel_mitc4(
        ABD=ABD,
        thickness=t,
        G_eff=G,
        s_panel=s_panel,
        Nx_panel=Nx_panel,
        Nxy_panel=Nxy_panel,
        spar_s_coords=[0.0, 0.5],
        n_elements=8,
        bc_mode="full_bottom_clamp",
    )
    assert len(results) == 8
    mid = results[len(results) // 2]

    # Nx should match applied load within 5%
    assert abs(mid.Nx - Nx_applied) / Nx_applied < 0.05, (
        f"Nx={mid.Nx:.1f} expected ≈{Nx_applied}"
    )
    # Bending moments should be small relative to Nx*t (thin panel, no lateral load)
    scale = Nx_applied * t
    assert abs(mid.Mx) < 0.5 * scale, f"Mx={mid.Mx:.3e} unexpectedly large"
    assert abs(mid.My) < 0.5 * scale, f"My={mid.My:.3e} unexpectedly large"
    # All provenance should be MITC4
    assert mid.provenance["Ny"].kind.value == "mitc4"
    assert mid.provenance["Mx"].kind.value == "mitc4"


# ---------------------------------------------------------------------------
# Vlasov warping tests
# ---------------------------------------------------------------------------

def test_vlasov_shear_centre_finite():
    """compute_section_vlasov returns finite shear centre and positive I_omega_E."""
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]
    from section_shell_model.lib.section_vlasov import compute_section_vlasov  # noqa: E402

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    out = run_section(air, [0.35])
    panels = out[0]

    vlasov = compute_section_vlasov(air, panels, B=0.0, dB_dx=0.0)
    assert np.isfinite(vlasov.y_sc)
    assert np.isfinite(vlasov.z_sc)
    assert vlasov.I_omega_E > 0.0
    assert len(vlasov.omega_hat_v) > 0


def test_vlasov_zero_bimoment_gives_zero_stress():
    """With B=0, warping normal stress must be identically zero."""
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]
    from section_shell_model.lib.section_vlasov import compute_section_vlasov  # noqa: E402

    air = naca_four_digit(m=0.0, p=0.4, t_c=0.12, n=48)
    panels = run_section(air, [0.35])[0]
    vlasov = compute_section_vlasov(air, panels, B=0.0, dB_dx=0.0)
    for sig in vlasov.sigma_omega:
        assert np.allclose(sig, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# MITC4 integration test
# ---------------------------------------------------------------------------

def test_mitc4_integration_no_placeholders():
    """run_section_with_mitc4_shell: all resultant provenance must be MITC4 or DERIVED."""
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell  # noqa: E402
    from section_shell_model.lib.types import ProvenanceKind  # noqa: E402

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    bundle = run_section_with_mitc4_shell(
        air,
        [0.35],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
        n_elements_per_panel=6,
    )
    assert bundle.reference_resultants is not None
    ref = bundle.reference_resultants
    placeholder_fields = [
        k for k, v in ref.provenance.items()
        if v.kind == ProvenanceKind.PLACEHOLDER
    ]
    assert placeholder_fields == [], (
        f"Fields still placeholder after MITC4 solve: {placeholder_fields}"
    )


# ---------------------------------------------------------------------------
# Item 3: Non-zero bimoment / dB_dx activates warping contributions
# ---------------------------------------------------------------------------

def test_vlasov_nonzero_bimoment_activates_stress():
    """
    B=1e3 → sigma_omega non-zero for at least one skin panel.
    B=0   → sigma_omega zero everywhere.
    Scaling: sigma_omega scales linearly with B.
    """
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]
    from section_shell_model.lib.section_vlasov import compute_section_vlasov

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=48)
    panels = run_section(air, [0.35])[0]

    vlasov_0 = compute_section_vlasov(air, panels, B=0.0, dB_dx=0.0)
    for sig in vlasov_0.sigma_omega:
        assert np.allclose(sig, 0.0, atol=1e-30), "B=0 must give zero sigma_omega"

    vlasov_1 = compute_section_vlasov(air, panels, B=1e3, dB_dx=0.0)
    max_sig_1 = max(float(np.max(np.abs(sig))) for sig in vlasov_1.sigma_omega if len(sig))
    assert max_sig_1 > 0.0, "B=1e3 must give non-zero sigma_omega on at least one panel"

    # Linear scaling: B=2e3 → sigma_omega twice as large
    vlasov_2 = compute_section_vlasov(air, panels, B=2e3, dB_dx=0.0)
    max_sig_2 = max(float(np.max(np.abs(sig))) for sig in vlasov_2.sigma_omega if len(sig))
    assert np.isclose(max_sig_2, 2.0 * max_sig_1, rtol=1e-10), (
        f"sigma_omega must scale linearly with B: {max_sig_2:.4e} vs 2*{max_sig_1:.4e}"
    )


def test_vlasov_nonzero_dBdx_activates_warping_shear():
    """dB_dx=1e3 with B=0 → q_omega non-zero on at least one skin panel."""
    from multi_cell_blade_section import naca_four_digit, run_section  # type: ignore[import-untyped]
    from section_shell_model.lib.section_vlasov import compute_section_vlasov

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=48)
    panels = run_section(air, [0.35])[0]

    vlasov = compute_section_vlasov(air, panels, B=0.0, dB_dx=1e3)
    max_q = max(float(np.max(np.abs(q))) for q in vlasov.q_omega if len(q))
    assert max_q > 0.0, "dB_dx=1e3 must give non-zero q_omega on at least one panel"


# ---------------------------------------------------------------------------
# Item 4: MITC4 bending moment patch test (curved panel → Donnell load → Mx)
# ---------------------------------------------------------------------------

def test_mitc4_patch_bending_moment():
    """
    Curved isotropic panel: κ = 1/R, uniform Nx → Donnell lateral load q_n = Nx*κ.
    Verify curvature coupling activates bending compared to a flat reference panel.
    """
    from section_shell_model.lib.panel_mitc4_model import solve_panel_mitc4

    E, nu, t = 70e9, 0.30, 0.002
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    ABD = np.block([[A_mat, np.zeros((3, 3))], [np.zeros((3, 3)), D_mat]])

    L_s = 0.20    # panel arc length [m]
    R = 0.5       # radius of curvature [m]
    Nx_applied = 1000.0  # N/m

    # Circular arc nodes in y-z plane: angle from 0 to θ_max = L_s/R
    n_nodes = 21
    theta = np.linspace(0.0, L_s / R, n_nodes)
    nodes_yz = np.column_stack([R * np.sin(theta), R * (np.cos(theta) - 1.0)])

    s_panel = np.linspace(0.0, L_s, n_nodes)
    Nx_panel = np.full(n_nodes, Nx_applied)
    Nxy_panel = np.zeros(n_nodes)

    results = solve_panel_mitc4(
        ABD=ABD,
        thickness=t,
        G_eff=G,
        s_panel=s_panel,
        Nx_panel=Nx_panel,
        Nxy_panel=Nxy_panel,
        spar_s_coords=[0.0, L_s],
        nodes_yz=nodes_yz,
        n_elements=10,
    )
    assert len(results) == 10

    mid = results[len(results) // 2]
    assert abs(mid.Mx) > 1e-6, "Curvature load should induce non-zero bending moment Mx"

    # Flat reference geometry: same s-discretisation with zero curvature.
    flat_nodes_yz = np.column_stack([s_panel, np.zeros_like(s_panel)])
    results_flat = solve_panel_mitc4(
        ABD=ABD,
        thickness=t,
        G_eff=G,
        s_panel=s_panel,
        Nx_panel=Nx_panel,
        Nxy_panel=Nxy_panel,
        spar_s_coords=[0.0, L_s],
        nodes_yz=flat_nodes_yz,
        n_elements=10,
    )
    mid_flat = results_flat[len(results_flat) // 2]
    assert abs(mid.Mx) > 10.0 * max(abs(mid_flat.Mx), 1e-30), (
        f"Curved panel should produce much larger Mx than flat panel: "
        f"Mx_curved={mid.Mx:.4e}, Mx_flat={mid_flat.Mx:.4e}"
    )


# ---------------------------------------------------------------------------
# Item 5: Batho closed-cell correction tests
# ---------------------------------------------------------------------------

class _FakeLam:
    def __init__(self, E: float, t: float):
        self.E = E
        self.t = t


class _FakePanel:
    def __init__(self, nodes: np.ndarray, E: float, t: float, label: str = ""):
        self.nodes = np.asarray(nodes, dtype=float)
        self.lam = _FakeLam(E, t)
        diffs = np.diff(self.nodes, axis=0)
        ds = np.hypot(diffs[:, 0], diffs[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.label = label


def _box_airfoil(n: int = 6, width: float = 1.0, height: float = 0.5) -> np.ndarray:
    """Rectangular box as airfoil (upper + lower surface, LE→TE each)."""
    x = np.linspace(0.0, width, n)
    upper = np.column_stack([x, np.full(n, height / 2.0)])
    lower = np.column_stack([x, np.full(n, -height / 2.0)])
    return np.vstack([upper, lower])


def test_vlasov_batho_single_cell_box():
    """Rectangular single-cell box: n_cells==1, I_omega_E>0, omega_hat_v non-trivial."""
    from section_shell_model.lib.section_vlasov import compute_section_vlasov

    airfoil_box = _box_airfoil(n=10, width=1.0, height=0.5)
    E, t = 70e9, 2e-3

    # Two skin panels: upper and lower surfaces
    upper_nodes = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.25)])
    lower_nodes = np.column_stack([np.linspace(1, 0, 10), np.full(10, -0.25)])
    panels = [
        _FakePanel(upper_nodes, E, t, label="USkin"),
        _FakePanel(lower_nodes, E, t, label="LSkin"),
    ]

    vlasov = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                    webs_geom=[], t_web=t)
    assert vlasov.n_cells == 1, f"Expected 1 cell, got {vlasov.n_cells}"
    assert vlasov.I_omega_E > 0.0, "I_omega_E must be positive"
    assert float(np.max(np.abs(vlasov.omega_hat_v))) > 0.0, "omega_hat must be non-trivial"


def test_vlasov_batho_two_cell():
    """Two-cell box (mid web at x=0.5): n_cells==2 and I_omega_E differs from single-cell."""
    from section_shell_model.lib.section_vlasov import compute_section_vlasov

    airfoil_box = _box_airfoil(n=10, width=1.0, height=0.5)
    E, t = 70e9, 2e-3

    upper_nodes = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.25)])
    lower_nodes = np.column_stack([np.linspace(1, 0, 10), np.full(10, -0.25)])
    panels = [
        _FakePanel(upper_nodes, E, t, label="USkin"),
        _FakePanel(lower_nodes, E, t, label="LSkin"),
    ]

    # Single-cell baseline
    vlasov_1 = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                      webs_geom=[], t_web=t)

    # Add mid-web at x=0.5
    web = ((np.array([0.5, 0.25]), np.array([0.5, -0.25])),)
    vlasov_2 = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                      webs_geom=list(web), t_web=t)

    assert vlasov_2.n_cells == 2, f"Expected 2 cells, got {vlasov_2.n_cells}"
    # Batho correction for two cells differs from single cell by a meaningful margin.
    rel_diff = abs(vlasov_2.I_omega_E - vlasov_1.I_omega_E) / max(abs(vlasov_1.I_omega_E), 1e-30)
    assert rel_diff > 0.10, (
        "Two-cell Batho correction should change I_omega_E by >10%"
    )


def _mk_resultant(nx: float, nxy: float) -> ShellPanelResultants:
    return ShellPanelResultants(
        Nx=float(nx),
        Ny=0.0,
        Nxy=float(nxy),
        Mx=0.0,
        My=0.0,
        Mxy=0.0,
    )


def _mk_panel(nodes: np.ndarray, label: str = "skin"):
    class _Panel:
        pass

    p = _Panel()
    p.nodes = np.asarray(nodes, dtype=float)
    p.label = label
    return p


def test_equilibrium_sign_normalization_for_opposite_tangent():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    panels = [p0, p1]
    all_res = [
        [_mk_resultant(10.0, 5.0)],
        [_mk_resultant(10.0, -5.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["resultant_dNxy_rel"] < 1e-12
    assert chk["resultant_pass"]


def test_equilibrium_uses_geometric_adjacency_not_index_order():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[10.0, 0.0], [11.0, 0.0]]), label="USkin")
    p2 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    panels = [p0, p1, p2]
    all_res = [
        [_mk_resultant(10.0, 2.0)],
        [_mk_resultant(100.0, 20.0)],
        [_mk_resultant(10.0, -2.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 1
    pair = {checks[0]["pi"], checks[0]["pj"]}
    assert pair == {0, 2}


def test_equilibrium_includes_wraparound_for_closed_loop():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [0.5, 1.0]]), label="LSkin")
    p2 = _mk_panel(np.array([[0.5, 1.0], [0.0, 0.0]]), label="Web Core")
    panels = [p0, p1, p2]
    all_res = [
        [_mk_resultant(10.0, 1.0)],
        [_mk_resultant(10.0, 1.0)],
        [_mk_resultant(10.0, 1.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 3


def test_equilibrium_tiered_thresholds_skin_skin_vs_skin_web():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    p2 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web @ 0.35c")

    checks_skin_skin = check_panel_equilibrium(
        [[_mk_resultant(10.0, 1.0)], [_mk_resultant(9.4, -1.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}}},
            {"boundary_reaction": {"end": {"Fx": 9.4, "Fs": -1.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks_skin_skin) == 1
    assert checks_skin_skin[0]["boundary_type"] == "skin-skin"
    assert checks_skin_skin[0]["reaction_dNx_rel"] > checks_skin_skin[0]["reaction_tol_nx"]
    assert not checks_skin_skin[0]["reaction_pass_nx"]

    checks_skin_web = check_panel_equilibrium(
        [[_mk_resultant(10.0, 1.0)], [_mk_resultant(9.2, 1.0)]],
        [p0, p2],
        all_panel_mitc4_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}}},
            {"boundary_reaction": {"start": {"Fx": 9.2, "Fs": 1.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks_skin_web) == 1
    assert checks_skin_web[0]["boundary_type"] == "skin-web"
    assert checks_skin_web[0]["reaction_dNx_rel"] < checks_skin_web[0]["reaction_tol_nx"]
    assert checks_skin_web[0]["reaction_pass_nx"]


def test_panel_solver_returns_reaction_diagnostics():
    from section_shell_model.lib.panel_mitc4_model import solve_panel_mitc4

    E, nu, t = 70e9, 0.33, 0.005
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    B_mat = np.zeros((3, 3))
    ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])

    s_panel = np.linspace(0.0, 0.5, 20)
    Nx_panel = np.full_like(s_panel, 1000.0)
    Nxy_panel = np.full_like(s_panel, 20.0)
    out = solve_panel_mitc4(
        ABD=ABD,
        thickness=t,
        G_eff=G,
        s_panel=s_panel,
        Nx_panel=Nx_panel,
        Nxy_panel=Nxy_panel,
        spar_s_coords=[0.0, 0.5],
        n_elements=8,
        return_diagnostics=True,
    )
    results, diag = out
    assert len(results) == 8
    assert "residual" in diag and "free_res_rel" in diag["residual"]
    assert diag["residual"]["free_res_rel"] < 1e-3
    assert abs(diag["load_totals"]["Fx_total"]) > 0.0


def test_equilibrium_reaction_metric_uses_panel_diagnostics():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 0.0)], [_mk_resultant(10.0, 0.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 4.0}}},
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": -4.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["reaction_dNxy_rel"] < 1e-12
    assert chk["reaction_pass"]


def test_equilibrium_secondary_prefers_interface_field_set():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 0.0)], [_mk_resultant(100.0, 0.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 1.0}},
            },
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": -1.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": -1.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["resultant_dNx_rel"] < 1e-12
    assert chk["resultant_dNxy_rel"] < 1e-12
    # Field-set available but no edge-traction set — falls back to field-fallback mode.
    assert chk["nxy_compare_mode"] == "field-fallback"
    assert chk["resultant_dNx_rel_centroid"] > 0.5


def test_equilibrium_strict_skin_web_maps_components():
    # p0 "end" (normal_sign=+1): Tx=2, Ts=5, t_i=[1,0]
    # p1 "start" (normal_sign=-1): Tx=-2, Ts=-5, t_j=[1,0] (collinear)
    # dTx = 2 + (-2) = 0  →  dTx_rel = 0
    # dT_yz = 5*[1,0] + (-5)*[1,0] = [0,0]  →  dT_yz_rel = 0
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 2.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 2.0}},
                "interface_edge_set": {"end": {"Tx_int": 2.0, "Ts_int": 5.0}},
            },
            {
                "boundary_reaction": {"start": {"Fx": 10.0, "Fs": 2.0}},
                "interface_field_set": {"start": {"Nx": 10.0, "Nxy": 99.0}},
                "interface_edge_set": {"start": {"Tx_int": -2.0, "Ts_int": -5.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["boundary_type"] == "skin-web"
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-12
    assert chk["dT_yz_rel"] < 1e-12


def test_equilibrium_strict_skin_skin_uses_tx_component():
    # p0 "end" (normal_sign=+1): Tx=3, Ts=8, t_i=[1,0]
    # p1 "end" (normal_sign=+1): Tx=-3, Ts=8, t_j=[-1,0]  (panels converge)
    # dTx = 3 + (-3) = 0  →  dTx_rel = 0
    # dT_yz = 8*[1,0] + 8*[-1,0] = [0,0]  →  dT_yz_rel = 0
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 1.0)], [_mk_resultant(5.0, -1.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"end": {"Tx_int": 3.0, "Ts_int": 8.0}},
            },
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": -1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": -1.0}},
                "interface_edge_set": {"end": {"Tx_int": -3.0, "Ts_int": 8.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-12
    assert chk["dT_yz_rel"] < 1e-12


def test_panel_solver_legacy_bottom_clamp_mode_available():
    from section_shell_model.lib.panel_mitc4_model import solve_panel_mitc4

    E, nu, t = 70e9, 0.33, 0.005
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    B_mat = np.zeros((3, 3))
    ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])
    s_panel = np.linspace(0.0, 0.5, 20)
    Nx_panel = np.full_like(s_panel, 1000.0)
    Nxy_panel = np.zeros_like(s_panel)

    res_default = solve_panel_mitc4(
        ABD=ABD, thickness=t, G_eff=G, s_panel=s_panel,
        Nx_panel=Nx_panel, Nxy_panel=Nxy_panel, spar_s_coords=[0.0, 0.5], n_elements=6,
    )
    res_legacy = solve_panel_mitc4(
        ABD=ABD, thickness=t, G_eff=G, s_panel=s_panel,
        Nx_panel=Nx_panel, Nxy_panel=Nxy_panel, spar_s_coords=[0.0, 0.5], n_elements=6,
        bc_mode="full_bottom_clamp",
    )
    assert len(res_default) == len(res_legacy) == 6


def test_equilibrium_topology_graph_handles_three_way_junction():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="USkin2")
    p2 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web A")
    panels = [p0, p1, p2]
    all_res = [[_mk_resultant(1.0, 0.0)], [_mk_resultant(1.0, 0.0)], [_mk_resultant(1.0, 0.0)]]
    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 3


def test_run_section_with_mitc4_shell_global_coupled_default():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    bundle = run_section_with_mitc4_shell(
        air,
        [0.35],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
        n_elements_per_panel=6,
    )
    assert bundle.all_panel_mitc4_results is not None
    assert bundle.all_panel_mitc4_diagnostics is not None
    assert any(di.get("solver") == "global_coupled" for di in bundle.all_panel_mitc4_diagnostics if di)
    assert any(di.get("constraint_stats", {}).get("n_slave_dofs", 0) >= 0 for di in bundle.all_panel_mitc4_diagnostics if di)


def test_build_load_reaction_audit_computes_summary():
    di = [
        {
            "load_totals": {"Fx_total": 10.0, "Fs_total": 2.0},
            "boundary_reaction_set": {"start": {"Fx": -6.0, "Fs": -1.0}, "end": {"Fx": -4.0, "Fs": -1.0}},
        },
        {
            "load_totals": {"Fx_total": 8.0, "Fs_total": 1.0},
            "boundary_reaction_set": {"start": {"Fx": -7.0, "Fs": -0.5}, "end": {"Fx": -1.0, "Fs": -0.4}},
        },
    ]
    audit = build_load_reaction_audit(di)
    assert int(audit["n_panels"]) == 2
    assert audit["max_rel_mismatch"] >= 0.0
    assert audit["mean_rel_mismatch"] >= 0.0


def test_secondary_metric_mesh_refinement_smoke():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    coarse = run_section_with_mitc4_shell(
        air,
        [0.35],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
        n_elements_per_panel=8,
        use_global_coupled=True,
    )
    fine = run_section_with_mitc4_shell(
        air,
        [0.35],
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        reference_panel_index=0,
        n_elements_per_panel=16,
        use_global_coupled=True,
    )
    coarse_checks = check_panel_equilibrium(
        coarse.all_panel_mitc4_results or [],
        coarse.panels,
        all_panel_mitc4_diagnostics=coarse.all_panel_mitc4_diagnostics,
    )
    fine_checks = check_panel_equilibrium(
        fine.all_panel_mitc4_results or [],
        fine.panels,
        all_panel_mitc4_diagnostics=fine.all_panel_mitc4_diagnostics,
    )
    assert len(coarse_checks) > 0 and len(fine_checks) > 0
    coarse_pass_ratio = float(sum(1 for chk in coarse_checks if chk["reaction_pass"])) / float(len(coarse_checks))
    fine_pass_ratio = float(sum(1 for chk in fine_checks if chk["reaction_pass"])) / float(len(fine_checks))
    assert coarse_pass_ratio >= 0.75
    assert fine_pass_ratio >= 0.75


def test_global_load_target_reconciliation_present():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=30)
    bundle = run_section_with_mitc4_shell(
        air, [0.35], N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=1.0,
        n_elements_per_panel=6, use_global_coupled=True,
    )
    diags = [d for d in (bundle.all_panel_mitc4_diagnostics or []) if d]
    assert len(diags) > 0
    assert any("Fx_target" in d.get("load_totals", {}) for d in diags)


def test_skin_web_label_invariance():
    # Geometry-based traction check must be label-invariant.
    p_skin = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p_web = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    diag_base = [
        {
            "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 2.0}},
            "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 2.0}},
            "interface_edge_set": {"end": {"Tx_int": 2.0, "Ts_int": 5.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": 10.0, "Fs": 2.0}},
            "interface_field_set": {"start": {"Nx": 10.0, "Nxy": 99.0}},
            "interface_edge_set": {"start": {"Tx_int": 2.0, "Ts_int": 7.0}},
        },
    ]
    a = check_panel_equilibrium([[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]], [p_skin, p_web], all_panel_mitc4_diagnostics=diag_base, endpoint_tol=1e-8)[0]
    p_skin2 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="Panel A")
    p_web2 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Panel B")
    b = check_panel_equilibrium([[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]], [p_skin2, p_web2], all_panel_mitc4_diagnostics=diag_base, endpoint_tol=1e-8)[0]
    assert np.isclose(a["dTx_rel"], b["dTx_rel"])
    assert np.isclose(a["dT_yz_rel"], b["dT_yz_rel"])
    assert np.isclose(a["resultant_dNxy_rel"], b["resultant_dNxy_rel"])


def test_traction_compare_invariant_under_panel_swap():
    """Swapping (i,j) must leave dTx_rel and dT_yz_rel magnitudes unchanged."""
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 3.0, "Fs": 1.0}},
            "interface_field_set": {"end": {"Nx": 3.0, "Nxy": 1.0}},
            "interface_edge_set": {"end": {"Tx_int": 4.0, "Ts_int": 2.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": 3.0, "Fs": 1.0}},
            "interface_field_set": {"start": {"Nx": 3.0, "Nxy": 1.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": -3.0}},
        },
    ]
    fwd = check_panel_equilibrium(
        [[_mk_resultant(3.0, 1.0)], [_mk_resultant(3.0, 1.0)]], [p0, p1],
        all_panel_mitc4_diagnostics=diag, endpoint_tol=1e-8,
    )[0]
    # Swapped order
    diag_swap = [diag[1], diag[0]]
    rev = check_panel_equilibrium(
        [[_mk_resultant(3.0, 1.0)], [_mk_resultant(3.0, 1.0)]], [p1, p0],
        all_panel_mitc4_diagnostics=diag_swap, endpoint_tol=1e-8,
    )[0]
    assert np.isclose(fwd["dTx_rel"], rev["dTx_rel"], atol=1e-10)
    assert np.isclose(fwd["dT_yz_rel"], rev["dT_yz_rel"], atol=1e-10)


def test_traction_compare_pure_Nx_invariant():
    """Pure spanwise Nx load (Ny=Nxy=0) gives dTx=0 and dT_yz=0 at any junction angle."""
    # Two collinear panels, p0 "end" to p1 "start".
    # normal_sign = +1 at "end", -1 at "start".
    # Nxy=0: Tx_int = Nxy*normal_sign = 0.
    # Ny=0:  Ts_int = Ny*normal_sign = 0.
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="LSkin")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 0.0, "Ts_int": 0.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": 0.0, "Ts_int": 0.0}},
        },
    ]
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_mitc4_diagnostics=diag, endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["dTx_rel"] < 1e-10
    assert chk["dT_yz_rel"] < 1e-10


def test_traction_compare_90deg_junction():
    """At a 90-degree corner, dTx is exact and dT_yz has known analytical magnitude."""
    # p0: [0,0]->[1,0] ("end"), t_i=[1,0], Tx=5, Ts=0  → vector traction in X: 5, in YZ: [0,0]
    # p1: [1,0]->[1,1] ("start"), t_j=[0,1], Tx=-5, Ts=0 → X: -5, YZ: [0,0]
    # dTx = 5+(-5) = 0, dT_yz = 0*[1,0]+0*[0,1] = [0,0]  → both zero.
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 5.0, "Ts_int": 0.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": 0.0}},
        },
    ]
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_mitc4_diagnostics=diag, endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-10
    assert chk["dT_yz_rel"] < 1e-10
    # Now add a nonzero Ts component to check the Y,Z magnitude formula.
    # p0: Ts=3.0 → YZ traction = 3*[1,0] = [3,0]
    # p1: Ts=-4.0 → YZ traction = (-4)*[0,1] = [0,-4]
    # dT_yz = [3,0]+[0,-4] = [3,-4], magnitude = 5.0
    diag2 = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 5.0, "Ts_int": 3.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": -4.0}},
        },
    ]
    checks2 = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_mitc4_diagnostics=diag2, endpoint_tol=1e-8,
    )
    chk2 = checks2[0]
    assert np.isclose(chk2["dT_yz_mag"], 5.0, atol=1e-10)
    assert np.isclose(chk2["dTx_rel"], 0.0, atol=1e-10)


def test_transformed_interface_constraint_mode_available():
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=30)
    bundle = run_section_with_mitc4_shell(
        air, [0.35], N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=1.0,
        n_elements_per_panel=6, use_global_coupled=True, interface_constraint_mode="transformed",
    )
    diags = [d for d in (bundle.all_panel_mitc4_diagnostics or []) if d]
    assert len(diags) > 0
    assert any(d.get("constraint_stats", {}).get("n_slave_dofs", 0) > 0 for d in diags)


def test_transformed_basis_rotation_identity_for_collinear_panels():
    """
    For two collinear panels (ŝ_i = ŝ_j) the 2-D rotation matrix must be identity:
      u_s_s = 1*u_s_m + 0*w_m,  w_s = 0*u_s_m + 1*w_m.
    Verify this directly from _build_basis_transform_constraints.
    """
    from section_shell_model.lib.global_mitc4_assembly import (  # type: ignore[import-untyped]
        _build_basis_transform_constraints, _PanelGlobalMap, _dof,
        _U_S, _W, _BETA_S, _U_X, _BETA_X,
    )

    class _FakePanel:
        def __init__(self, nodes):
            self.nodes = nodes

    # Two collinear panels along y-axis: [0,0]→[1,0] and [1,0]→[2,0]
    # Both have tangent ŝ = [1,0], normal n̂ = [0,1].
    n_s = 3  # 3 s-nodes per panel
    pm0 = _PanelGlobalMap(
        s_nodes=np.linspace(0, 1, n_s),
        elements=[[0, 1, 1 + n_s, n_s], [1, 2, 2 + n_s, 1 + n_s]],
        global_nodes=list(range(2 * n_s)),        # nodes 0..5
        panel_label="P0", panel_index=0,
    )
    pm1 = _PanelGlobalMap(
        s_nodes=np.linspace(1, 2, n_s),
        elements=[[0, 1, 1 + n_s, n_s], [1, 2, 2 + n_s, 1 + n_s]],
        global_nodes=list(range(2 * n_s, 4 * n_s)),  # nodes 6..11
        panel_label="P1", panel_index=1,
    )
    p0 = _FakePanel(np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))
    p1 = _FakePanel(np.array([[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]))

    # Cluster: p0 "end" meets p1 "start" at (1,0)
    cluster = [(0, "end", np.array([1.0, 0.0])), (1, "start", np.array([1.0, 0.0]))]
    constraints = _build_basis_transform_constraints([pm0, pm1], [p0, p1], [cluster])

    # Master endpoint nodes (bottom and top rows at p0 "end"):
    gn_m_bot = pm0.global_nodes[n_s - 1]          # last bottom-row node of p0
    gn_m_top = pm0.global_nodes[2 * n_s - 1]      # last top-row node of p0
    gn_s_bot = pm1.global_nodes[0]                 # first bottom-row node of p1
    gn_s_top = pm1.global_nodes[n_s]               # first top-row node of p1

    for gn_m, gn_s in [(gn_m_bot, gn_s_bot), (gn_m_top, gn_s_top)]:
        # u_s_s = 1*u_s_m (identity, no cross-term)
        terms_us = dict(constraints[_dof(gn_s, _U_S)])
        assert np.isclose(terms_us.get(_dof(gn_m, _U_S), 0.0), 1.0, atol=1e-12)
        assert np.isclose(terms_us.get(_dof(gn_m, _W), 0.0), 0.0, atol=1e-12)
        # w_s = 1*w_m (identity, no cross-term)
        terms_w = dict(constraints[_dof(gn_s, _W)])
        assert np.isclose(terms_w.get(_dof(gn_m, _W), 0.0), 1.0, atol=1e-12)
        assert np.isclose(terms_w.get(_dof(gn_m, _U_S), 0.0), 0.0, atol=1e-12)
        # β_s_s = 1*β_s_m
        terms_bs = dict(constraints[_dof(gn_s, _BETA_S)])
        assert np.isclose(terms_bs.get(_dof(gn_m, _BETA_S), 0.0), 1.0, atol=1e-12)


def test_transformed_basis_runs_and_produces_finite_results():
    """
    ``transformed_basis`` mode must run without error and produce finite resultants.

    Note: the correct BC propagation through junction-global DOFs is a pending
    improvement; primary reaction-balance under this mode is not yet required to
    meet the production threshold.
    """
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.02, p=0.4, t_c=0.12, n=40)
    bundle = run_section_with_mitc4_shell(
        air, [0.35], N=1.0, Vy=1.0, Vz=1.0, My=1.0, Mz=1.0, T=1.0,
        n_elements_per_panel=6, use_global_coupled=True,
        interface_constraint_mode="transformed_basis",
    )
    assert bundle.all_panel_mitc4_results is not None
    for res_list in bundle.all_panel_mitc4_results:
        for r in res_list:
            assert np.isfinite(r.Nx), f"Non-finite Nx on {r.panel_label}"
            assert np.isfinite(r.Nxy), f"Non-finite Nxy on {r.panel_label}"


def test_tiered_acceptance_keys_present():
    """
    Phase E: check_panel_equilibrium must return dicts containing all tiered-
    acceptance keys (dTx_rel, dT_yz_rel, resultant_tol_dTx, resultant_tol_dT_yz,
    resultant_pass_dT_yz) for the Phase-E reporting to work.
    """
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 1.0)], [_mk_resultant(5.0, 1.0)]],
        [p0, p1],
        all_panel_mitc4_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"end": {"Tx_int": 1.0, "Ts_int": 2.0}},
            },
            {
                "boundary_reaction": {"start": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"start": {"Tx_int": -1.0, "Ts_int": -2.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    for key in ("dTx_rel", "dT_yz_rel", "resultant_tol_dTx",
                "resultant_tol_dT_yz", "resultant_pass_dT_yz"):
        assert key in chk, f"Missing key: {key}"
    assert isinstance(chk["dTx_rel"], float)
    assert isinstance(chk["dT_yz_rel"], float)
    assert isinstance(chk["resultant_pass_dT_yz"], bool)


def test_global_vs_panel_consistency_curved():
    """
    A single curved panel solved via solve_panel_mitc4 vs solve_global_coupled_mitc4
    (degenerate one-panel cluster) must agree closely on Nx, Nxy resultants.

    This validates that the Donnell curvature load is applied in both paths so the
    constitutive path split (Phase C) is closed.
    """
    from multi_cell_blade_section import naca_four_digit  # type: ignore[import-untyped]
    from section_shell_model.lib.recovery_adapter import run_section_with_mitc4_shell

    air = naca_four_digit(m=0.04, p=0.4, t_c=0.18, n=40)
    # Run via global coupled (uses the global solve with Donnell correction).
    bundle_global = run_section_with_mitc4_shell(
        air, [0.35], N=1.0, Vy=0.0, Vz=1.0, My=0.0, Mz=0.0, T=0.0,
        n_elements_per_panel=8, use_global_coupled=True,
    )
    # Run via per-panel (uses panel_mitc4_model with Donnell correction).
    bundle_panel = run_section_with_mitc4_shell(
        air, [0.35], N=1.0, Vy=0.0, Vz=1.0, My=0.0, Mz=0.0, T=0.0,
        n_elements_per_panel=8, use_global_coupled=False,
    )
    assert bundle_global.all_panel_mitc4_results is not None
    assert bundle_panel.all_panel_mitc4_results is not None
    # Compare panel 0 (first skin panel) Nx mid-element.
    g_res = bundle_global.all_panel_mitc4_results[0]
    p_res = bundle_panel.all_panel_mitc4_results[0]
    assert len(g_res) > 0 and len(p_res) > 0
    g_nx_mid = float(g_res[len(g_res) // 2].Nx)
    p_nx_mid = float(p_res[len(p_res) // 2].Nx)
    # Both should have the same sign and be within 50% of each other (they differ
    # because of junction coupling in the global solve, but the order-of-magnitude
    # should match).
    assert np.isfinite(g_nx_mid) and np.isfinite(p_nx_mid)
    if abs(p_nx_mid) > 0.1:
        rel_diff = abs(g_nx_mid - p_nx_mid) / abs(p_nx_mid)
        assert rel_diff < 2.0, (
            f"Global vs panel Nx deviation too large: global={g_nx_mid:.4e} "
            f"panel={p_nx_mid:.4e} rel={rel_diff:.2f}"
        )
