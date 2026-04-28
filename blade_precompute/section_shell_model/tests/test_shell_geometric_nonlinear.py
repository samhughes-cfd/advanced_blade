"""Tests for linear buckling API (dense pencil) and nonlinear placeholder."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


def test_linear_buckling_smallest_eigenvalues_rank1() -> None:
    from blade_precompute.section_shell_model.lib.shell_geometric_nonlinear import (
        linear_buckling_smallest_eigenvalues,
        nonlinear_shell_solve_not_implemented,
    )

    # Proportional K and K_g share eigenvectors ⇒ one distinct buckling λ for each mode scale.
    n = 4
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    dk = np.linspace(10.0, 50.0, n)
    dkg = np.linspace(-2.1, -0.5, n)
    K = sp.csr_matrix(q @ np.diag(dk) @ q.T)
    K_g = sp.csr_matrix(q @ np.diag(dkg) @ q.T)

    lambdas, modes = linear_buckling_smallest_eigenvalues(K, K_g, nev=2)
    assert lambdas.shape[0] == 2
    assert modes.shape[0] == n
    assert np.all(np.isfinite(lambdas))

    with pytest.raises(NotImplementedError):
        nonlinear_shell_solve_not_implemented()
