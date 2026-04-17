"""Gauss→nodal projector: linear recovery and integration with beam solve."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.global_beam_model.engine.element import element_gauss_shape_matrix
from blade_precompute.global_beam_model.engine.nodal_result_projector import (
    _recover_nodal_from_gauss,
)


def test_recover_nodal_linear_exact_for_two_gauss_points():
    """2×2 N_mat: nodal values reproduce Gauss samples for linear axial fields."""
    _, _, N_mat = element_gauss_shape_matrix(2)
    u_nodes = np.array([1.25, -0.5], dtype=np.float64)
    values_g = (N_mat @ u_nodes).reshape(-1, 1)
    values_g = np.hstack([values_g] * 7)
    recovered = _recover_nodal_from_gauss(N_mat, values_g)
    np.testing.assert_allclose(recovered[:, 0], u_nodes, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered, np.outer(u_nodes, np.ones(7)), rtol=0, atol=1e-12)


def test_recover_nodal_lstsq_overdetermined_three_gauss():
    _, _, N_mat = element_gauss_shape_matrix(3)
    u_nodes = np.array([2.0, 4.0], dtype=np.float64)
    values_g = (N_mat @ u_nodes).reshape(-1, 1)
    values_g = np.hstack([values_g] * 7)
    recovered = _recover_nodal_from_gauss(N_mat, values_g)
    np.testing.assert_allclose(recovered[:, 0], u_nodes, rtol=0, atol=1e-10)
