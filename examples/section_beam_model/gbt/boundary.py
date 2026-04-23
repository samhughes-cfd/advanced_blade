"""
boundary.py — Boundary condition specification for the GBT member solver.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class EndCondition:
    w_fixed:     bool  = False
    theta_fixed: bool  = False
    k_trans:     float = 0.0
    k_rot:       float = 0.0


@dataclass
class BoundaryConditions:
    start: EndCondition = field(default_factory=EndCondition)
    end:   EndCondition = field(default_factory=EndCondition)

    @classmethod
    def simply_supported(cls):
        return cls(start=EndCondition(w_fixed=True), end=EndCondition(w_fixed=True))

    @classmethod
    def clamped_free(cls):
        return cls(start=EndCondition(w_fixed=True, theta_fixed=True),
                   end=EndCondition())

    @classmethod
    def clamped_clamped(cls):
        return cls(start=EndCondition(w_fixed=True, theta_fixed=True),
                   end=EndCondition(w_fixed=True, theta_fixed=True))

    @classmethod
    def clamped_pinned(cls):
        return cls(start=EndCondition(w_fixed=True, theta_fixed=True),
                   end=EndCondition(w_fixed=True))

    @classmethod
    def elastic(cls, k_trans: float, k_rot: float = 0.0):
        ec = EndCondition(k_trans=k_trans, k_rot=k_rot)
        return cls(start=ec, end=ec)

    def constrained_dofs(self, n_nodes_per_mode: int, n_modes: int) -> list[int]:
        constrained = []
        n_dof_mode  = 2 * n_nodes_per_mode
        for m in range(n_modes):
            base = m * n_dof_mode
            if self.start.w_fixed:     constrained.append(base)
            if self.start.theta_fixed: constrained.append(base + 1)
            if self.end.w_fixed:       constrained.append(base + n_dof_mode - 2)
            if self.end.theta_fixed:   constrained.append(base + n_dof_mode - 1)
        return constrained

    def spring_contributions(self, n_nodes_per_mode: int, n_modes: int):
        n_dof_mode = 2 * n_nodes_per_mode
        n_total    = n_dof_mode * n_modes
        K_s = np.zeros((n_total, n_total))
        for m in range(n_modes):
            base = m * n_dof_mode
            if self.start.k_trans > 0: K_s[base, base]           += self.start.k_trans
            if self.start.k_rot > 0:   K_s[base+1, base+1]       += self.start.k_rot
            i_w = base + n_dof_mode - 2
            if self.end.k_trans > 0:   K_s[i_w, i_w]             += self.end.k_trans
            if self.end.k_rot > 0:     K_s[i_w+1, i_w+1]         += self.end.k_rot
        # Second return value (geometric spring contribution) is always zero
        # and reserved for future elastic foundation Kg terms.
        return K_s, np.zeros_like(K_s)
