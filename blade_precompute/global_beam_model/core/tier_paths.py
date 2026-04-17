"""
Beam analysis tiers (spanwise stiffness → resultants / stresses).

**Tier A — Geometrically exact beam** (:mod:`beam_model.engine.solver`)
    Full nonlinear / linearised static solve on :class:`~beam_model.core.types.BeamModel`.
    Primary API: :class:`beam_model.api.BeamAnalysis`.

**Tier B — Prescribed section resultants** (:mod:`section_optimisation.engine.beam_k7`)
    Ultimate-envelope style internal forces taken as data; small-angle nodal frame
    from reference curvature via :func:`beam_model.engine.kinematics.rotmat_from_small_curvature`.
    Primary API: :class:`section_optimisation.core.protocols.PrescribedResultantDriver`.

**Tier C — Fused recovery** (:mod:`recovery_cache.engine.builder`)
    Precomputed linear maps from beam resultants to ply / isotropic stresses and cached
    failure tensors. Primary API: :class:`recovery_cache.api.RecoveryCacheBuilder`.
"""

__all__: list[str] = []
