"""Proxy — see ``blade_precompute.global_beam_model.engine.interp``."""

from blade_precompute.global_beam_model.engine.interp import (
    interp_K7,
    interp_matrix,
    interp_scalar_stations,
    sample_field_at_z,
    stations_from_arrays,
)

__all__ = [
    "interp_matrix",
    "interp_K7",
    "interp_scalar_stations",
    "sample_field_at_z",
    "stations_from_arrays",
]
