"""Operational vs extreme workflow contracts and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .loads import ResultantHistory


@dataclass(frozen=True)
class ExtremeWorkflowSpec:
    """
    Metadata for the ultimate/characteristic envelope path.

    The envelope itself is solved upstream (Tier A/Tier B). This spec keeps a
    calibration tag so operational fatigue can assert shared section/recovery
    calibration.
    """

    z_stations: NDArray[np.float64]
    calibration_tag: str


@dataclass(frozen=True)
class OperationalWorkflowSpec:
    """
    Operational fatigue path using prescribed resultant histories.
    """

    history: ResultantHistory
    calibration_tag: str


def validate_shared_calibration(
    extreme: ExtremeWorkflowSpec,
    operational: OperationalWorkflowSpec,
    *,
    cache_z_stations: NDArray[np.float64] | None = None,
) -> None:
    """
    Ensure operational and extreme workflows use consistent structural calibration.
    """
    if extreme.calibration_tag != operational.calibration_tag:
        raise ValueError(
            "Operational and extreme workflows use different calibration_tag values."
        )
    z_ext = np.asarray(extreme.z_stations, dtype=np.float64).ravel()
    z_op = np.asarray(operational.history.z_stations, dtype=np.float64).ravel()
    if z_ext.shape != z_op.shape or not np.allclose(z_ext, z_op, atol=1e-9, rtol=0.0):
        raise ValueError("Operational and extreme workflows must share identical z_stations.")
    if cache_z_stations is not None:
        zc = np.asarray(cache_z_stations, dtype=np.float64).ravel()
        if zc.shape != z_op.shape or not np.allclose(zc, z_op, atol=1e-9, rtol=0.0):
            raise ValueError("Recovery cache z_stations do not match operational history.")
