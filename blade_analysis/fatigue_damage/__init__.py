"""Vectorised fatigue stress recovery and Miner damage (Prompt 3c).

Run: ``python -m blade_analysis.fatigue_damage``
"""

from .api import FatigueAnalysis
from .core.loads import ResultantHistory, StressHistory
from .core.workflows import ExtremeWorkflowSpec, OperationalWorkflowSpec, validate_shared_calibration
from .core.types import FatigueResult, RainflowBins
from .engine.conversion import (
    beam_resultants_to_cache_order,
    resultants_to_stress_history,
    resultants_to_stress_history_lazy,
    stress_history_memory_mb,
)
from .engine.damage import life_from_damage, miner_damage
from .engine.pipeline import FatiguePipeline
from .engine.rainflow import (
    IncrementalRainflowAccumulator,
    count_cycles_iso_vm,
    count_cycles_ply_stresses,
)
from .engine.sn_curves import SNcurve

__all__ = [
    "FatigueAnalysis",
    "FatiguePipeline",
    "FatigueResult",
    "IncrementalRainflowAccumulator",
    "RainflowBins",
    "ResultantHistory",
    "SNcurve",
    "StressHistory",
    "ExtremeWorkflowSpec",
    "OperationalWorkflowSpec",
    "beam_resultants_to_cache_order",
    "count_cycles_iso_vm",
    "count_cycles_ply_stresses",
    "life_from_damage",
    "miner_damage",
    "resultants_to_stress_history",
    "resultants_to_stress_history_lazy",
    "stress_history_memory_mb",
    "validate_shared_calibration",
]
