"""Former ``stress_recovery`` / ``recovery_operators`` packages must not resolve."""

from __future__ import annotations

import importlib

import pytest


def test_stress_recovery_package_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("blade_utilities.stress_recovery")


def test_recovery_operators_package_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("blade_utilities.recovery_operators")
