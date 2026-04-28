"""Pytest hooks: headless-safe matplotlib before any test imports pyplot."""

from __future__ import annotations


def pytest_configure() -> None:
    import matplotlib

    matplotlib.use("Agg")
