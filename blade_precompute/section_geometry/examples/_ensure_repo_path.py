"""Ensure the repository root is on ``sys.path`` when examples are run as scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_path() -> Path:
    """Insert the checkout root (parent of ``blade_precompute/``) first on ``sys.path``.

    Raises
    ------
    RuntimeError
        If no ancestor of this file contains ``blade_precompute/__init__.py``.
    """
    for p in Path(__file__).resolve().parents:
        if (p / "blade_precompute" / "__init__.py").is_file():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return p
    raise RuntimeError(
        "Could not find the blade_precompute package. Clone the repository and run this "
        "script from inside it, or install the project in editable mode "
        "(from the repo root: pip install -e .)."
    )
