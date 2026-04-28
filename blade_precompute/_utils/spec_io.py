"""Mapping IO for blade_precompute specs (JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a top-level mapping spec from JSON."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a top-level mapping/object.")
    return raw
