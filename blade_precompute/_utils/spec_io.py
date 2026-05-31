"""Mapping IO for blade_precompute specs (JSON or YAML)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a top-level mapping spec from JSON or YAML."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency packaging guard
            raise RuntimeError("PyYAML is required to load YAML specs.") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, dict):
        raise ValueError(f"{p} must contain a top-level mapping/object.")
    return raw
