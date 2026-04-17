"""JSON-line logging helpers for the fatigue pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping


def log_json(logger: logging.Logger, level: int, event: str, fields: Mapping[str, Any] | None = None) -> None:
    """Emit a single JSON object as the log message (``event`` + optional fields)."""
    payload: dict[str, Any] = {"event": event}
    if fields:
        for k, v in fields.items():
            payload[k] = v
    logger.log(level, "%s", json.dumps(payload, default=str))
