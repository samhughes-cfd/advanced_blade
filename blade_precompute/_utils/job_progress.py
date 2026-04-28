"""Job-level live progress for precompute runs.

Human-readable lines go to stdout with prefix ``[precompute]``. Append-only
``<job_dir>/progress.jsonl`` records the same milestones for ``tail -f`` or
downstream tooling.

Each JSON line is one object with:

- ``ts``: ISO8601 UTC (millisecond precision, ``Z`` suffix)
- ``kind``: ``phase_start`` | ``phase_end`` | ``event`` | ``run_log``
- ``phase``: logical phase name (e.g. ``section_geometry``, ``run_log`` rows use
  the package name as ``phase``)
- ``elapsed_total_s``: wall seconds since :class:`JobProgressReporter` construction
- ``elapsed_phase_s``: wall seconds since the innermost open ``phase_start`` (when
  applicable)
- ``status``: optional (e.g. ``ok`` on ``phase_end``)
- ``meta``: JSON-serializable dict of caller-supplied fields

Environment:

- ``ADVANCED_BLADE_LIVE_PROGRESS=0|false|off`` — disable stdout lines and skip
  writing ``progress.jsonl`` when the reporter is constructed with
  ``enabled=None`` (see :func:`live_progress_enabled_from_env`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from blade_precompute._utils.jsonutil import to_jsonable

_PREFIX = "[precompute]"

# Public alias for the stdout prefix (e.g. final `print` in main_precompute).
CONSOLE_LOG_PREFIX = _PREFIX


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def live_progress_enabled_from_env() -> bool:
    v = os.environ.get("ADVANCED_BLADE_LIVE_PROGRESS", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True


def mirror_run_log_progress_from_env() -> bool:
    v = os.environ.get("ADVANCED_BLADE_PROGRESS_MIRROR_RUNLOG", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class JobProgressReporter:
    """Stdout + ``progress.jsonl`` for one job directory."""

    def __init__(self, job_dir: Path, *, enabled: bool = True) -> None:
        self._job_dir = Path(job_dir).resolve()
        self._enabled = bool(enabled)
        self._t0 = time.perf_counter()
        self._jsonl_path = self._job_dir / "progress.jsonl"
        self._phase_stack: list[tuple[str, float]] = []

    @property
    def job_dir(self) -> Path:
        return self._job_dir

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _elapsed_total(self) -> float:
        return float(time.perf_counter() - self._t0)

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_jsonable(record), separators=(",", ":")) + "\n"
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _elapsed_since_innermost_phase(self) -> float | None:
        if not self._phase_stack:
            return None
        _, t_p = self._phase_stack[-1]
        return float(time.perf_counter() - t_p)

    def _emit(
        self,
        *,
        kind: str,
        phase: str,
        echo: bool = True,
        elapsed_phase_s: float | None = None,
        status: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        elapsed_total_s = self._elapsed_total()
        meta_clean = to_jsonable(dict(meta or {}))
        rec: dict[str, Any] = {
            "ts": _utc_iso(),
            "kind": kind,
            "phase": phase,
            "elapsed_total_s": round(elapsed_total_s, 3),
            "meta": meta_clean,
        }
        if elapsed_phase_s is not None:
            rec["elapsed_phase_s"] = round(float(elapsed_phase_s), 3)
        if status is not None:
            rec["status"] = status
        self._write_jsonl(rec)
        if not echo or not self._enabled:
            return
        parts = [
            _PREFIX,
            kind,
            phase,
            f"total={elapsed_total_s:.1f}s",
        ]
        if elapsed_phase_s is not None:
            parts.append(f"phase={elapsed_phase_s:.1f}s")
        if status:
            parts.append(f"status={status}")
        if meta_clean:
            brief: list[str] = []
            for k in sorted(meta_clean.keys())[:8]:
                v = meta_clean[k]
                if isinstance(v, float):
                    brief.append(f"{k}={v:.4g}" if abs(v) < 1e5 else f"{k}={v:.3e}")
                else:
                    s = str(v)
                    brief.append(f"{k}={s[:48]}" + ("…" if len(s) > 48 else ""))
            if brief:
                parts.append(" ".join(brief))
        print(" | ".join(str(p) for p in parts), file=sys.stdout, flush=True)

    def phase_start(self, phase: str, **meta: Any) -> None:
        ep_parent: float | None = None
        if self._phase_stack:
            _, t_parent = self._phase_stack[-1]
            ep_parent = float(time.perf_counter() - t_parent)
        self._phase_stack.append((phase, time.perf_counter()))
        meta2 = dict(meta)
        if ep_parent is not None:
            meta2["since_parent_phase_s"] = round(ep_parent, 3)
        self._emit(
            kind="phase_start",
            phase=phase,
            elapsed_phase_s=ep_parent,
            meta=meta2,
        )

    def phase_end(self, phase: str, **meta: Any) -> None:
        elapsed_phase: float | None = None
        if self._phase_stack:
            # Pop innermost matching phase (tolerate balanced nesting).
            for i in range(len(self._phase_stack) - 1, -1, -1):
                if self._phase_stack[i][0] == phase:
                    _, t_p = self._phase_stack.pop(i)
                    elapsed_phase = float(time.perf_counter() - t_p)
                    break
        merged = dict(meta)
        if elapsed_phase is not None:
            merged.setdefault("duration_s", round(elapsed_phase, 3))
        self._emit(
            kind="phase_end",
            phase=phase,
            elapsed_phase_s=elapsed_phase,
            status="ok",
            meta=merged,
        )

    def event(self, phase: str, **meta: Any) -> None:
        """Point-in-time milestone (stdout + jsonl when enabled)."""
        ep = self._elapsed_since_innermost_phase()
        self._emit(kind="event", phase=phase, elapsed_phase_s=ep, meta=meta)

    def event_jsonl_only(self, phase: str, **meta: Any) -> None:
        """Append ``progress.jsonl`` only (no stdout)."""
        ep = self._elapsed_since_innermost_phase()
        self._emit(kind="event", phase=phase, elapsed_phase_s=ep, echo=False, meta=meta)

    def mirror_run_log(
        self,
        *,
        package: str,
        event: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Structured mirror of :class:`RunLogger` INFO rows (jsonl only)."""
        ep = self._elapsed_since_innermost_phase()
        rec: dict[str, Any] = {
            "ts": _utc_iso(),
            "kind": "run_log",
            "phase": package,
            "elapsed_total_s": round(self._elapsed_total(), 3),
            "meta": {"event": event, "payload": to_jsonable(dict(payload or {}))},
        }
        if ep is not None:
            rec["elapsed_phase_s"] = round(ep, 3)
        self._write_jsonl(rec)


def throttled_emit_station_index(i: int, n: int) -> bool:
    """When True, emit a console line for station index ``i`` (0..n-1) of ``n`` total.

    All stations if ``n <= 20``; otherwise first, last, and ~every 20% along the list.
    """
    if n <= 0:
        return False
    if n <= 20:
        return True
    if i == 0 or i == n - 1:
        return True
    step = max(1, n // 5)
    return (i % step) == 0


def emit_progress_event(progress: Any, phase: str, **meta: Any) -> None:
    """Call ``JobProgressReporter.event`` if ``progress`` is set and enabled."""
    if progress is None or not getattr(progress, "enabled", True):
        return
    progress.event(phase, **meta)
