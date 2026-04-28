"""Structured run logging helpers for precompute jobs."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .jsonutil import to_jsonable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "finite": True, "has_nan": False}
    finite = np.isfinite(arr)
    finite_all = bool(np.all(finite))
    has_nan = bool(np.any(np.isnan(arr)))
    if finite_all:
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "finite": True,
            "has_nan": has_nan,
        }
    return {"min": None, "max": None, "mean": None, "finite": False, "has_nan": has_nan}


@dataclass(frozen=True)
class _LogTarget:
    package: str
    package_dir: Path
    log_path: Path
    jsonl_path: Path
    package_manifest_path: Path
    job_manifest_path: Path
    package_manifest_jsonl_path: Path
    job_manifest_jsonl_path: Path


_DUMP_LEVEL_RANK = {
    "none": 0,
    "summary": 1,
    "intermediate": 2,
    "full": 3,
}


_PROGRESS_MIRROR_EVENTS = frozenset({"stage.start", "stage.end", "evaluation.initial", "optimizer.final"})


def _event_level(event: str) -> int:
    if event in {"logger.init", "logger.close", "stage.start", "stage.end", "stage.error"}:
        return 0
    if event in {"evaluation.initial", "optimizer.final", "artefact"}:
        return 1
    if event == "tensor":
        return 3
    return 2


class RunLogger(AbstractContextManager["RunLogger"]):
    """Per-package logger with text/jsonl outputs and manifest registration."""

    def __init__(
        self,
        *,
        package: str,
        job_dir: Path,
        station_subdir: str | None = None,
        dump_level: str = "intermediate",
        progress_reporter: Any | None = None,
        mirror_progress_jsonl: bool = False,
    ) -> None:
        self._package = package
        self._job_dir = Path(job_dir).resolve()
        self._station_subdir = station_subdir
        self._dump_level = str(dump_level).lower().strip()
        if self._dump_level not in _DUMP_LEVEL_RANK:
            self._dump_level = "intermediate"
        self._stage: str | None = None
        self._records_written: int = 0
        self._progress_reporter = progress_reporter
        self._mirror_progress_jsonl = bool(mirror_progress_jsonl)

        package_dir = (self._job_dir / package).resolve()
        if station_subdir:
            package_dir = (package_dir / station_subdir).resolve()
        package_dir.mkdir(parents=True, exist_ok=True)
        self._target = _LogTarget(
            package=package,
            package_dir=package_dir,
            log_path=(package_dir / "run.log").resolve(),
            jsonl_path=(package_dir / "run.jsonl").resolve(),
            package_manifest_path=((self._job_dir / package).resolve() / "logs.manifest.json").resolve(),
            job_manifest_path=(self._job_dir / "logs.manifest.json").resolve(),
            package_manifest_jsonl_path=((self._job_dir / package).resolve() / "logs.manifest.jsonl").resolve(),
            job_manifest_jsonl_path=(self._job_dir / "logs.manifest.jsonl").resolve(),
        )
        self.info_event("logger.init", dump_level=dump_level, station_subdir=station_subdir)

    @property
    def package(self) -> str:
        return self._package

    @property
    def dump_level(self) -> str:
        return self._dump_level

    @property
    def package_output_dir(self) -> Path:
        """Resolved directory for this package (e.g. ``.../section_optimisation``)."""
        return self._target.package_dir

    def scope(self, stage: str) -> "RunLogger":
        self._stage = stage
        self.info_event("stage.start", stage=stage)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._stage is not None:
            if exc is None:
                self.info_event("stage.end", stage=self._stage, status="ok")
            else:
                self.info_event("stage.error", stage=self._stage, status="error", error=str(exc))
        self.info_event("logger.close")
        self._write_manifest_index()
        return False

    def _should_emit(self, event: str) -> bool:
        return _event_level(event) <= _DUMP_LEVEL_RANK[self._dump_level]

    def _write_manifest_entry(self, payload: dict[str, Any]) -> None:
        line = json.dumps(to_jsonable(payload), separators=(",", ":")) + "\n"
        for path in (self._target.package_manifest_jsonl_path, self._target.job_manifest_jsonl_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

    def _write_manifest_index(self) -> None:
        payload = {
            "version": 2,
            "package": self._package,
            "dump_level": self._dump_level,
            "records_written": int(self._records_written),
            "run_log": str(self._target.log_path),
            "run_jsonl": str(self._target.jsonl_path),
            "manifest_jsonl": str(self._target.package_manifest_jsonl_path),
            "job_manifest_jsonl": str(self._target.job_manifest_jsonl_path),
            "closed_at": _utc_now_iso(),
        }
        for path in (self._target.package_manifest_path, self._target.job_manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")

    def _write_record(self, level: str, event: str, payload: dict[str, Any]) -> None:
        if not self._should_emit(event):
            return
        ts = _utc_now_iso()
        record = {
            "ts": ts,
            "level": level,
            "package": self._package,
            "stage": self._stage,
            "station": self._station_subdir,
            "event": event,
            "payload": payload,
        }
        text = f"{ts} {level:<5} [{self._package}] {event} {json.dumps(to_jsonable(payload), separators=(',', ':'))}\n"
        self._target.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._target.log_path.open("a", encoding="utf-8") as f:
            f.write(text)
        with self._target.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), separators=(",", ":")) + "\n")
        self._records_written += 1
        self._write_manifest_entry(
            {
                "kind": "event",
                "path": str(self._target.log_path),
                "jsonl_path": str(self._target.jsonl_path),
                "package": self._package,
                "stage": self._stage,
                "station": self._station_subdir,
                "event": event,
                "payload": payload,
            }
        )
        if (
            level == "INFO"
            and self._mirror_progress_jsonl
            and self._progress_reporter is not None
            and bool(getattr(self._progress_reporter, "enabled", True))
            and event in _PROGRESS_MIRROR_EVENTS
        ):
            self._progress_reporter.mirror_run_log(package=self._package, event=event, payload=payload)

    def info_event(self, event: str, **payload: Any) -> None:
        self._write_record("INFO", event, payload)

    def log_iteration(self, step: int, **metrics: Any) -> None:
        self._write_record("DEBUG", "iteration", {"step": int(step), **metrics})

    def log_artefact(self, path: Path, kind: str, **payload: Any) -> None:
        self._write_record(
            "INFO",
            "artefact",
            {"kind": kind, "path": str(Path(path).resolve()), **payload},
        )

    def log_tensor(
        self,
        name: str,
        arr: np.ndarray,
        *,
        step: str = "",
        dump: bool | None = None,
    ) -> Path | None:
        a = np.asarray(arr)
        stats = _safe_stats(a.astype(np.float64, copy=False))
        payload = {
            "name": name,
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            **stats,
        }
        if step:
            payload["step"] = step
        do_dump = dump if dump is not None else (self._dump_level == "full")
        out_path: Path | None = None
        if do_dump:
            shape_txt = "x".join(str(x) for x in a.shape) if a.shape else "scalar"
            step_txt = f"{step}__" if step else ""
            out_path = (self._target.package_dir / "arrays" / f"{step_txt}{name}__{shape_txt}.npy").resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, a)
            payload["dump_path"] = str(out_path)
        self._write_record("DEBUG", "tensor", payload)
        return out_path


def get_run_logger(
    *,
    package: str,
    job_dir: Path,
    station_subdir: str | None = None,
    dump_level: str = "intermediate",
    progress_reporter: Any | None = None,
    mirror_progress_jsonl: bool = False,
) -> RunLogger:
    return RunLogger(
        package=package,
        job_dir=job_dir,
        station_subdir=station_subdir,
        dump_level=dump_level,
        progress_reporter=progress_reporter,
        mirror_progress_jsonl=mirror_progress_jsonl,
    )
