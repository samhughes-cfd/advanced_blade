from __future__ import annotations

import json
from pathlib import Path

from blade_precompute._utils.job_progress import JobProgressReporter


def test_job_progress_reporter_writes_jsonl_and_phases(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    r = JobProgressReporter(job, enabled=True)
    r.phase_start("alpha", foo=1)
    r.phase_end("alpha", bar=2)
    r.event("ping", x=3)
    p = job / "progress.jsonl"
    assert p.is_file()
    lines = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[0]["kind"] == "phase_start"
    assert lines[0]["phase"] == "alpha"
    assert "elapsed_total_s" in lines[0]
    assert lines[1]["kind"] == "phase_end"
    assert lines[2]["kind"] == "event"
    assert lines[2]["phase"] == "ping"


def test_job_progress_reporter_disabled_skips_jsonl(tmp_path: Path) -> None:
    job = tmp_path / "job2"
    job.mkdir()
    r = JobProgressReporter(job, enabled=False)
    r.phase_start("x")
    r.phase_end("x")
    assert not (job / "progress.jsonl").exists()
