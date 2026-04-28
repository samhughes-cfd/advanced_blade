# Run Logging Architecture

## Purpose

`blade_precompute` writes structured run logs so every stage outcome, tensor shape, and exported artifact is traceable after a job finishes. This supports automated debugging workflows and post-run diagnostics without re-running computations.

## Output Layout

For job directory `outputs/<job>/`:

- `precompute.run.log`: top-level orchestration log (optional, caller-owned)
- `logs.manifest.json`: job-level index of all event and artifact records
- `section_geometry/run.log`
- `section_properties/run.log`
- `section_shell_model/run.log`
- `global_beam_model/run.log`
- `section_optimisation/run.log`

Each package directory may also include:

- `run.jsonl`: machine-readable event stream
- `logs.manifest.json`: package-local event and artifact records
- `arrays/*.npy`: tensor dumps when `log_dump_level == "full"`

## Event Vocabulary

Events are written as structured records (`run.jsonl`) and mirrored to text (`run.log`):

- `logger.init`
- `stage.start`
- `stage.end`
- `stage.error`
- `stage.skipped`
- `iteration`
- `tensor`
- `artefact`

## Tensor Logging

`RunLogger.log_tensor(...)` records:

- tensor name
- shape
- dtype
- finite summary (`min`, `max`, `mean`, `has_nan`)
- optional `dump_path` (when full dumps are enabled)

## Configuration

`PrecomputeInputs` supports:

- `log_dump_level`: `summary | intermediate | full`
- `log_dir`: optional override root (reserved for external runners)

`main_precompute.py` sets `log_dump_level` via `LOG_DUMP_LEVEL`.

## Manifest Schema

Each manifest entry includes:

- `kind` (event/artifact)
- `path` and optional `jsonl_path`
- `package`
- `stage`
- `station`
- `event`
- `payload`

Both package-level and job-level manifests append the same canonical entries.
