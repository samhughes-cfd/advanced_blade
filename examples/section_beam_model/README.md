# Section beam model (GBT example)

Cross-section Generalised Beam Theory: strip mesh, prebuckling, cross-section modes, member buckling.

## Layout

| Path | Role |
|------|------|
| `gbt/` | GBT implementation |
| `tests/` | Pytest suite |

## Run tests

From the repository root (``examples`` must be on ``PYTHONPATH``):

```bash
pytest examples/section_beam_model/tests -q
```

Or rely on [pyproject.toml](../../pyproject.toml) ``pythonpath`` / ``testpaths`` after configuring pytest for this repo.

## Related

- Loads → JSON/plots bridge: [examples/section_buckling](../section_buckling).
