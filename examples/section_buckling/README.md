# Section buckling (example)

Bridge ``SectionDefinition`` / extreme loads from ``blade_precompute.section_properties`` to GBT workflows; station JSON and matplotlib plots.

## Dependencies

- ``blade_precompute`` (geometry, laminates, mesh)
- ``section_beam_model`` (``examples/section_beam_model``) on ``PYTHONPATH`` together with this package

## Run tests

```bash
pytest examples/section_buckling/tests -q
```

## Precompute

``blade_precompute`` orchestration does not run GBT buckling. Use this package (and ``examples/section_beam_model``) directly for GBT buckling workflows.
