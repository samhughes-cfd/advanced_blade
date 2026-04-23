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

The ``section_buckling`` stage inside ``blade_precompute`` orchestration is a **stub**; it does not import this package. Use this example code directly for GBT buckling workflows.
