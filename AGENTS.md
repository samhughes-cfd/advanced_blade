# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

`sandbox-beam` is a pure-Python scientific/engineering computation library for wind/tidal turbine blade structural analysis and optimisation. There are no external services, databases, or web servers — only a local Python numerical pipeline.

### Running tests

```bash
python3 -m pytest --tb=short -q
```

The full test suite is **very compute-intensive** (numerical optimisation solvers); expect 30+ minutes on a single-core VM. For faster feedback, run a targeted subset:

```bash
# Fast unit tests (~15 s)
python3 -m pytest tests/test_orchestration_inputs.py tests/test_material_library.py \
  tests/test_axial_loading.py tests/test_job_progress_reporter.py \
  blade_precompute/section_geometry/tests/ --tb=short -q

# Example package tests (~3 s)
python3 -m pytest examples/section_beam_model/tests/ examples/section_buckling/tests/ --tb=short -q
```

**Known issue:** 12 test files under `tests/` import `section_model.*` which does not exist as a top-level package (the classes live in `blade_precompute.section_properties`). These tests fail at collection with `ModuleNotFoundError`. This is a pre-existing repo issue, not an environment problem.

### Running the main pipeline

```bash
cd /workspace && python3 main_precompute.py
```

The default configuration in `main_precompute.py` runs a full blade structural optimisation (120 SLSQP iterations, 10 structural stations). This takes significant wall time. To do a quick smoke test, reduce `N_STRUCTURAL`, `DESIGN_MAX_ITER`, and set `DESIGN_OPTIMISE = False` in the module variables.

### Key caveats

- Use `python3` (not `python`) — there is no `python` symlink in the system path.
- `scikit-fmm` (optional dep) requires `g++`, `python3-dev`, and `pkg-config` to build from source. These are pre-installed in the VM snapshot. If rebuilding the environment from scratch, install them via `apt-get install -y g++ python3-dev pkg-config`.
- `pip install` requires `--break-system-packages` on Ubuntu 24.04 (PEP 668) since there is no virtualenv.
- Build isolation for `scikit-fmm` requires explicit `CC=gcc CXX=g++` environment variables when `meson` cannot auto-detect the compiler.
- `conftest.py` at repo root sets `matplotlib.use("Agg")` for headless rendering — no display server needed.
