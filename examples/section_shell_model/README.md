# Section shell model (example entry)

Implementation and tests live under **`blade_precompute/section_shell_model`**. PNG outputs stay in this directory: `outputs/`.

When you run [`main_precompute.py`](../../main_precompute.py), the same style of PNGs is written under each job folder: `outputs/<timestamp>/section_shell_model/` (single mesh density per station; no mesh-refinement sweeps).

## Run

From the repository root:

```bash
python examples/section_shell_model/run_example.py
```

or:

```bash
python blade_precompute/section_shell_model/run_example.py
```

See [blade_precompute/section_shell_model/docs/SHELL_MODEL.md](../../blade_precompute/section_shell_model/docs/SHELL_MODEL.md) for design notes.
