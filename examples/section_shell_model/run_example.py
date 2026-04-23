"""
Thin wrapper: implementation lives in :mod:`blade_precompute.section_shell_model.run_example`.

Run from repo root::

    python examples/section_shell_model/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from blade_precompute.section_shell_model.run_example import main as _impl_main

    return _impl_main()


if __name__ == "__main__":
    main()
