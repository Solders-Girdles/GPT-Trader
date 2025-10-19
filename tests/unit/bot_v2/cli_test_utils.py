"""Shared helpers for CLI subprocess-based tests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def cli_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return env with ``PYTHONPATH`` including project ``src/`` for CLI runs."""
    env = dict((base or os.environ).items())
    project_root = Path(__file__).resolve().parents[3]
    src_path = project_root / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_path) + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONASYNCIODEBUG", "0")
    return env


__all__ = ["cli_env"]
