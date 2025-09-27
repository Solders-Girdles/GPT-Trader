"""Centralized filesystem paths for GPT-Trader runtime artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
VAR_DIR = PROJECT_ROOT / "var"
LOG_DIR = VAR_DIR / "logs"
RESULTS_DIR = VAR_DIR / "results"
RUNTIME_DATA_DIR = VAR_DIR / "data"
PERPS_RUNTIME_DIR = RUNTIME_DATA_DIR / "perps_bot"
DEFAULT_EVENT_STORE_DIR = PERPS_RUNTIME_DIR / "shared"


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    targets = list(paths or (VAR_DIR, LOG_DIR, RESULTS_DIR, RUNTIME_DATA_DIR))
    for path in targets:
        path.mkdir(parents=True, exist_ok=True)


# Create standard runtime directories during import so CLI commands just work.
ensure_directories()
