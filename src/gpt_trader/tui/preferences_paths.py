"""Helpers for resolving TUI preferences storage locations.

We keep a legacy preferences file under `config/` for backward compatibility,
but write all runtime preferences (theme, mode, credential cache, etc.) to a
runtime path to avoid dirtying tracked config files.
"""

from __future__ import annotations

import os
from pathlib import Path

from gpt_trader.config import path_registry

ENV_TUI_PREFERENCES_PATH = "GPT_TRADER_TUI_PREFERENCES_PATH"


def default_preferences_path() -> Path:
    """Return the default runtime preferences path (writable, ignored by git)."""
    return path_registry.RUNTIME_DATA_DIR / "tui_preferences.json"


def legacy_preferences_path() -> Path:
    """Return the legacy tracked preferences path (read-only fallback)."""
    return path_registry.PROJECT_ROOT / "config" / "tui_preferences.json"


def resolve_preferences_paths(
    preferences_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Resolve the primary preferences path and an optional legacy fallback.

    Resolution order:
    1) Explicit `preferences_path` argument (no fallback)
    2) `GPT_TRADER_TUI_PREFERENCES_PATH` env var (no fallback)
    3) Default runtime path + legacy config fallback
    """
    if preferences_path is not None:
        return preferences_path, None

    env_path = os.getenv(ENV_TUI_PREFERENCES_PATH)
    if env_path:
        return Path(env_path).expanduser(), None

    return default_preferences_path(), legacy_preferences_path()

