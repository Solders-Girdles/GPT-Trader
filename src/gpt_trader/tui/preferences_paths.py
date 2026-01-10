"""Helpers for resolving TUI preferences storage locations."""

from __future__ import annotations

import os
from pathlib import Path

from gpt_trader.config import path_registry

ENV_TUI_PREFERENCES_PATH = "GPT_TRADER_TUI_PREFERENCES_PATH"


def default_preferences_path() -> Path:
    """Return the default runtime preferences path (writable, ignored by git)."""
    return path_registry.RUNTIME_DATA_DIR / "tui_preferences.json"


def resolve_preferences_path(preferences_path: Path | None = None) -> Path:
    """Resolve the preferences path.

    Resolution order:
    1) Explicit `preferences_path` argument
    2) `GPT_TRADER_TUI_PREFERENCES_PATH` env var
    3) Default runtime path
    """
    if preferences_path is not None:
        return preferences_path

    env_path = os.getenv(ENV_TUI_PREFERENCES_PATH)
    if env_path:
        return Path(env_path).expanduser()

    return default_preferences_path()
