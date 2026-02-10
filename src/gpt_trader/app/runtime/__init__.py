"""
Runtime utilities for GPT-Trader application.

This module provides runtime path resolution and storage management.

Usage:
    from gpt_trader.app.runtime import RuntimePaths, resolve_runtime_paths

    paths = resolve_runtime_paths(config=config, profile=profile)
"""

from gpt_trader.app.runtime.paths import RuntimePaths, resolve_runtime_paths
from gpt_trader.app.runtime.settings import (
    RuntimeSettingsSnapshot,
    create_runtime_settings_snapshot,
    ensure_runtime_settings_snapshot,
)

__all__ = [
    "RuntimePaths",
    "resolve_runtime_paths",
    "RuntimeSettingsSnapshot",
    "create_runtime_settings_snapshot",
    "ensure_runtime_settings_snapshot",
]
