"""
Runtime utilities for GPT-Trader application.

This module provides runtime path resolution and storage management.

Usage:
    from gpt_trader.app.runtime import RuntimePaths, resolve_runtime_paths

    paths = resolve_runtime_paths(config=config, profile=profile)
"""

from gpt_trader.app.runtime.fingerprint import (
    StartupConfigFingerprint,
    compare_startup_config_fingerprints,
    compute_startup_config_fingerprint,
    load_startup_config_fingerprint,
    write_startup_config_fingerprint,
)
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
    "StartupConfigFingerprint",
    "compute_startup_config_fingerprint",
    "compare_startup_config_fingerprints",
    "write_startup_config_fingerprint",
    "load_startup_config_fingerprint",
]
