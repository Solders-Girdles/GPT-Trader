"""
Runtime utilities for GPT-Trader application.

This module provides runtime path resolution and storage management.

Usage:
    from gpt_trader.app.runtime import RuntimePaths, resolve_runtime_paths

    paths = resolve_runtime_paths(config=config, profile=profile)
"""

from gpt_trader.app.runtime.paths import RuntimePaths, resolve_runtime_paths

__all__ = ["RuntimePaths", "resolve_runtime_paths"]
