"""Compatibility shim re-exporting modular BotConfig components."""

from __future__ import annotations

from bot_v2.config.types import Profile

from .bot_config import (
    BotConfig,
    ConfigState,
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
)

__all__ = [
    "BotConfig",
    "ConfigState",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
    "Profile",
]
