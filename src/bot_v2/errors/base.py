"""Legacy error hierarchy for GPT-Trader.

The classes defined here originated in ``bot_v2/errors.py`` and are retained
for backwards compatibility. New code should prefer the richer hierarchy
available via ``bot_v2.errors`` (package import).
"""

from __future__ import annotations

__all__ = [
    "BotError",
    "ExecutionError",
    "ValidationError",
    "ConfigError",
    "RiskError",
]


class BotError(Exception):
    """Base error for bot operations."""


class ExecutionError(BotError):
    """Error during order execution."""


class ValidationError(BotError):
    """Validation error."""


class ConfigError(BotError):
    """Configuration error."""


class RiskError(BotError):
    """Risk management error."""
