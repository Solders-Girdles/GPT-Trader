"""Shared helpers for normalizing trading symbols across profiles.

This module re-exports symbol utilities from the shared package to maintain
backward compatibility with existing imports. New code should import directly
from bot_v2.orchestration.shared.symbol_utils.
"""

from bot_v2.orchestration.shared.symbol_utils import (
    PERPS_ALLOWLIST,
    derivatives_enabled,
    normalize_symbols,
)

__all__ = [
    "PERPS_ALLOWLIST",
    "derivatives_enabled",
    "normalize_symbols",
]
