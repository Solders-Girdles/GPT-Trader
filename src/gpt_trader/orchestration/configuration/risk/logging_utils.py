"""
DEPRECATED: Logging helper for risk configuration.

This module provides a simple logger instance. For new code, use:

    from gpt_trader.utilities.logging_patterns import get_logger
    logger = get_logger(__name__, component="your_component")
"""

from __future__ import annotations

import warnings

from gpt_trader.utilities.logging import get_logger

logger = get_logger(__name__, component="risk_config_core")

__all__ = ["logger"]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.configuration.risk.logging_utils is deprecated. "  # naming: allow
    "Use gpt_trader.utilities.logging_patterns.get_logger directly.",
    DeprecationWarning,
    stacklevel=2,
)
