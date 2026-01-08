"""
DEPRECATED: logging helpers have moved to gpt_trader.features.live_trade.orchestrator.

This shim exists for backward compatibility. Update imports to the new logging helpers.
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.logging_utils import (  # naming: allow
    json_logger,
    logger,
)

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.logging_utils is deprecated. "  # naming: allow
    "Import from gpt_trader.features.live_trade.orchestrator.logging_utils instead.",  # naming: allow
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["logger", "json_logger"]
