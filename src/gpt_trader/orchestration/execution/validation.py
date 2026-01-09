"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.validation

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.execution.validation import ValidationFailureTracker

    # New (preferred)
    from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.features.live_trade.execution.validation import (
    METRIC_CONSECUTIVE_FAILURES_ESCALATION,
    METRIC_MARK_STALENESS_CHECK_FAILED,
    METRIC_ORDER_PREVIEW_FAILED,
    METRIC_SLIPPAGE_GUARD_CHECK_FAILED,
    OrderValidator,
    ValidationFailureTracker,
    configure_failure_tracker,
    get_failure_tracker,
    get_validation_metrics,
    quantize_price_side_aware,
    record_counter,
    spec_validate_order,
)

__all__ = [
    "METRIC_CONSECUTIVE_FAILURES_ESCALATION",
    "METRIC_MARK_STALENESS_CHECK_FAILED",
    "METRIC_ORDER_PREVIEW_FAILED",
    "METRIC_SLIPPAGE_GUARD_CHECK_FAILED",
    "OrderValidator",
    "ValidationFailureTracker",
    "configure_failure_tracker",
    "get_failure_tracker",
    "get_validation_metrics",
    "quantize_price_side_aware",
    "record_counter",
    "spec_validate_order",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.execution.validation is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.validation instead.",
    DeprecationWarning,
    stacklevel=2,
)
