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
from typing import Any

import gpt_trader.features.live_trade.execution.validation as _validation
from gpt_trader.features.live_trade.execution.validation import (
    _FALLBACK_FAILURE_TRACKER,
    METRIC_CONSECUTIVE_FAILURES_ESCALATION,
    METRIC_MARK_STALENESS_CHECK_FAILED,
    METRIC_ORDER_PREVIEW_FAILED,
    METRIC_SLIPPAGE_GUARD_CHECK_FAILED,
    OrderValidator,
    ValidationFailureTracker,
    quantize_price_side_aware,
    record_counter,
    spec_validate_order,
)

__all__ = [
    "METRIC_CONSECUTIVE_FAILURES_ESCALATION",
    "METRIC_MARK_STALENESS_CHECK_FAILED",
    "METRIC_ORDER_PREVIEW_FAILED",
    "METRIC_SLIPPAGE_GUARD_CHECK_FAILED",
    "_FALLBACK_FAILURE_TRACKER",
    "OrderValidator",
    "ValidationFailureTracker",
    "configure_failure_tracker",
    "get_failure_tracker",
    "get_validation_metrics",
    "quantize_price_side_aware",
    "record_counter",
    "spec_validate_order",
]


def _proxy_spec_validate_order(*args: object, **kwargs: object) -> object:
    return spec_validate_order(*args, **kwargs)


def _proxy_quantize_price_side_aware(*args: object, **kwargs: object) -> object:
    return quantize_price_side_aware(*args, **kwargs)


def _proxy_record_counter(*args: object, **kwargs: object) -> object:
    return record_counter(*args, **kwargs)


_validation.spec_validate_order = _proxy_spec_validate_order
_validation.quantize_price_side_aware = _proxy_quantize_price_side_aware
_validation.record_counter = _proxy_record_counter


def get_failure_tracker() -> ValidationFailureTracker:
    _validation._FALLBACK_FAILURE_TRACKER = _FALLBACK_FAILURE_TRACKER
    return _validation.get_failure_tracker()


def configure_failure_tracker(
    escalation_threshold: int = 5,
    escalation_callback: object | None = None,
) -> None:
    _validation._FALLBACK_FAILURE_TRACKER = _FALLBACK_FAILURE_TRACKER
    _validation.configure_failure_tracker(
        escalation_threshold=escalation_threshold,
        escalation_callback=escalation_callback,
    )


def get_validation_metrics() -> dict[str, Any]:
    _validation._FALLBACK_FAILURE_TRACKER = _FALLBACK_FAILURE_TRACKER
    return _validation.get_validation_metrics()


# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.execution.validation is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.validation instead.",
    DeprecationWarning,
    stacklevel=2,
)
