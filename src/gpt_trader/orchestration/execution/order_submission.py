"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.order_submission
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.order_submission import (
    OrderSubmitter,
    _classify_rejection_reason,
    _record_execution_telemetry,
    _record_order_submission_metric,
)

__all__ = [
    "OrderSubmitter",
    "_classify_rejection_reason",
    "_record_execution_telemetry",
    "_record_order_submission_metric",
]

warnings.warn(
    "gpt_trader.orchestration.execution.order_submission is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.order_submission instead.",
    DeprecationWarning,
    stacklevel=2,
)
