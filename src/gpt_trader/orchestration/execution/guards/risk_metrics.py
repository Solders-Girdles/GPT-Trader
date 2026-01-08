"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.risk_metrics
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.risk_metrics import RiskMetricsGuard

__all__ = ["RiskMetricsGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.risk_metrics is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.risk_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)
