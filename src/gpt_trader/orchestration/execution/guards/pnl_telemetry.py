"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.pnl_telemetry
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.pnl_telemetry import PnLTelemetryGuard

__all__ = ["PnLTelemetryGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.pnl_telemetry is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.pnl_telemetry instead.",
    DeprecationWarning,
    stacklevel=2,
)
