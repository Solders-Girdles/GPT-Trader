"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.broker_executor
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor

__all__ = ["BrokerExecutor"]

warnings.warn(
    "gpt_trader.orchestration.execution.broker_executor is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.broker_executor instead.",
    DeprecationWarning,
    stacklevel=2,
)
