"""
Typed contracts for TUI data flow.

This module defines the contract between StatusReporter and TuiState,
providing type safety and eliminating the need for defensive parsing.

The contracts are type aliases to the existing StatusReporter dataclasses,
ensuring a single source of truth for the data structure.
"""

from __future__ import annotations

# Import existing status dataclasses from StatusReporter
from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    OrderStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
    TradeStatus,
)

# Type aliases for clarity - these are the contracts between StatusReporter and TuiState
# In Phase 1, we use the existing structures directly
# In Phase 5, we'll update these to use Decimal instead of str for numeric fields

# Root contract - the complete bot status snapshot
BotStatusSnapshot = BotStatus

# Component contracts - status of each subsystem
EngineStatusContract = EngineStatus
MarketStatusContract = MarketStatus
PositionStatusContract = PositionStatus
OrderStatusContract = OrderStatus
TradeStatusContract = TradeStatus
AccountStatusContract = AccountStatus
StrategyStatusContract = StrategyStatus
RiskStatusContract = RiskStatus
SystemStatusContract = SystemStatus
HeartbeatStatusContract = HeartbeatStatus

# Export all contracts
__all__ = [
    "BotStatusSnapshot",
    "EngineStatusContract",
    "MarketStatusContract",
    "PositionStatusContract",
    "OrderStatusContract",
    "TradeStatusContract",
    "AccountStatusContract",
    "StrategyStatusContract",
    "RiskStatusContract",
    "SystemStatusContract",
    "HeartbeatStatusContract",
]
