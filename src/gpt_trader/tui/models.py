from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class MarketData:
    """Data structure for market information."""

    prices: dict[str, str] = field(default_factory=dict)
    last_update: float = 0.0
    price_history: dict[str, list[Decimal]] = field(default_factory=dict)


@dataclass
class PositionData:
    """Data structure for position information."""

    positions: dict[str, Any] = field(default_factory=dict)
    total_unrealized_pnl: str = "0.00"
    equity: str = "0.00"


@dataclass
class OrderData:
    """Data structure for order information."""

    orders: list[dict[str, str]] = field(default_factory=list)
    # We can also have a more structured list if needed, but dict is flexible for now
    # Let's define what keys we expect: order_id, symbol, side, quantity, price, status, type, time_in_force


@dataclass
class TradeData:
    """Data structure for trade information."""

    trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AccountData:
    """Data structure for account metrics."""

    volume_30d: str = "0.00"
    fees_30d: str = "0.00"
    fee_tier: str = ""
    balances: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DecisionData:
    """Data structure for a single strategy decision."""

    symbol: str
    action: str
    reason: str
    confidence: float
    indicators: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class StrategyData:
    """Data structure for strategy information."""

    active_strategies: list[str] = field(default_factory=list)
    last_decisions: dict[str, DecisionData] = field(default_factory=dict)


@dataclass
class RiskData:
    """Data structure for risk management information."""

    max_leverage: float = 0.0
    daily_loss_limit_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    reduce_only_mode: bool = False
    reduce_only_reason: str = ""
    active_guards: list[str] = field(default_factory=list)


@dataclass
class SystemData:
    """Data structure for system health and brokerage connection."""

    api_latency: float = 0.0
    connection_status: str = "UNKNOWN"
    rate_limit_usage: str = "0%"
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"
