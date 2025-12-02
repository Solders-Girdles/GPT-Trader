from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class MarketState:
    """Data structure for market information."""

    prices: dict[str, str] = field(default_factory=dict)
    last_update: float = 0.0
    price_history: dict[str, list[Decimal]] = field(default_factory=dict)


@dataclass
class Position:
    """Data structure for a single position."""

    symbol: str
    quantity: str
    entry_price: str = "N/A"
    unrealized_pnl: str = "0.00"
    mark_price: str = "0.00"
    side: str = ""


@dataclass
class PortfolioSummary:
    """Data structure for position information."""

    positions: dict[str, Position] = field(default_factory=dict)
    total_unrealized_pnl: str = "0.00"
    equity: str = "0.00"


@dataclass
class Order:
    """Data structure for a single order."""

    order_id: str
    symbol: str
    side: str
    quantity: str
    price: str
    status: str
    type: str = "UNKNOWN"
    time_in_force: str = "UNKNOWN"
    creation_time: str = ""


@dataclass
class ActiveOrders:
    """Data structure for order information."""

    orders: list[Order] = field(default_factory=list)


@dataclass
class Trade:
    """Data structure for a single trade."""

    trade_id: str
    symbol: str
    side: str
    quantity: str
    price: str
    order_id: str
    time: str
    fee: str = "0.00"


@dataclass
class TradeHistory:
    """Data structure for trade information."""

    trades: list[Trade] = field(default_factory=list)


@dataclass
class AccountBalance:
    """Data structure for a single asset balance."""

    asset: str
    total: str
    available: str
    hold: str = "0.00"


@dataclass
class AccountSummary:
    """Data structure for account metrics."""

    volume_30d: str = "0.00"
    fees_30d: str = "0.00"
    fee_tier: str = ""
    balances: list[AccountBalance] = field(default_factory=list)


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
class StrategyState:
    """Data structure for strategy information."""

    active_strategies: list[str] = field(default_factory=list)
    last_decisions: dict[str, DecisionData] = field(default_factory=dict)


@dataclass
class RiskState:
    """Data structure for risk management information."""

    max_leverage: float = 0.0
    daily_loss_limit_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    reduce_only_mode: bool = False
    reduce_only_reason: str = ""
    active_guards: list[str] = field(default_factory=list)
    position_leverage: dict[str, float] = field(default_factory=dict)  # Per-position leverage


@dataclass
class SystemStatus:
    """Data structure for system health and brokerage connection."""

    api_latency: float = 0.0
    connection_status: str = "UNKNOWN"
    rate_limit_usage: str = "0%"
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"
