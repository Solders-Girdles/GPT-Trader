"""Status data models for operational monitoring.

Pure data contracts describing a point-in-time snapshot of the trading bot's
state, plus the small serialization helpers they depend on. The behavior that
populates these models and writes them to disk lives in
``gpt_trader.monitoring.status_reporter``; keeping the contracts here lets
consumers import the data shapes without pulling in the
reporter machinery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gpt_trader.utilities.time_provider import get_clock


def _format_timestamp_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _clock_time() -> float:
    return get_clock().time()


@dataclass
class EngineStatus:
    """Status snapshot of the trading engine."""

    running: bool = False
    uptime_seconds: float = 0.0
    cycle_count: int = 0
    last_cycle_time: float | None = None
    errors_count: int = 0
    last_error: str | None = None
    last_error_time: float | None = None


@dataclass
class MarketStatus:
    """Status snapshot of market data."""

    symbols: list[str] = field(default_factory=list)
    last_prices: dict[str, Decimal] = field(default_factory=dict)
    last_price_update: float | None = None
    price_history: dict[str, list[Decimal]] = field(default_factory=dict)


@dataclass
class TickerFreshnessSummary:
    """Status summary of ticker freshness health check."""

    symbol_count: int = 0
    stale_count: int = 0
    stale_symbols: list[str] = field(default_factory=list)
    stale_symbols_capped: bool = False
    severity: str = "unknown"
    reason: str = ""
    status: str = "unknown"


@dataclass
class PositionStatus:
    """Status snapshot of positions."""

    count: int = 0
    symbols: list[str] = field(default_factory=list)
    total_unrealized_pnl: Decimal = Decimal("0")
    equity: Decimal = Decimal("0")
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class HeartbeatStatus:
    """Status snapshot of heartbeat service."""

    enabled: bool = False
    running: bool = False
    heartbeat_count: int = 0
    last_heartbeat: float | None = None
    is_healthy: bool = False


@dataclass
class OrderStatus:
    """Status snapshot of an active order."""

    order_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal | None
    status: str
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    creation_time: float = 0.0
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None


@dataclass
class TradeStatus:
    """Status snapshot of a recent trade."""

    trade_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    time: str
    order_id: str
    fee: Decimal = Decimal("0")


@dataclass
class BalanceEntry:
    """Single balance entry with Decimal amounts."""

    asset: str
    total: Decimal
    available: Decimal
    hold: Decimal = Decimal("0")


@dataclass
class DecisionEntry:
    """Single strategy decision entry with typed fields."""

    symbol: str
    action: str = "HOLD"
    reason: str = ""
    confidence: float = 0.0
    indicators: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    decision_id: str = ""  # Unique ID for linking to orders/trades
    blocked_by: str = ""  # Guard/reason that blocked execution (empty if executed)
    # Indicator contributions for transparency (list of dicts with name, value, contribution)
    contributions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AccountStatus:
    """Status snapshot of account metrics."""

    volume_30d: Decimal = Decimal("0")
    fees_30d: Decimal = Decimal("0")
    fee_tier: str = ""
    balances: list[BalanceEntry] = field(default_factory=list)


@dataclass
class StrategyStatus:
    """Status snapshot of strategy engine."""

    active_strategies: list[str] = field(default_factory=list)
    last_decisions: list[DecisionEntry] = field(default_factory=list)
    # Live performance metrics (from real trades)
    performance: dict[str, Any] | None = None
    # Historical backtest performance (from backtesting)
    backtest_performance: dict[str, Any] | None = None
    # Strategy indicator parameters (RSI period, MA periods, etc.)
    parameters: dict[str, Any] | None = None


@dataclass
class GuardStatus:
    """Status snapshot of a single risk guard."""

    name: str
    severity: str = "MEDIUM"
    last_triggered: float = 0.0
    triggered_count: int = 0
    description: str = ""


@dataclass
class RiskStatus:
    """Status snapshot of risk management metrics."""

    max_leverage: float = 0.0
    daily_loss_limit_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    reduce_only_mode: bool = False
    reduce_only_reason: str = ""
    guards: list[GuardStatus] = field(default_factory=list)


@dataclass
class WebSocketStatus:
    """Status snapshot of WebSocket connection health."""

    connected: bool = False
    last_message_ts: float | None = None
    last_heartbeat_ts: float | None = None
    last_close_ts: float | None = None
    last_error_ts: float | None = None
    gap_count: int = 0
    reconnect_count: int = 0
    message_stale: bool = False
    heartbeat_stale: bool = False


@dataclass
class SystemStatus:
    """Status snapshot of system health and brokerage connection."""

    api_latency: float = 0.0
    connection_status: str = "UNKNOWN"
    rate_limit_usage: str = "0%"
    memory_usage: str = "0MB"
    cpu_usage: str = "0%"


@dataclass
class BotStatus:
    """Complete status snapshot of the trading bot."""

    bot_id: str = ""
    timestamp: float = field(default_factory=_clock_time)
    timestamp_iso: str = ""
    version: str = "1.0.0"

    engine: EngineStatus = field(default_factory=EngineStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    positions: PositionStatus = field(default_factory=PositionStatus)
    orders: list[OrderStatus] = field(default_factory=list)
    trades: list[TradeStatus] = field(default_factory=list)
    account: AccountStatus = field(default_factory=AccountStatus)
    strategy: StrategyStatus = field(default_factory=StrategyStatus)
    risk: RiskStatus = field(default_factory=RiskStatus)
    system: SystemStatus = field(default_factory=SystemStatus)
    heartbeat: HeartbeatStatus = field(default_factory=HeartbeatStatus)
    websocket: WebSocketStatus = field(default_factory=WebSocketStatus)
    ticker_freshness: TickerFreshnessSummary = field(default_factory=TickerFreshnessSummary)

    # Overall health
    healthy: bool = True
    health_issues: list[str] = field(default_factory=list)
    # Signal-based health state (OK/WARN/CRIT/UNKNOWN)
    health_state: str = "UNKNOWN"
    # Execution health signals summary
    execution_signals: dict[str, Any] | None = None

    # Reporter interval for observer connection health tracking
    observer_interval: float = 2.0

    def __post_init__(self) -> None:
        if not self.timestamp_iso:
            self.timestamp_iso = _format_timestamp_iso(self.timestamp)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)
