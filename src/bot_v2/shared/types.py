"""
Shared Data Transfer Objects (DTOs) and common types.

This module consolidates frequently used types across features to eliminate
ad-hoc dict usage and improve type safety.

Design Principles:
- Use dataclasses for immutability and clarity
- Prefer Decimal for financial calculations
- Include validation where appropriate
- Keep DTOs lightweight and focused
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

# Re-export commonly used types from bot_v2.types for convenience
from bot_v2.types.trading import (
    AccountSnapshot,
    PerformanceSummary,
    TradeFill,
    TradingPosition,
    TradingSessionResult,
)

__all__ = [
    # Re-exported from bot_v2.types.trading
    "AccountSnapshot",
    "PerformanceSummary",
    "TradeFill",
    "TradingPosition",
    "TradingSessionResult",
    # New shared types
    "SignalAction",
    "TradingSignal",
    "RiskMetrics",
    "PositionUpdate",
    "OrderRequest",
    "OrderResult",
    "StrategyConfig",
    "BacktestConfig",
    "MetricsSnapshot",
]


# =============================================================================
# Enums
# =============================================================================


class SignalAction(str, Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


# =============================================================================
# Trading Signals
# =============================================================================


@dataclass(frozen=True)
class TradingSignal:
    """
    Standardized trading signal across all strategies.

    Replaces ad-hoc dict[str, Any] patterns with typed structure.
    """

    symbol: str
    action: SignalAction
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    strategy_name: str
    target_position_size: Decimal | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal fields."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")


# =============================================================================
# Risk & Position Management
# =============================================================================


@dataclass(frozen=True)
class RiskMetrics:
    """Risk metrics for a position or portfolio."""

    total_exposure: Decimal
    leverage_used: Decimal
    margin_utilization: float  # 0.0 to 1.0
    liquidation_distance: Decimal | None = None
    var_1day: Decimal | None = None  # Value at Risk (1-day)
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None


@dataclass
class PositionUpdate:
    """Position update event (replaces dict patterns)."""

    symbol: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    unrealized_pnl: Decimal | None = None
    realized_pnl: Decimal | None = None
    margin_used: Decimal | None = None


# =============================================================================
# Order Management
# =============================================================================


@dataclass(frozen=True)
class OrderRequest:
    """
    Standardized order request across execution engines.

    Replaces various dict-based order representations.
    """

    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    order_type: str  # "market", "limit", etc.
    price: Decimal | None = None
    time_in_force: str = "GTC"
    stop_price: Decimal | None = None
    reduce_only: bool = False
    post_only: bool = False
    client_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """
    Order execution result.

    Replaces dict[str, Any] return patterns.
    """

    success: bool
    order_id: str | None = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    commission: Decimal = Decimal("0")
    error_message: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Strategy & Configuration
# =============================================================================


@dataclass
class StrategyConfig:
    """
    Generic strategy configuration.

    Provides type-safe alternative to dict[str, Any] for strategy params.
    """

    strategy_name: str
    enabled: bool = True
    risk_per_trade_pct: float = 0.01
    max_position_size: Decimal | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Backtest configuration (replaces dict patterns)."""

    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    commission_pct: float = 0.001
    slippage_bps: int = 10
    strategy_configs: list[StrategyConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Metrics & Monitoring
# =============================================================================


@dataclass
class MetricsSnapshot:
    """
    Point-in-time metrics snapshot.

    Replaces dict[str, Any] patterns in dashboard and monitoring.
    """

    timestamp: datetime
    total_equity: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    total_return_pct: float
    positions_count: int
    open_orders_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Type Aliases for Common Patterns
# =============================================================================

# Price map: symbol -> price
PriceMap = dict[str, Decimal]

# Position map: symbol -> position
PositionMap = dict[str, TradingPosition]

# Metrics dict (for backward compatibility during migration)
MetricsDict = dict[str, Any]


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_symbol(symbol: str) -> None:
    """Validate trading symbol format."""
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    if not symbol.isupper():
        raise ValueError(f"Symbol must be uppercase: {symbol}")


def validate_quantity(quantity: Decimal) -> None:
    """Validate order quantity."""
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive: {quantity}")


def validate_price(price: Decimal | None, allow_none: bool = False) -> None:
    """Validate price value."""
    if price is None:
        if not allow_none:
            raise ValueError("Price cannot be None")
        return
    if price <= 0:
        raise ValueError(f"Price must be positive: {price}")
