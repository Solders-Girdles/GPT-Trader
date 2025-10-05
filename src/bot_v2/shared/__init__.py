"""Shared utilities, types, and cross-cutting concerns."""

from bot_v2.shared.types import (
    AccountSnapshot,
    BacktestConfig,
    MetricsDict,
    MetricsSnapshot,
    OrderRequest,
    OrderResult,
    PerformanceSummary,
    PositionMap,
    PositionUpdate,
    PriceMap,
    RiskMetrics,
    SignalAction,
    StrategyConfig,
    TradeFill,
    TradingPosition,
    TradingSessionResult,
    TradingSignal,
    validate_price,
    validate_quantity,
    validate_symbol,
)

__all__ = [
    # Types
    "AccountSnapshot",
    "BacktestConfig",
    "MetricsDict",
    "MetricsSnapshot",
    "OrderRequest",
    "OrderResult",
    "PerformanceSummary",
    "PositionMap",
    "PositionUpdate",
    "PriceMap",
    "RiskMetrics",
    "SignalAction",
    "StrategyConfig",
    "TradeFill",
    "TradingPosition",
    "TradingSessionResult",
    "TradingSignal",
    # Validators
    "validate_price",
    "validate_quantity",
    "validate_symbol",
]
