"""Shared helpers for backtesting report tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics


def create_mock_broker(
    initial_equity: Decimal = Decimal("100000"),
    final_equity: Decimal = Decimal("110000"),
    total_return_pct: Decimal = Decimal("10"),
    total_return_usd: Decimal = Decimal("10000"),
    total_fees_paid: Decimal = Decimal("500"),
    funding_pnl: Decimal = Decimal("100"),
) -> MagicMock:
    """Create a mock SimulatedBroker."""
    broker = MagicMock()

    broker.get_statistics.return_value = {
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "total_return_usd": total_return_usd,
        "total_fees_paid": total_fees_paid,
        "funding_pnl": funding_pnl,
        "total_trades": 50,
        "winning_trades": 30,
        "losing_trades": 20,
    }

    # Mock positions for unrealized PnL calculation
    broker.positions = {}

    # Mock equity curve for risk metrics
    broker.equity_curve = [
        (datetime(2024, 1, 1), initial_equity),
        (datetime(2024, 1, 2), initial_equity + Decimal("500")),
        (datetime(2024, 1, 3), final_equity),
    ]

    # Mock orders for trade statistics
    broker.orders = []

    return broker


def create_mock_trade_stats() -> TradeStatistics:
    """Create mock TradeStatistics."""
    return TradeStatistics(
        total_trades=50,
        winning_trades=30,
        losing_trades=20,
        breakeven_trades=0,
        win_rate=Decimal("60"),
        loss_rate=Decimal("40"),
        profit_factor=Decimal("3.0"),
        net_profit_factor=Decimal("2.0"),
        fee_drag_per_trade=Decimal("10"),
        total_pnl=Decimal("10000"),
        gross_profit=Decimal("15000"),
        gross_loss=Decimal("-5000"),
        avg_profit_per_trade=Decimal("200"),
        avg_win=Decimal("500"),
        avg_loss=Decimal("-250"),
        largest_win=Decimal("2000"),
        largest_loss=Decimal("-1000"),
        avg_position_size_usd=Decimal("10000"),
        max_position_size_usd=Decimal("25000"),
        avg_leverage=Decimal("5.0"),
        max_leverage=Decimal("10.0"),
        avg_slippage_bps=Decimal("2.5"),
        total_fees_paid=Decimal("500"),
        limit_orders_filled=40,
        limit_orders_cancelled=10,
        limit_fill_rate=Decimal("80"),
        avg_hold_time_minutes=Decimal("120"),
        max_hold_time_minutes=Decimal("480"),
        max_consecutive_wins=5,
        max_consecutive_losses=3,
        current_streak=2,
    )


def create_mock_risk_metrics() -> RiskMetrics:
    """Create mock RiskMetrics."""
    return RiskMetrics(
        max_drawdown_pct=Decimal("15.5"),
        max_drawdown_usd=Decimal("15500"),
        avg_drawdown_pct=Decimal("5.0"),
        drawdown_duration_days=5,
        total_return_pct=Decimal("20"),
        annualized_return_pct=Decimal("73"),
        daily_return_avg=Decimal("0.05"),
        daily_return_std=Decimal("1.5"),
        sharpe_ratio=Decimal("1.5"),
        sortino_ratio=Decimal("2.1"),
        calmar_ratio=Decimal("0.8"),
        volatility_annualized=Decimal("23.7"),
        downside_volatility=Decimal("15.0"),
        max_leverage_used=Decimal("10.0"),
        avg_leverage_used=Decimal("5.0"),
        time_in_market_pct=Decimal("60"),
        var_95_daily=Decimal("2500"),
        var_99_daily=Decimal("4000"),
    )
