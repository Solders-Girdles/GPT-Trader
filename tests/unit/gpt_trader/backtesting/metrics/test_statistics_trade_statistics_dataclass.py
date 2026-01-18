"""Tests for TradeStatistics dataclass."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.backtesting.metrics.statistics import TradeStatistics


class TestTradeStatisticsDataclass:
    """Tests for TradeStatistics dataclass."""

    def test_all_fields_accessible(self) -> None:
        stats = TradeStatistics(
            total_trades=100,
            winning_trades=60,
            losing_trades=30,
            breakeven_trades=10,
            win_rate=Decimal("60"),
            loss_rate=Decimal("30"),
            profit_factor=Decimal("2.0"),
            net_profit_factor=Decimal("1.5"),
            fee_drag_per_trade=Decimal("1.5"),
            total_pnl=Decimal("5000"),
            gross_profit=Decimal("8000"),
            gross_loss=Decimal("-3000"),
            avg_profit_per_trade=Decimal("50"),
            avg_win=Decimal("133.33"),
            avg_loss=Decimal("-100"),
            largest_win=Decimal("500"),
            largest_loss=Decimal("-200"),
            avg_position_size_usd=Decimal("10000"),
            max_position_size_usd=Decimal("25000"),
            avg_leverage=Decimal("2.5"),
            max_leverage=Decimal("5"),
            avg_slippage_bps=Decimal("3"),
            total_fees_paid=Decimal("150"),
            limit_orders_filled=50,
            limit_orders_cancelled=10,
            limit_fill_rate=Decimal("83.33"),
            avg_hold_time_minutes=Decimal("45"),
            max_hold_time_minutes=Decimal("240"),
            max_consecutive_wins=8,
            max_consecutive_losses=3,
            current_streak=2,
        )
        assert stats.total_trades == 100
        assert stats.win_rate == Decimal("60")
        assert stats.profit_factor == Decimal("2.0")
        assert stats.max_consecutive_wins == 8

    def test_streak_can_be_negative(self) -> None:
        stats = TradeStatistics(
            total_trades=10,
            winning_trades=3,
            losing_trades=7,
            breakeven_trades=0,
            win_rate=Decimal("30"),
            loss_rate=Decimal("70"),
            profit_factor=Decimal("0.5"),
            net_profit_factor=Decimal("0.3"),
            fee_drag_per_trade=Decimal("5"),
            total_pnl=Decimal("-1000"),
            gross_profit=Decimal("500"),
            gross_loss=Decimal("-1500"),
            avg_profit_per_trade=Decimal("-100"),
            avg_win=Decimal("166.67"),
            avg_loss=Decimal("-214.29"),
            largest_win=Decimal("200"),
            largest_loss=Decimal("-500"),
            avg_position_size_usd=Decimal("5000"),
            max_position_size_usd=Decimal("10000"),
            avg_leverage=Decimal("1"),
            max_leverage=Decimal("1"),
            avg_slippage_bps=Decimal("2"),
            total_fees_paid=Decimal("50"),
            limit_orders_filled=5,
            limit_orders_cancelled=2,
            limit_fill_rate=Decimal("71.43"),
            avg_hold_time_minutes=Decimal("30"),
            max_hold_time_minutes=Decimal("120"),
            max_consecutive_wins=2,
            max_consecutive_losses=5,
            current_streak=-3,  # Currently on a 3-loss streak
        )
        assert stats.current_streak == -3
        assert stats.max_consecutive_losses == 5
