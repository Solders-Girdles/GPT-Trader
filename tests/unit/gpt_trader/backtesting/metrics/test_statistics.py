"""Tests for trade statistics calculation module."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.backtesting.metrics.statistics import (
    TradeStatistics,
    _calculate_pnl_metrics,
    _calculate_position_metrics,
    _calculate_streak_metrics,
    _calculate_timing_metrics,
    calculate_trade_statistics,
)
from gpt_trader.core import Order, OrderSide, OrderStatus, OrderType


def _create_order(
    symbol: str = "BTC-USD",
    side: str = "BUY",
    order_type: str = "MARKET",
    quantity: Decimal = Decimal("1"),
    avg_fill_price: Decimal | None = Decimal("50000"),
    filled_quantity: Decimal = Decimal("1"),
    submitted_at: datetime | None = None,
) -> Order:
    """Create a real Order instance for testing.

    Uses proper core types instead of MagicMock for better type safety.
    """
    return Order(
        id=f"test-order-{symbol}-{side}",
        symbol=symbol,
        side=OrderSide(side),
        type=OrderType(order_type),
        quantity=quantity,
        status=OrderStatus.FILLED,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
        submitted_at=submitted_at or datetime.now(),
    )


# Keep alias for backward compatibility during migration
_create_mock_order = _create_order


class TestCalculatePnlMetrics:
    """Tests for _calculate_pnl_metrics function."""

    def test_empty_orders_returns_zeros(self) -> None:
        result = _calculate_pnl_metrics([])
        assert result["total_pnl"] == Decimal("0")
        assert result["gross_profit"] == Decimal("0")
        assert result["gross_loss"] == Decimal("0")
        assert result["largest_win"] == Decimal("0")
        assert result["largest_loss"] == Decimal("0")

    def test_orders_without_fill_price_skipped(self) -> None:
        order = _create_mock_order(avg_fill_price=None)
        result = _calculate_pnl_metrics([order])
        # Should still return zeros (no valid orders)
        assert result["total_pnl"] == Decimal("0")

    def test_returns_structure_correct(self) -> None:
        orders = [_create_mock_order()]
        result = _calculate_pnl_metrics(orders)
        assert "total_pnl" in result
        assert "gross_profit" in result
        assert "gross_loss" in result
        assert "largest_win" in result
        assert "largest_loss" in result


class TestCalculatePositionMetrics:
    """Tests for _calculate_position_metrics function."""

    def test_empty_orders_returns_defaults(self) -> None:
        broker = MagicMock()
        result = _calculate_position_metrics([], broker)
        assert result["avg_size"] == Decimal("0")
        assert result["max_size"] == Decimal("0")
        assert result["avg_leverage"] == Decimal("1")
        assert result["max_leverage"] == Decimal("1")

    def test_calculates_average_position_size(self) -> None:
        orders = [
            _create_mock_order(
                filled_quantity=Decimal("1"),
                avg_fill_price=Decimal("50000"),
            ),
            _create_mock_order(
                filled_quantity=Decimal("2"),
                avg_fill_price=Decimal("50000"),
            ),
        ]
        broker = MagicMock()
        broker.positions = {}
        result = _calculate_position_metrics(orders, broker)
        # First order: 1 * 50000 = 50000
        # Second order: 2 * 50000 = 100000
        # Average: 75000
        assert float(result["avg_size"]) == pytest.approx(75000.0, rel=0.01)
        assert float(result["max_size"]) == pytest.approx(100000.0, rel=0.01)

    def test_extracts_leverage_from_position(self) -> None:
        order = _create_mock_order(symbol="BTC-PERP")
        broker = MagicMock()
        position = MagicMock()
        position.leverage = 5
        broker.positions = {"BTC-PERP": position}
        result = _calculate_position_metrics([order], broker)
        assert float(result["max_leverage"]) == 5.0
        assert float(result["avg_leverage"]) == 5.0

    def test_multiple_orders_different_leverage(self) -> None:
        orders = [
            _create_mock_order(symbol="BTC-PERP"),
            _create_mock_order(symbol="ETH-PERP"),
        ]
        broker = MagicMock()
        btc_pos = MagicMock()
        btc_pos.leverage = 3
        eth_pos = MagicMock()
        eth_pos.leverage = 5
        broker.positions = {"BTC-PERP": btc_pos, "ETH-PERP": eth_pos}
        result = _calculate_position_metrics(orders, broker)
        assert float(result["max_leverage"]) == 5.0
        assert float(result["avg_leverage"]) == 4.0  # (3+5)/2

    def test_orders_without_fill_price_skipped(self) -> None:
        orders = [
            _create_mock_order(avg_fill_price=None),
            _create_mock_order(
                filled_quantity=Decimal("1"),
                avg_fill_price=Decimal("10000"),
            ),
        ]
        broker = MagicMock()
        broker.positions = {}
        result = _calculate_position_metrics(orders, broker)
        # Only second order counted
        assert float(result["avg_size"]) == pytest.approx(10000.0, rel=0.01)


class TestCalculateTimingMetrics:
    """Tests for _calculate_timing_metrics function."""

    def test_empty_orders_returns_zeros(self) -> None:
        result = _calculate_timing_metrics([])
        assert result["avg_hold"] == Decimal("0")
        assert result["max_hold"] == Decimal("0")

    def test_orders_without_submitted_at_skipped(self) -> None:
        order = _create_mock_order(submitted_at=None)
        result = _calculate_timing_metrics([order])
        assert result["avg_hold"] == Decimal("0")

    def test_order_with_none_submitted_at_explicitly_skipped(self) -> None:
        """Test that orders with submitted_at=None are skipped in timing calculation."""
        now = datetime.now()
        order_with_time = _create_mock_order(side="BUY", submitted_at=now)
        order_none = _create_mock_order(side="SELL")
        order_none.submitted_at = None  # Explicitly set to None after creation
        result = _calculate_timing_metrics([order_with_time, order_none])
        # With None submitted_at on sell order, no complete round trip
        assert result["avg_hold"] == Decimal("0")

    def test_calculates_hold_time_for_round_trip(self) -> None:
        now = datetime.now()
        buy_order = _create_mock_order(
            side="BUY",
            submitted_at=now,
        )
        sell_order = _create_mock_order(
            side="SELL",
            submitted_at=now + timedelta(minutes=30),
        )
        result = _calculate_timing_metrics([buy_order, sell_order])
        # Should calculate 30 minute hold time
        assert float(result["avg_hold"]) == pytest.approx(30.0, rel=0.01)
        assert float(result["max_hold"]) == pytest.approx(30.0, rel=0.01)

    def test_multiple_round_trips(self) -> None:
        now = datetime.now()
        orders = [
            # First trade: 10 minutes
            _create_mock_order(side="BUY", symbol="BTC-USD", submitted_at=now),
            _create_mock_order(
                side="SELL", symbol="BTC-USD", submitted_at=now + timedelta(minutes=10)
            ),
            # Second trade: 30 minutes
            _create_mock_order(
                side="BUY", symbol="BTC-USD", submitted_at=now + timedelta(minutes=20)
            ),
            _create_mock_order(
                side="SELL", symbol="BTC-USD", submitted_at=now + timedelta(minutes=50)
            ),
        ]
        result = _calculate_timing_metrics(orders)
        # Average: (10 + 30) / 2 = 20 minutes
        assert float(result["avg_hold"]) == pytest.approx(20.0, rel=0.01)
        assert float(result["max_hold"]) == pytest.approx(30.0, rel=0.01)

    def test_different_symbols_tracked_separately(self) -> None:
        now = datetime.now()
        orders = [
            _create_mock_order(side="BUY", symbol="BTC-USD", submitted_at=now),
            _create_mock_order(
                side="BUY", symbol="ETH-USD", submitted_at=now + timedelta(minutes=5)
            ),
            _create_mock_order(
                side="SELL", symbol="BTC-USD", submitted_at=now + timedelta(minutes=20)
            ),
            _create_mock_order(
                side="SELL", symbol="ETH-USD", submitted_at=now + timedelta(minutes=25)
            ),
        ]
        result = _calculate_timing_metrics(orders)
        # BTC: 20 minutes, ETH: 20 minutes
        assert float(result["avg_hold"]) == pytest.approx(20.0, rel=0.01)


class TestCalculateStreakMetrics:
    """Tests for _calculate_streak_metrics function."""

    def test_returns_zeros_placeholder(self) -> None:
        # Current implementation returns zeros
        result = _calculate_streak_metrics([])
        assert result["max_wins"] == 0
        assert result["max_losses"] == 0
        assert result["current"] == 0

    def test_with_orders_returns_zeros(self) -> None:
        orders = [_create_mock_order() for _ in range(5)]
        result = _calculate_streak_metrics(orders)
        # Placeholder implementation
        assert result["max_wins"] == 0
        assert result["max_losses"] == 0
        assert result["current"] == 0


class TestCalculateTradeStatistics:
    """Tests for calculate_trade_statistics function."""

    def _create_mock_broker(
        self,
        filled_orders: list[MagicMock] | None = None,
        cancelled_orders: list[MagicMock] | None = None,
        total_trades: int = 10,
        winning_trades: int = 6,
        losing_trades: int = 3,
    ) -> MagicMock:
        broker = MagicMock()
        broker._filled_orders = {i: order for i, order in enumerate(filled_orders or [])}
        broker._cancelled_orders = {i: order for i, order in enumerate(cancelled_orders or [])}
        broker._total_trades = total_trades
        broker._winning_trades = winning_trades
        broker._losing_trades = losing_trades
        broker._total_slippage_bps = Decimal("10")
        broker._total_fees_paid = Decimal("50")
        broker.positions = {}
        # Return empty list for completed trades to use legacy fallback path
        broker.get_completed_trades.return_value = []
        return broker

    def test_calculates_win_rate(self) -> None:
        broker = self._create_mock_broker(
            total_trades=100,
            winning_trades=60,
            losing_trades=30,
        )
        result = calculate_trade_statistics(broker)
        assert float(result.win_rate) == pytest.approx(60.0, rel=0.01)
        assert float(result.loss_rate) == pytest.approx(30.0, rel=0.01)

    def test_calculates_breakeven_trades(self) -> None:
        broker = self._create_mock_broker(
            total_trades=100,
            winning_trades=60,
            losing_trades=30,
        )
        result = calculate_trade_statistics(broker)
        # 100 - 60 - 30 = 10 breakeven
        assert result.breakeven_trades == 10

    def test_zero_trades_returns_zeros(self) -> None:
        broker = self._create_mock_broker(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        result = calculate_trade_statistics(broker)
        assert result.win_rate == Decimal("0")
        assert result.loss_rate == Decimal("0")
        assert result.avg_profit_per_trade == Decimal("0")

    def test_profit_factor_capped_when_no_losses(self) -> None:
        # When gross_loss is 0, profit_factor should be capped
        broker = self._create_mock_broker(
            total_trades=10,
            winning_trades=10,
            losing_trades=0,
        )
        result = calculate_trade_statistics(broker)
        assert result.profit_factor == Decimal("999.99")

    def test_counts_limit_orders(self) -> None:
        filled = [
            _create_mock_order(order_type="LIMIT"),
            _create_mock_order(order_type="LIMIT"),
            _create_mock_order(order_type="MARKET"),
        ]
        cancelled = [
            _create_mock_order(order_type="LIMIT"),
        ]
        broker = self._create_mock_broker(
            filled_orders=filled,
            cancelled_orders=cancelled,
        )
        result = calculate_trade_statistics(broker)
        assert result.limit_orders_filled == 2
        assert result.limit_orders_cancelled == 1
        # Fill rate: 2 / (2+1) = 66.67%
        assert float(result.limit_fill_rate) == pytest.approx(66.67, rel=0.1)

    def test_limit_fill_rate_100_when_no_limit_orders(self) -> None:
        filled = [
            _create_mock_order(order_type="MARKET"),
        ]
        broker = self._create_mock_broker(filled_orders=filled)
        result = calculate_trade_statistics(broker)
        # Default to 100% when no limit orders
        assert result.limit_fill_rate == Decimal("100")

    def test_calculates_avg_slippage(self) -> None:
        broker = self._create_mock_broker(total_trades=5)
        broker._total_slippage_bps = Decimal("25")  # 25 bps total over 5 trades
        result = calculate_trade_statistics(broker)
        # 25 / 5 = 5 bps average
        assert float(result.avg_slippage_bps) == pytest.approx(5.0, rel=0.01)

    def test_includes_total_fees(self) -> None:
        broker = self._create_mock_broker()
        broker._total_fees_paid = Decimal("123.45")
        result = calculate_trade_statistics(broker)
        assert result.total_fees_paid == Decimal("123.45")


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
