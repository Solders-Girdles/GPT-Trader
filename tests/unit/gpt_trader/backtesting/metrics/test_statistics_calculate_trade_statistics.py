"""Tests for calculate_trade_statistics."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.statistics_test_helpers import _create_mock_order

from gpt_trader.backtesting.metrics.statistics import calculate_trade_statistics
from gpt_trader.backtesting.types import CompletedTrade, TradeOutcome, TradeSide


class TestCalculateTradeStatistics:
    """Tests for calculate_trade_statistics function."""

    def _create_mock_broker(
        self,
        filled_orders: list[Any] | None = None,
        cancelled_orders: list[Any] | None = None,
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
        assert result.net_profit_factor == Decimal("0")

    def test_profit_factor_capped_when_no_losses(self) -> None:
        broker = self._create_mock_broker(
            total_trades=10,
            winning_trades=10,
            losing_trades=0,
        )
        result = calculate_trade_statistics(broker)
        assert result.profit_factor == Decimal("999.99")

    def test_net_profit_factor_accounts_for_fees(self) -> None:
        now = datetime.now()
        trades = [
            CompletedTrade(
                trade_id="t1",
                symbol="BTC-USD",
                side=TradeSide.LONG,
                entry_time=now,
                entry_price=Decimal("100"),
                exit_time=now + timedelta(minutes=5),
                exit_price=Decimal("110"),
                quantity=Decimal("1"),
                realized_pnl=Decimal("100"),
                fees_paid=Decimal("20"),
                net_pnl=Decimal("80"),
                outcome=TradeOutcome.WIN,
                hold_time_seconds=300,
            ),
            CompletedTrade(
                trade_id="t2",
                symbol="BTC-USD",
                side=TradeSide.LONG,
                entry_time=now + timedelta(minutes=10),
                entry_price=Decimal("100"),
                exit_time=now + timedelta(minutes=20),
                exit_price=Decimal("95"),
                quantity=Decimal("1"),
                realized_pnl=Decimal("-50"),
                fees_paid=Decimal("10"),
                net_pnl=Decimal("-60"),
                outcome=TradeOutcome.LOSS,
                hold_time_seconds=600,
            ),
        ]
        broker = self._create_mock_broker()
        broker.get_completed_trades.return_value = trades
        broker._total_fees_paid = Decimal("30")

        result = calculate_trade_statistics(broker)
        assert result.total_trades == 2
        assert result.profit_factor == Decimal("2")
        assert float(result.net_profit_factor) == pytest.approx(1.25, rel=1e-6)
        assert float(result.fee_drag_per_trade) == pytest.approx(15.0, rel=1e-6)

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
        assert float(result.limit_fill_rate) == pytest.approx(66.67, rel=0.1)

    def test_limit_fill_rate_100_when_no_limit_orders(self) -> None:
        filled = [
            _create_mock_order(order_type="MARKET"),
        ]
        broker = self._create_mock_broker(filled_orders=filled)
        result = calculate_trade_statistics(broker)
        assert result.limit_fill_rate == Decimal("100")

    def test_calculates_avg_slippage(self) -> None:
        broker = self._create_mock_broker(total_trades=5)
        broker._total_slippage_bps = Decimal("25")  # 25 bps total over 5 trades
        result = calculate_trade_statistics(broker)
        assert float(result.avg_slippage_bps) == pytest.approx(5.0, rel=0.01)

    def test_includes_total_fees(self) -> None:
        broker = self._create_mock_broker()
        broker._total_fees_paid = Decimal("123.45")
        result = calculate_trade_statistics(broker)
        assert result.total_fees_paid == Decimal("123.45")
