"""Tests for trade statistics timing calculations."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.metrics.statistics_test_helpers import _create_mock_order

from gpt_trader.backtesting.metrics.statistics import _calculate_timing_metrics


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
        assert float(result["avg_hold"]) == pytest.approx(30.0, rel=0.01)
        assert float(result["max_hold"]) == pytest.approx(30.0, rel=0.01)

    def test_multiple_round_trips(self) -> None:
        now = datetime.now()
        orders = [
            _create_mock_order(side="BUY", symbol="BTC-USD", submitted_at=now),
            _create_mock_order(
                side="SELL", symbol="BTC-USD", submitted_at=now + timedelta(minutes=10)
            ),
            _create_mock_order(
                side="BUY", symbol="BTC-USD", submitted_at=now + timedelta(minutes=20)
            ),
            _create_mock_order(
                side="SELL", symbol="BTC-USD", submitted_at=now + timedelta(minutes=50)
            ),
        ]
        result = _calculate_timing_metrics(orders)
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
        assert float(result["avg_hold"]) == pytest.approx(20.0, rel=0.01)
