"""Tests for OrderFillModel queue fill percentage estimation."""

from datetime import datetime, timezone
from decimal import Decimal

from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle


class TestQueueFillEstimation:
    """Test queue fill percentage estimation."""

    def test_high_volume_ratio_full_fill(self) -> None:
        """Test 100% fill when volume ratio >= 10x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),
        )
        order = make_order(quantity=Decimal("10"))  # 1000/10 = 100x

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("1.0")

    def test_medium_high_volume_80_fill(self) -> None:
        """Test 80% fill when volume ratio is 5-10x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("70"),  # 70/10 = 7x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.8")

    def test_medium_volume_50_fill(self) -> None:
        """Test 50% fill when volume ratio is 2-5x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("30"),  # 30/10 = 3x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.5")

    def test_low_volume_20_fill(self) -> None:
        """Test 20% fill when volume ratio < 2x."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime.now(timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("15"),  # 15/10 = 1.5x
        )
        order = make_order(quantity=Decimal("10"))

        fill_pct = model._estimate_queue_fill_percentage(
            order=order,
            bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )
        assert fill_pct == Decimal("0.2")
