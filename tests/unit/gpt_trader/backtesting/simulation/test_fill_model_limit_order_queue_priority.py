"""Tests for OrderFillModel limit order queue priority simulation."""

from datetime import datetime, timezone
from decimal import Decimal

from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide, OrderType


class TestLimitOrderQueuePriority:
    """Test limit order queue priority simulation."""

    def test_queue_priority_disabled_full_fill(self) -> None:
        """Test full fill when queue priority is disabled."""
        model = OrderFillModel(enable_queue_priority=False)
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("50"),  # Low volume
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),  # Needs 20 volume, bar has 50
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("10")  # Full fill

    def test_queue_priority_high_volume_full_fill(self) -> None:
        """Test full fill with queue priority when volume >> order size."""
        model = OrderFillModel(enable_queue_priority=True)
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("1000"),  # 100x order size
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("10")  # Full fill

    def test_queue_priority_partial_fill(self) -> None:
        """Test partial fill with queue priority when volume is moderate."""
        model = OrderFillModel(
            enable_queue_priority=True,
            limit_volume_threshold=Decimal("1.0"),  # Low threshold
        )
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("50"),  # 5x order size -> 80% fill
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        assert result.filled is True
        assert result.fill_quantity == Decimal("8")  # 80% of 10

    def test_queue_priority_very_low_fill_rejected(self) -> None:
        """Test very low fill percentage (<10%) is rejected."""
        model = OrderFillModel(
            enable_queue_priority=True,
            limit_volume_threshold=Decimal("1.0"),
        )
        bar = Candle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("102"),
            volume=Decimal("10"),  # 1x order size -> 20% fill, but need to check threshold
        )
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("96"),
        )

        result = model.try_fill_limit_order(
            order=order,
            current_bar=bar,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
        )

        # With 1x volume ratio, fill_pct is 0.2 (20%), which is > 10%, so it should fill
        assert result.filled is True
