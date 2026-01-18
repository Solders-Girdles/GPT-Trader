"""Tests for OrderFillModel stop order fill simulation."""

from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide, OrderType


class TestStopOrderFill:
    """Test stop order fill simulation."""

    def test_buy_stop_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test buy stop triggers when price rises to stop price."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("1"),
            stop_price=Decimal("50400"),  # Within bar's high of 50500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is True
        assert result.is_maker is False  # Stop fills as market order

    def test_buy_stop_not_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test buy stop not triggered when price doesn't reach stop."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=Decimal("51000"),  # Above bar's high of 50500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is False
        assert "Stop not triggered" in result.reason

    def test_sell_stop_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test sell stop triggers when price drops to stop price."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("1"),
            stop_price=Decimal("49600"),  # Within bar's low of 49500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is True

    def test_sell_stop_not_triggered(
        self, fill_model: OrderFillModel, current_bar: Candle, next_bar: Candle
    ) -> None:
        """Test sell stop not triggered when price doesn't drop to stop."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=Decimal("49000"),  # Below bar's low of 49500
        )

        result = fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
            next_bar=next_bar,
        )

        assert result.filled is False

    def test_stop_order_without_stop_price_raises(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test that stop order without stop_price raises ValueError."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=None,
        )

        with pytest.raises(ValueError, match="Stop order must have stop_price"):
            fill_model.try_fill_stop_order(
                order=order,
                current_bar=current_bar,
                best_bid=Decimal("50100"),
                best_ask=Decimal("50150"),
            )
