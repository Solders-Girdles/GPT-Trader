"""Tests for OrderFillModel limit order fill simulation."""

from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide, OrderType


class TestLimitOrderFill:
    """Test limit order fill simulation."""

    def test_buy_limit_price_touched_sufficient_volume(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test buy limit order fills when price drops to limit and volume is sufficient."""
        # Limit price within bar's low range
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),  # Volume threshold: 10 * 2 = 20, bar has 100
            price=Decimal("49600"),  # Below bar high, at/below bar low
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("49600")
        assert result.is_maker is True  # Limit orders are maker
        assert result.slippage_bps == Decimal("0")

    def test_buy_limit_price_not_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test buy limit order not filled when price doesn't drop to limit."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("49000"),  # Below bar's low of 49500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Price not touched" in result.reason

    def test_sell_limit_price_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test sell limit order fills when price rises to limit."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            price=Decimal("50400"),  # At/above bar high of 50500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is True
        assert result.fill_price == Decimal("50400")

    def test_sell_limit_price_not_touched(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test sell limit order not filled when price doesn't reach limit."""
        order = make_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("51000"),  # Above bar's high of 50500
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Price not touched" in result.reason

    def test_limit_order_insufficient_volume(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test limit order not filled when bar volume is insufficient."""
        # Order size 100, threshold 2x = 200, bar volume is 100
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            price=Decimal("49600"),
        )

        result = fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=Decimal("50100"),
            best_ask=Decimal("50150"),
        )

        assert result.filled is False
        assert "Insufficient volume" in result.reason

    def test_limit_order_without_price_raises(
        self, fill_model: OrderFillModel, current_bar: Candle
    ) -> None:
        """Test that limit order without price raises ValueError."""
        order = make_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=None,
        )

        with pytest.raises(ValueError, match="Limit order must have price"):
            fill_model.try_fill_limit_order(
                order=order,
                current_bar=current_bar,
                best_bid=Decimal("50100"),
                best_ask=Decimal("50150"),
            )


class TestPriceTouchLogic:
    """Test price touch detection for limit orders."""

    def test_buy_limit_touched_at_exact_low(self) -> None:
        """Test buy limit touched when limit equals bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("100"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_buy_limit_touched_above_low(self) -> None:
        """Test buy limit touched when limit is above bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("105"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_buy_limit_not_touched_below_low(self) -> None:
        """Test buy limit not touched when limit is below bar low."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("99"),
            is_buy=True,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is False

    def test_sell_limit_touched_at_exact_high(self) -> None:
        """Test sell limit touched when limit equals bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("110"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_sell_limit_touched_below_high(self) -> None:
        """Test sell limit touched when limit is below bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("105"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is True

    def test_sell_limit_not_touched_above_high(self) -> None:
        """Test sell limit not touched when limit is above bar high."""
        model = OrderFillModel()
        touched = model._is_price_touched(
            limit_price=Decimal("115"),
            is_buy=False,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        assert touched is False
