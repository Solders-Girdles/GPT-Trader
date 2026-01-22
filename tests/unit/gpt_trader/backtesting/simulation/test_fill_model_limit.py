"""Tests for OrderFillModel limit and stop order handling."""

from decimal import Decimal

import pytest
from tests.unit.gpt_trader.backtesting.simulation.fill_model_test_utils import (  # naming: allow
    make_order,  # naming: allow
)

from gpt_trader.backtesting.simulation.fill_model import OrderFillModel
from gpt_trader.core import Candle, OrderSide, OrderType

BID = Decimal("50100")
ASK = Decimal("50150")


@pytest.mark.parametrize(
    ("side", "price", "expected_filled", "expected_reason"),
    [
        (OrderSide.BUY, Decimal("49600"), True, None),
        (OrderSide.BUY, Decimal("49000"), False, "Price not touched"),
        (OrderSide.SELL, Decimal("50400"), True, None),
        (OrderSide.SELL, Decimal("51000"), False, "Price not touched"),
    ],
)
def test_limit_order_price_touch(
    fill_model: OrderFillModel,
    current_bar: Candle,
    side: OrderSide,
    price: Decimal,
    expected_filled: bool,
    expected_reason: str | None,
) -> None:
    order = make_order(
        side=side,
        order_type=OrderType.LIMIT,
        quantity=Decimal("10"),
        price=price,
    )

    result = fill_model.try_fill_limit_order(
        order=order,
        current_bar=current_bar,
        best_bid=BID,
        best_ask=ASK,
    )

    assert result.filled is expected_filled
    if expected_filled:
        assert result.fill_price == price
        assert result.is_maker is True
        assert result.slippage_bps == Decimal("0")
    else:
        assert expected_reason in result.reason


def test_limit_order_insufficient_volume(fill_model: OrderFillModel, current_bar: Candle) -> None:
    order = make_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        price=Decimal("49600"),
    )

    result = fill_model.try_fill_limit_order(
        order=order,
        current_bar=current_bar,
        best_bid=BID,
        best_ask=ASK,
    )

    assert result.filled is False
    assert "Insufficient volume" in result.reason


def test_limit_order_without_price_raises(fill_model: OrderFillModel, current_bar: Candle) -> None:
    order = make_order(
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=None,
    )

    with pytest.raises(ValueError, match="Limit order must have price"):
        fill_model.try_fill_limit_order(
            order=order,
            current_bar=current_bar,
            best_bid=BID,
            best_ask=ASK,
        )


@pytest.mark.parametrize(
    ("limit_price", "is_buy", "expected"),
    [
        (Decimal("100"), True, True),
        (Decimal("105"), True, True),
        (Decimal("99"), True, False),
        (Decimal("110"), False, True),
        (Decimal("105"), False, True),
        (Decimal("115"), False, False),
    ],
)
def test_limit_price_touch_logic(limit_price: Decimal, is_buy: bool, expected: bool) -> None:
    model = OrderFillModel()
    assert (
        model._is_price_touched(
            limit_price=limit_price,
            is_buy=is_buy,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("side", "stop_price", "expected_filled", "expected_reason"),
    [
        (OrderSide.BUY, Decimal("50400"), True, None),
        (OrderSide.BUY, Decimal("51000"), False, "Stop not triggered"),
        (OrderSide.SELL, Decimal("49600"), True, None),
        (OrderSide.SELL, Decimal("49000"), False, "Stop not triggered"),
    ],
)
def test_stop_order_triggering(
    fill_model: OrderFillModel,
    current_bar: Candle,
    next_bar: Candle,
    side: OrderSide,
    stop_price: Decimal,
    expected_filled: bool,
    expected_reason: str | None,
) -> None:
    order = make_order(
        side=side,
        order_type=OrderType.STOP,
        quantity=Decimal("1"),
        stop_price=stop_price,
    )

    result = fill_model.try_fill_stop_order(
        order=order,
        current_bar=current_bar,
        best_bid=BID,
        best_ask=ASK,
        next_bar=next_bar,
    )

    assert result.filled is expected_filled
    if expected_filled:
        assert result.is_maker is False
    else:
        assert expected_reason in result.reason


def test_stop_order_without_stop_price_raises(
    fill_model: OrderFillModel, current_bar: Candle
) -> None:
    order = make_order(
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        stop_price=None,
    )

    with pytest.raises(ValueError, match="Stop order must have stop_price"):
        fill_model.try_fill_stop_order(
            order=order,
            current_bar=current_bar,
            best_bid=BID,
            best_ask=ASK,
        )


@pytest.mark.parametrize(
    ("stop_price", "is_buy_stop", "expected"),
    [
        (Decimal("110"), True, True),
        (Decimal("105"), True, True),
        (Decimal("115"), True, False),
        (Decimal("100"), False, True),
        (Decimal("105"), False, True),
        (Decimal("95"), False, False),
    ],
)
def test_stop_trigger_logic(stop_price: Decimal, is_buy_stop: bool, expected: bool) -> None:
    model = OrderFillModel()
    assert (
        model._is_stop_triggered(
            stop_price=stop_price,
            is_buy_stop=is_buy_stop,
            bar_high=Decimal("110"),
            bar_low=Decimal("100"),
        )
        is expected
    )
