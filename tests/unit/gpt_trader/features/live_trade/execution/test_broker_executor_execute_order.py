"""Tests for `BrokerExecutor.execute_order`."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


def _execute_order(
    executor: BrokerExecutor,
    *,
    submit_id: str = "client-123",
    symbol: str = "BTC-USD",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("1.0"),
    price: Decimal | None = None,
    stop_price: Decimal | None = None,
    tif: TimeInForce | None = None,
    reduce_only: bool = False,
    leverage: int | None = None,
    use_retry: bool = False,
) -> Order:
    return executor.execute_order(
        submit_id=submit_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price,
        tif=tif,
        reduce_only=reduce_only,
        leverage=leverage,
        use_retry=use_retry,
    )


class TestBrokerExecutorExecuteOrder:
    def test_calls_broker_with_correct_args(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        mock_broker.place_order.return_value = sample_order

        result = _execute_order(
            executor,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            tif=TimeInForce.GTC,
        )

        mock_broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
            client_id="client-123",
        )
        assert result is sample_order

    @pytest.mark.parametrize(
        ("kwargs", "expected_place_order_kwargs"),
        [
            (
                {"order_type": OrderType.MARKET, "price": None, "tif": None},
                {"price": None, "tif": None},
            ),
            ({"reduce_only": True}, {"reduce_only": True}),
            ({"symbol": "BTC-PERP", "leverage": 10}, {"leverage": 10}),
            (
                {
                    "order_type": OrderType.STOP_LIMIT,
                    "side": OrderSide.SELL,
                    "price": Decimal("48000"),
                    "stop_price": Decimal("49000"),
                    "tif": TimeInForce.GTC,
                },
                {"stop_price": Decimal("49000")},
            ),
        ],
    )
    def test_passes_through_fields(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
        kwargs: dict,
        expected_place_order_kwargs: dict,
    ) -> None:
        mock_broker.place_order.return_value = sample_order

        _execute_order(executor, **kwargs)

        call_kwargs = mock_broker.place_order.call_args.kwargs
        for key, expected in expected_place_order_kwargs.items():
            assert call_kwargs[key] == expected

    def test_preserves_decimal_precision(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        mock_broker.place_order.return_value = sample_order
        precise_quantity = Decimal("0.00123456789")
        precise_price = Decimal("50000.123456")

        _execute_order(
            executor,
            order_type=OrderType.LIMIT,
            quantity=precise_quantity,
            price=precise_price,
            tif=TimeInForce.GTC,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["quantity"] == precise_quantity
        assert call_kwargs["price"] == precise_price

    def test_propagates_broker_exception(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        mock_broker.place_order.side_effect = RuntimeError("Broker unavailable")

        with pytest.raises(RuntimeError, match="Broker unavailable"):
            _execute_order(
                executor,
                order_type=OrderType.LIMIT,
                price=Decimal("50000"),
                tif=TimeInForce.GTC,
            )
