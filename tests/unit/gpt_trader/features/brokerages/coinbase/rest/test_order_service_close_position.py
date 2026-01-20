"""Tests for `OrderService.close_position`."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


def _mock_position(symbol: str, quantity: Decimal) -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.quantity = quantity
    return pos


class TestClosePosition(OrderServiceTestBase):
    def test_close_position_success(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        mock_position_provider.list_positions.return_value = [
            _mock_position("BTC-PERP", Decimal("1.0"))
        ]
        mock_client.close_position.return_value = {"order": sample_order_response}

        result = order_service.close_position("BTC-PERP")

        assert isinstance(result, Order)
        mock_client.close_position.assert_called_once_with({"product_id": "BTC-PERP"})

    @pytest.mark.parametrize(
        "positions",
        [
            [],
            [_mock_position("BTC-PERP", Decimal("0"))],
            [_mock_position("ETH-PERP", Decimal("1.0"))],
        ],
    )
    def test_close_position_no_open_position_raises(
        self,
        order_service: OrderService,
        mock_position_provider: MagicMock,
        positions: list[MagicMock],
    ) -> None:
        mock_position_provider.list_positions.return_value = positions

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_with_client_order_id(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        mock_position_provider.list_positions.return_value = [
            _mock_position("ETH-PERP", Decimal("2.0"))
        ]
        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("ETH-PERP", client_order_id="my-close-123")

        mock_client.close_position.assert_called_once_with(
            {"product_id": "ETH-PERP", "client_order_id": "my-close-123"}
        )

    def test_close_position_fallback_on_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
    ) -> None:
        mock_position_provider.list_positions.return_value = [
            _mock_position("BTC-PERP", Decimal("1.0"))
        ]
        mock_client.close_position.side_effect = RuntimeError("API error")

        fallback_order = MagicMock()
        fallback = MagicMock(return_value=fallback_order)

        result = order_service.close_position("BTC-PERP", fallback=fallback)

        assert result is fallback_order
        fallback.assert_called_once()

    def test_close_position_no_fallback_raises(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
    ) -> None:
        mock_position_provider.list_positions.return_value = [
            _mock_position("BTC-PERP", Decimal("1.0"))
        ]
        mock_client.close_position.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            order_service.close_position("BTC-PERP")

    def test_close_position_finds_correct_symbol_among_multiple(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        mock_position_provider.list_positions.return_value = [
            _mock_position("ETH-PERP", Decimal("5.0")),
            _mock_position("BTC-PERP", Decimal("2.0")),
            _mock_position("SOL-PERP", Decimal("10.0")),
        ]
        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("BTC-PERP")

        mock_client.close_position.assert_called_once_with({"product_id": "BTC-PERP"})
