"""Tests for `OrderService.close_position`."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestClosePosition(OrderServiceTestBase):
    """Tests for close_position method."""

    def test_close_position_success(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test successful position close."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        result = order_service.close_position("BTC-PERP")

        assert isinstance(result, Order)
        mock_client.close_position.assert_called_once()

    def test_close_position_no_position_raises(
        self,
        order_service: OrderService,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that ValidationError is raised when no position exists."""
        mock_position_provider.list_positions.return_value = []

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_zero_quantity_raises(
        self,
        order_service: OrderService,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that zero quantity position is not considered open."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0")
        mock_position_provider.list_positions.return_value = [mock_position]

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_with_client_order_id(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test passing client_order_id."""
        mock_position = MagicMock()
        mock_position.symbol = "ETH-PERP"
        mock_position.quantity = Decimal("2.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("ETH-PERP", client_order_id="my-close-123")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["client_order_id"] == "my-close-123"

    def test_close_position_fallback_on_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that fallback is called on exception."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.side_effect = RuntimeError("API error")

        fallback_order = Order(
            id="fallback-order",
            client_id="fallback-client",
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
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
        """Test that exception is raised when no fallback provided."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            order_service.close_position("BTC-PERP")

    def test_close_position_finds_correct_symbol(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test that correct symbol position is found among multiple."""
        pos1 = MagicMock()
        pos1.symbol = "ETH-PERP"
        pos1.quantity = Decimal("5.0")

        pos2 = MagicMock()
        pos2.symbol = "BTC-PERP"
        pos2.quantity = Decimal("2.0")

        pos3 = MagicMock()
        pos3.symbol = "SOL-PERP"
        pos3.quantity = Decimal("10.0")

        mock_position_provider.list_positions.return_value = [pos1, pos2, pos3]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("BTC-PERP")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["product_id"] == "BTC-PERP"
