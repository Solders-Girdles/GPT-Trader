"""Contract tests for Coinbase REST OrderService place/cancel flows."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.base as rest_base_module
from gpt_trader.core import InsufficientFunds, InvalidRequestError, Order, OrderSide, OrderType
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.errors import OrderCancellationError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuiteOrderServicePlaceAndCancel(CoinbaseRestContractSuiteBase):
    def test_place_order_quantity_resolution_success(
        self,
        order_service,
        service_core,
        mock_product_catalog,
        mock_product,
        mock_client,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test successful order placement with quantity resolution."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.return_value = {"order_id": "test_123"}

        original_execute = service_core.execute_order_payload
        mock_execute = Mock(side_effect=original_execute)
        monkeypatch.setattr(service_core, "execute_order_payload", mock_execute)
        monkeypatch.setattr(rest_base_module, "to_order", Mock(return_value=Mock(spec=Order)))

        order = order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.00"),
        )

        assert order is not None
        call_args = mock_execute.call_args
        payload = call_args[0][1]
        assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.12345678"

    def test_place_order_quantity_resolution_error_branch(
        self, order_service, mock_product_catalog, mock_product
    ):
        """Test order placement with quantity below minimum."""
        mock_product_catalog.get.return_value = mock_product

        with pytest.raises(InvalidRequestError, match="quantity .* is below minimum size"):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_insufficient_funds(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order placement with insufficient funds error."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.side_effect = InsufficientFunds("Insufficient balance")

        with pytest.raises(InsufficientFunds):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_place_order_error_branch_validation_error(
        self, order_service, mock_product_catalog, mock_product, mock_client
    ):
        """Test order placement with validation error."""
        mock_product_catalog.get.return_value = mock_product
        mock_client.place_order.side_effect = ValidationError("Invalid order parameters")

        with pytest.raises(ValidationError):
            order_service.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
            )

    def test_cancel_order_success(self, order_service, mock_client):
        """Test successful order cancellation."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": True}]
        }

        result = order_service.cancel_order("test_123")

        assert result is True

    def test_cancel_order_failure(self, order_service, mock_client):
        """Test order cancellation failure raises OrderCancellationError."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "test_123", "success": False}]
        }

        with pytest.raises(OrderCancellationError, match="Cancellation rejected"):
            order_service.cancel_order("test_123")
