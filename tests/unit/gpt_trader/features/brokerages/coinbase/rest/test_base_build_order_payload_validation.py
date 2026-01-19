"""Tests for CoinbaseRestServiceCore order payload validation paths."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import (
    InvalidRequestError,
    NotFoundError,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.errors import ValidationError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreBuildOrderPayloadValidation(RestServiceCoreTestBase):
    def test_build_order_payload_quantize_values(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.123456"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        config = payload["order_configuration"]["limit_limit_gtc"]
        assert config["base_size"] == "0.12345678"
        assert config["limit_price"] == "50000.12"

    def test_build_order_payload_quantity_below_min_size(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        with pytest.raises(InvalidRequestError) as exc:
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.0001"),
                price=Decimal("50000.00"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )
        assert "Order quantity 0.00010000 is below minimum size 0.001" in str(exc.value)

    def test_build_order_payload_notional_below_min_notional(self) -> None:
        self.product_catalog.get.return_value = self.mock_product
        mock_quote = Mock()
        mock_quote.last = Decimal("50.00")
        self.service.get_rest_quote = Mock(return_value=mock_quote)

        with pytest.raises(InvalidRequestError) as exc:
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50.00"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )
        assert "Order notional 0.0500000000 is below minimum 10" in str(exc.value)

    def test_build_order_payload_limit_order_requires_price(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        with pytest.raises(ValidationError, match="price is required for limit orders"):
            self.service._build_order_payload(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=None,
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )

    def test_build_order_payload_product_not_found(self) -> None:
        self.product_catalog.get.side_effect = NotFoundError("Product not found")
        self.service.get_product = Mock(return_value=None)

        with pytest.raises(NotFoundError):
            self.service._build_order_payload(
                symbol="UNKNOWN-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("50000.00"),
                stop_price=None,
                tif=TimeInForce.GTC,
                client_id=None,
                reduce_only=False,
                leverage=None,
            )
