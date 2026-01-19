"""Tests for CoinbaseRestServiceCore order payload configurations."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreBuildOrderPayloadConfigurations(RestServiceCoreTestBase):
    def test_build_order_payload_limit_order(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test_client_123",
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "BUY"
        assert "order_configuration" in payload
        assert "limit_limit_gtc" in payload["order_configuration"]
        assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.1"
        assert payload["order_configuration"]["limit_limit_gtc"]["limit_price"] == "50000.00"
        assert payload["client_order_id"] == "test_client_123"

    def test_build_order_payload_market_order(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "SELL"
        assert "order_configuration" in payload
        assert "market_market_ioc" in payload["order_configuration"]
        assert payload["order_configuration"]["market_market_ioc"]["base_size"] == "0.1"
        assert "client_order_id" in payload
        assert payload["client_order_id"].startswith("perps_")

    def test_build_order_payload_stop_limit(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00"),
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["product_id"] == "BTC-USD"
        assert "order_configuration" in payload
        assert "stop_limit_stop_limit_gtc" in payload["order_configuration"]
        config = payload["order_configuration"]["stop_limit_stop_limit_gtc"]
        assert config["base_size"] == "0.1"
        assert config["limit_price"] == "50000.00"
        assert config["stop_price"] == "49000.00"

    def test_build_order_payload_ioc_fok_limit_orders(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.IOC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert "limit_limit_ioc" in payload["order_configuration"]

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.FOK,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert "limit_limit_fok" in payload["order_configuration"]

    def test_build_order_payload_fallback_configuration(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=Decimal("49000.00"),
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["type"] == "STOP"
        assert payload["size"] == "0.1"
        assert payload["time_in_force"] == "GTC"
        assert payload["price"] == "50000.00"
        assert payload["stop_price"] == "49000.00"
