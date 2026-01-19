"""Tests for CoinbaseRestServiceCore order payload flags and coercion."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreBuildOrderPayloadFlagsAndCoercion(RestServiceCoreTestBase):
    def test_build_order_payload_with_reduce_only(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=True,
            leverage=None,
        )

        assert payload["reduce_only"] is True

    def test_build_order_payload_with_leverage(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=5,
        )

        assert payload["leverage"] == 5

    def test_build_order_payload_post_only(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
            post_only=True,
        )

        assert payload["post_only"] is True

    def test_build_order_payload_exclude_client_id(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
            include_client_id=False,
        )

        assert "client_order_id" not in payload

    def test_build_order_payload_enum_coercion(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif="GTC",
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["side"] == "BUY"
        assert payload["type"] == "LIMIT"
        assert payload["time_in_force"] == "GTC"

    def test_build_order_payload_invalid_side_passthrough(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side="LONG",
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["side"] == "LONG"

    def test_build_order_payload_gtd_conversion(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif="GTD",
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["time_in_force"] == "GTC"

    def test_build_order_payload_invalid_tif_passthrough(self) -> None:
        self.product_catalog.get.return_value = self.mock_product

        payload = self.service._build_order_payload(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif="DAY",
            client_id=None,
            reduce_only=False,
            leverage=None,
        )

        assert payload["time_in_force"] == "DAY"
