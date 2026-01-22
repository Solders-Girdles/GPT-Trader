"""Tests for CoinbaseRestServiceCore order payload build paths."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.core import InvalidRequestError, NotFoundError, OrderSide, OrderType, TimeInForce
from gpt_trader.errors import ValidationError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreBuildOrderPayload(RestServiceCoreTestBase):
    def _build_payload(self, **overrides: object) -> dict[str, object]:
        if self.product_catalog.get.side_effect is None:
            self.product_catalog.get.return_value = self.mock_product
        args = {
            "symbol": "BTC-USD",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": Decimal("0.1"),
            "price": Decimal("50000.00"),
            "stop_price": None,
            "tif": TimeInForce.GTC,
            "client_id": None,
            "reduce_only": False,
            "leverage": None,
        }
        args.update(overrides)
        return self.service._build_order_payload(**args)

    def test_build_order_payload_quantize_values(self) -> None:
        payload = self._build_payload(
            quantity=Decimal("0.123456789"),
            price=Decimal("50000.123456"),
        )

        config = payload["order_configuration"]["limit_limit_gtc"]
        assert config["base_size"] == "0.12345678"
        assert config["limit_price"] == "50000.12"

    def test_build_order_payload_quantity_below_min_size(self) -> None:
        with pytest.raises(InvalidRequestError) as exc:
            self._build_payload(quantity=Decimal("0.0001"))
        assert "Order quantity 0.00010000 is below minimum size 0.001" in str(exc.value)

    def test_build_order_payload_notional_below_min_notional(self) -> None:
        mock_quote = Mock()
        mock_quote.last = Decimal("50.00")
        self.service.get_rest_quote = Mock(return_value=mock_quote)

        with pytest.raises(InvalidRequestError) as exc:
            self._build_payload(quantity=Decimal("0.001"), price=Decimal("50.00"))
        assert "Order notional 0.0500000000 is below minimum 10" in str(exc.value)

    def test_build_order_payload_limit_order_requires_price(self) -> None:
        with pytest.raises(ValidationError, match="price is required for limit orders"):
            self._build_payload(price=None)

    def test_build_order_payload_product_not_found(self) -> None:
        self.product_catalog.get.side_effect = NotFoundError("Product not found")
        self.service.get_product = Mock(return_value=None)

        with pytest.raises(NotFoundError):
            self._build_payload(symbol="UNKNOWN-USD")

    def test_build_order_payload_limit_order(self) -> None:
        payload = self._build_payload(client_id="test_client_123")

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "BUY"
        assert "order_configuration" in payload
        assert "limit_limit_gtc" in payload["order_configuration"]
        assert payload["order_configuration"]["limit_limit_gtc"]["base_size"] == "0.1"
        assert payload["order_configuration"]["limit_limit_gtc"]["limit_price"] == "50000.00"
        assert payload["client_order_id"] == "test_client_123"

    def test_build_order_payload_market_order(self) -> None:
        payload = self._build_payload(
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=None,
            tif=TimeInForce.IOC,
        )

        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "SELL"
        assert "order_configuration" in payload
        assert "market_market_ioc" in payload["order_configuration"]
        assert payload["order_configuration"]["market_market_ioc"]["base_size"] == "0.1"
        assert payload["client_order_id"].startswith("perps_")

    def test_build_order_payload_stop_limit(self) -> None:
        payload = self._build_payload(
            order_type=OrderType.STOP_LIMIT,
            stop_price=Decimal("49000.00"),
        )

        assert payload["product_id"] == "BTC-USD"
        assert "order_configuration" in payload
        assert "stop_limit_stop_limit_gtc" in payload["order_configuration"]
        config = payload["order_configuration"]["stop_limit_stop_limit_gtc"]
        assert config["base_size"] == "0.1"
        assert config["limit_price"] == "50000.00"
        assert config["stop_price"] == "49000.00"

    @pytest.mark.parametrize(
        ("tif", "expected_key"),
        [
            (TimeInForce.IOC, "limit_limit_ioc"),
            (TimeInForce.FOK, "limit_limit_fok"),
        ],
    )
    def test_build_order_payload_ioc_fok_limit_orders(
        self, tif: TimeInForce, expected_key: str
    ) -> None:
        payload = self._build_payload(tif=tif)
        assert expected_key in payload["order_configuration"]

    def test_build_order_payload_fallback_configuration(self) -> None:
        payload = self._build_payload(
            order_type=OrderType.STOP,
            stop_price=Decimal("49000.00"),
        )

        assert payload["type"] == "STOP"
        assert payload["size"] == "0.1"
        assert payload["time_in_force"] == "GTC"
        assert payload["price"] == "50000.00"
        assert payload["stop_price"] == "49000.00"

    @pytest.mark.parametrize(
        ("overrides", "expected_key", "expected_value"),
        [
            ({"reduce_only": True}, "reduce_only", True),
            ({"leverage": 5}, "leverage", 5),
            ({"post_only": True}, "post_only", True),
        ],
    )
    def test_build_order_payload_flags(
        self, overrides: dict[str, object], expected_key: str, expected_value: object
    ) -> None:
        payload = self._build_payload(**overrides)
        assert payload[expected_key] == expected_value

    def test_build_order_payload_exclude_client_id(self) -> None:
        payload = self._build_payload(include_client_id=False)
        assert "client_order_id" not in payload

    def test_build_order_payload_enum_coercion(self) -> None:
        payload = self._build_payload(side="BUY", order_type="LIMIT", tif="GTC")

        assert payload["side"] == "BUY"
        assert payload["type"] == "LIMIT"
        assert payload["time_in_force"] == "GTC"

    def test_build_order_payload_invalid_side_passthrough(self) -> None:
        payload = self._build_payload(side="LONG", order_type=OrderType.MARKET, price=None)
        assert payload["side"] == "LONG"

    @pytest.mark.parametrize(
        ("order_type", "price", "tif", "expected_tif"),
        [
            (OrderType.LIMIT, Decimal("50000.00"), "GTD", "GTC"),
            (OrderType.MARKET, None, "DAY", "DAY"),
        ],
    )
    def test_build_order_payload_tif_handling(
        self,
        order_type: OrderType,
        price: Decimal | None,
        tif: str,
        expected_tif: str,
    ) -> None:
        payload = self._build_payload(order_type=order_type, price=price, tif=tif)
        assert payload["time_in_force"] == expected_tif
