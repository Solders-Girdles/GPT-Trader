"""Coinbase system and client infrastructure tests."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pytest

from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.brokerages.coinbase.errors import InsufficientFunds
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import (
    SYSTEM_ENDPOINT_CASES,
    CoinbaseBrokerage,
    make_client,
)

pytestmark = [pytest.mark.endpoints]


class TestCoinbaseSystem:
    @pytest.mark.parametrize("case", SYSTEM_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_system_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
            return 200, {}, json.dumps(case.get("response", {}))

        client.set_transport_for_testing(transport)

        result = getattr(client, case["method"])(*case.get("args", ()), **case.get("kwargs", {}))

        assert recorded["method"] == case["expected_method"]
        parsed = urlparse(recorded["url"])
        assert parsed.path.endswith(case["expected_path"])

        expected_query = case.get("expected_query")
        if expected_query is not None:
            assert parse_qs(parsed.query) == expected_query
        else:
            assert parsed.query in ("", None)

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result

    def test_connection_validation(self) -> None:
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
        )
        broker = CoinbaseBrokerage(config)
        mock_accounts_response = {
            "accounts": [{"uuid": "test-account-123", "currency": "USD", "balance": "100.00"}]
        }
        broker.client.get_accounts = MagicMock(return_value=mock_accounts_response)
        result = broker.connect()
        assert result is True
        assert broker._connected is True
        assert broker._account_id == "test-account-123"

    def test_position_list_spot_trading(self) -> None:
        config = APIConfig(
            api_key="test_key",
            api_secret="test_secret",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
            enable_derivatives=False,
        )
        broker = CoinbaseBrokerage(config)
        broker.client.list_positions = MagicMock(return_value={"positions": []})
        positions = broker.list_positions()
        assert positions == []
        broker.client.cfm_positions = MagicMock()
        positions = broker.list_positions()
        assert positions == []
        broker.client.cfm_positions.assert_not_called()

    def test_order_error_handling(self) -> None:
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase="test",
            base_url="https://api.coinbase.com",
            api_mode="advanced",
        )
        broker = CoinbaseBrokerage(config)

        # Mock client to raise InsufficientFunds
        broker.client.place_order = MagicMock(side_effect=InsufficientFunds("Not enough funds"))

        mock_product = MagicMock()
        mock_product.step_size = Decimal("0.001")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_size = Decimal("0.001")
        mock_product.min_notional = None
        mock_product.symbol = "BTC-USD"
        broker.product_catalog.get = MagicMock(return_value=mock_product)

        mock_product = MagicMock()
        mock_product.step_size = Decimal("0.001")
        mock_product.price_increment = Decimal("0.01")
        mock_product.min_size = Decimal("0.001")
        mock_product.min_notional = None
        mock_product.symbol = "BTC-USD"
        broker.product_catalog.get = MagicMock(return_value=mock_product)

        with pytest.raises(InsufficientFunds):
            broker.place_order(
                Order(
                    id="new",
                    symbol="BTC-USD",
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    quantity=Decimal("1.0"),
                    status=None,
                )
            )
