"""Coinbase account and portfolio management tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import bot_v2.features.brokerages.coinbase.client as client_mod
from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product

from tests.unit.bot_v2.features.brokerages.coinbase.test_helpers import (
    ACCOUNT_ENDPOINT_CASES,
    StubBroker,
    StubEventStore,
    _decode_body,
    make_adapter,
    make_client,
)


pytestmark = pytest.mark.endpoints


class TestCoinbaseAccounts:
    @pytest.mark.parametrize("case", ACCOUNT_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_account_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
            recorded["body"] = body
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

        expected_payload = case.get("expected_payload")
        if expected_payload is not None:
            assert _decode_body(recorded.get("body")) == expected_payload
        else:
            assert not recorded.get("body")

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result

    def test_list_balances_advanced_shape(self) -> None:
        adapter = make_adapter()
        payload = {
            "accounts": [
                {
                    "currency": "USD",
                    "available_balance": {"value": "100.25"},
                    "hold": {"value": "10.75"},
                },
                {"currency": "USDC", "available_balance": {"value": "50"}, "hold": "0"},
            ]
        }
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].available == Decimal("100.25")
        assert data["USD"].hold == Decimal("10.75")
        assert data["USD"].total == Decimal("111.00")
        assert data["USDC"].total == Decimal("50")

    def test_list_balances_exchange_shape(self) -> None:
        adapter = make_adapter(api_mode="exchange")
        payload = [
            {"currency": "USD", "balance": "200", "available": "150", "hold": "50"},
            {"currency": "BTC", "balance": "0.1", "available": "0.1", "hold": "0"},
        ]
        adapter.client.get_accounts = lambda: payload  # type: ignore[attr-defined]
        balances = adapter.list_balances()
        data = {balance.asset: balance for balance in balances}
        assert data["USD"].total == Decimal("200")
        assert data["USD"].available == Decimal("150")
        assert data["USD"].hold == Decimal("50")

    def test_funding_enrichment_uses_product_catalog(self) -> None:
        adapter = make_adapter()

        class Catalog:
            def get(self, client, symbol):
                return Product(
                    symbol=symbol,
                    base_asset="BTC",
                    quote_asset="USD",
                    market_type=MarketType.PERPETUAL,
                    min_size=Decimal("0.001"),
                    step_size=Decimal("0.001"),
                    min_notional=None,
                    price_increment=Decimal("0.01"),
                )

            def get_funding(self, client, symbol):
                return Decimal("0.0005"), datetime.utcnow() + timedelta(hours=8)

        adapter.product_catalog = Catalog()
        adapter.client.get_product = lambda pid: {
            "product_id": "BTC-USD-PERP",
            "base_increment": "0.001",
            "quote_increment": "0.01",
            "base_min_size": "0.001",
            "contract_type": "perpetual",
        }  # type: ignore[attr-defined]

        product = adapter.get_product("BTC-USD-PERP")
        assert product.funding_rate == Decimal("0.0005")
        assert isinstance(product.next_funding_time, datetime)

    def test_list_positions_maps_from_cfm(self, monkeypatch) -> None:
        broker = CoinbaseBrokerage(
            APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
        )
        broker.connect()

        def fake_positions(self):
            return {
                "positions": [
                    {
                        "product_id": "BTC-USD-PERP",
                        "size": "1.5",
                        "entry_price": "100",
                        "mark_price": "110",
                        "unrealized_pnl": "15",
                        "realized_pnl": "2",
                        "leverage": 5,
                        "side": "long",
                    },
                    {
                        "product_id": "ETH-USD-PERP",
                        "contracts": "2",
                        "avg_entry_price": "2000",
                        "index_price": "1950",
                        "unrealizedPnl": "-100",
                        "realizedPnl": "5",
                        "leverage": 3,
                        "side": "short",
                    },
                ]
            }

        monkeypatch.setattr(client_mod.CoinbaseClient, "list_positions", fake_positions)
        positions = broker.list_positions()
        assert len(positions) == 2
        assert positions[0].symbol == "BTC-USD-PERP" and positions[0].quantity == Decimal("1.5")
        assert positions[1].symbol == "ETH-USD-PERP" and positions[1].quantity == Decimal("2")

    def test_account_manager_snapshot_collects_all_sections(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["key_permissions"]["can_trade"] is True
        assert snapshot["fee_schedule"]["tier"] == "Advanced"
        assert snapshot["limits"]["max_order"] == "100000"
        assert snapshot["transaction_summary"]["total_volume"] == "12345"
        assert snapshot["payment_methods"][0]["id"] == "pm-1"
        assert snapshot["portfolios"][0]["uuid"] == "pf-1"
        assert any(
            metric[1].get("event_type") == "account_manager_snapshot" for metric in store.metrics
        )

    def test_account_manager_convert_commits_when_requested(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        result = manager.convert({"from": "USD", "to": "USDC", "amount": "100"}, commit=True)

        assert result["trade_id"] == "trade-1"
        assert any(call[0] == "convert_quote" for call in broker.calls)
        assert any(call[0] == "commit_trade" for call in broker.calls)
        assert any(metric[1].get("event_type") == "convert_commit" for metric in store.metrics)

    def test_account_manager_move_funds_delegates_to_broker(self) -> None:
        broker = StubBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        payload = {"from_portfolio": "pf-1", "to_portfolio": "pf-2", "amount": "5"}
        result = manager.move_funds(payload)

        assert result["status"] == "ok"
        assert ("move_funds", payload) in broker.calls
        assert any(metric[1].get("event_type") == "portfolio_move" for metric in store.metrics)
