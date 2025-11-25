"""Coinbase account and portfolio management tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest

import gpt_trader.features.brokerages.coinbase.client as client_mod
from gpt_trader.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.core.interfaces import InvalidRequestError, MarketType, Product
from tests.unit.gpt_trader.features.brokerages.coinbase.test_helpers import (
    ACCOUNT_ENDPOINT_CASES,
    CoinbaseBrokerage,
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
                    leverage_max=None,
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

    def test_adapter_cfm_telemetry_delegates_to_rest_service(self) -> None:
        adapter = make_adapter()
        adapter.rest_service.get_cfm_balance_summary = Mock(
            return_value={"portfolio_value": Decimal("100.5")}
        )
        adapter.rest_service.list_cfm_sweeps = Mock(return_value=[{"amount": Decimal("5.25")}])
        adapter.rest_service.get_cfm_sweeps_schedule = Mock(return_value={"windows": ["08:00Z"]})
        adapter.rest_service.get_cfm_margin_window = Mock(return_value={"margin_window": "TEST"})
        adapter.rest_service.update_cfm_margin_window = Mock(return_value={"status": "ok"})

        summary = adapter.get_cfm_balance_summary()
        sweeps = adapter.list_cfm_sweeps()
        schedule = adapter.get_cfm_sweeps_schedule()
        margin = adapter.get_cfm_margin_window()
        update = adapter.update_cfm_margin_window("TEST")

        assert summary["portfolio_value"] == Decimal("100.5")
        assert sweeps[0]["amount"] == Decimal("5.25")
        assert schedule["windows"] == ["08:00Z"]
        assert margin["margin_window"] == "TEST"
        assert update["status"] == "ok"
        adapter.rest_service.get_cfm_balance_summary.assert_called_once()
        adapter.rest_service.list_cfm_sweeps.assert_called_once()
        adapter.rest_service.get_cfm_sweeps_schedule.assert_called_once()
        adapter.rest_service.get_cfm_margin_window.assert_called_once()
        adapter.rest_service.update_cfm_margin_window.assert_called_once_with(
            "TEST",
            effective_time=None,
            extra_payload=None,
        )

    def test_adapter_cfm_telemetry_respects_derivatives_gating(self) -> None:
        adapter = make_adapter(enable_derivatives=False)
        adapter.client.cfm_balance_summary = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_sweeps = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_sweeps_schedule = Mock(side_effect=AssertionError("should not call"))
        adapter.client.cfm_intraday_current_margin_window = Mock(
            side_effect=AssertionError("should not call")
        )

        summary = adapter.get_cfm_balance_summary()
        sweeps = adapter.list_cfm_sweeps()
        schedule = adapter.get_cfm_sweeps_schedule()
        margin = adapter.get_cfm_margin_window()

        assert summary == {}
        assert sweeps == []
        assert schedule == {}
        assert margin == {}

        with pytest.raises(InvalidRequestError):
            adapter.update_cfm_margin_window("TEST")

    def test_adapter_intx_methods_delegate_to_rest_service(self) -> None:
        adapter = make_adapter()
        adapter.rest_service.intx_allocate = Mock(return_value={"status": "accepted"})
        adapter.rest_service.get_intx_balances = Mock(return_value=[{"asset": "USD"}])
        adapter.rest_service.get_intx_portfolio = Mock(return_value={"uuid": "pf-1"})
        adapter.rest_service.list_intx_positions = Mock(return_value=[{"symbol": "BTC"}])
        adapter.rest_service.get_intx_position = Mock(return_value={"symbol": "ETH"})
        adapter.rest_service.get_intx_multi_asset_collateral = Mock(
            return_value={"ratio": Decimal("0.5")}
        )

        allocation = adapter.intx_allocate({"amount": "10"})
        balances = adapter.get_intx_balances("pf-1")
        portfolio = adapter.get_intx_portfolio("pf-1")
        positions = adapter.list_intx_positions("pf-1")
        position = adapter.get_intx_position("pf-1", "ETH")
        collateral = adapter.get_intx_multi_asset_collateral()

        assert allocation["status"] == "accepted"
        assert balances[0]["asset"] == "USD"
        assert portfolio["uuid"] == "pf-1"
        assert positions[0]["symbol"] == "BTC"
        assert position["symbol"] == "ETH"
        assert collateral["ratio"] == Decimal("0.5")
        adapter.rest_service.intx_allocate.assert_called_once()
        adapter.rest_service.get_intx_balances.assert_called_once_with("pf-1")
        adapter.rest_service.get_intx_portfolio.assert_called_once_with("pf-1")
        adapter.rest_service.list_intx_positions.assert_called_once_with("pf-1")
        adapter.rest_service.get_intx_position.assert_called_once_with("pf-1", "ETH")
        adapter.rest_service.get_intx_multi_asset_collateral.assert_called_once()

    def test_adapter_intx_methods_respect_intx_support(self) -> None:
        adapter = make_adapter(api_mode="exchange")

        balances = adapter.get_intx_balances("pf-1")
        portfolio = adapter.get_intx_portfolio("pf-1")
        positions = adapter.list_intx_positions("pf-1")
        position = adapter.get_intx_position("pf-1", "BTC")
        collateral = adapter.get_intx_multi_asset_collateral()

        assert balances == []
        assert portfolio == {}
        assert positions == []
        assert position == {}
        assert collateral == {}

        with pytest.raises(InvalidRequestError):
            adapter.intx_allocate({"amount": "5"})

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
        assert snapshot["cfm_balance_summary"]["portfolio_value"] == "250.50"
        assert snapshot["cfm_sweeps"][0]["sweep_id"] == "sweep-1"
        assert snapshot["cfm_sweeps_schedule"]["windows"][0] == "00:00Z"
        assert snapshot["cfm_margin_window"]["margin_window"] == "INTRADAY_STANDARD"
        assert snapshot["intx_available"] is True
        assert snapshot["intx_portfolio_uuid"] == "pf-1"
        assert snapshot["intx_balances"][0]["asset"] == "USD"
        assert snapshot["intx_positions"][0]["symbol"] == "BTC-USD"
        assert snapshot["intx_collateral"]["collateral_value"] == "750.00"
        assert any(
            metric[1].get("event_type") == "account_manager_snapshot" for metric in store.metrics
        )

    def test_account_manager_intx_unavailable_marks_reason(self) -> None:
        broker = StubBroker()
        broker.intx_supported = False
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["intx_available"] is False
        assert snapshot["intx_unavailable_reason"] in {
            "intx_not_supported",
            "intx_portfolio_not_found",
        }
        assert snapshot["intx_balances"] == []

    def test_account_manager_intx_recovers_after_refresh(self) -> None:
        class FailingIntxBroker(StubBroker):
            def __init__(self) -> None:
                super().__init__()
                self.intx_resolved_uuid = "pf-bad"

            def get_intx_balances(self, portfolio_uuid=None):
                if portfolio_uuid == "pf-bad":
                    raise InvalidRequestError("bad portfolio")
                return super().get_intx_balances(portfolio_uuid)

            def resolve_intx_portfolio(self, preferred_uuid=None, refresh=False):
                if refresh:
                    return "pf-1"
                return super().resolve_intx_portfolio(preferred_uuid, refresh)

        broker = FailingIntxBroker()
        store = StubEventStore()
        manager = CoinbaseAccountManager(broker, event_store=store)

        snapshot = manager.snapshot()

        assert snapshot["intx_available"] is True
        assert snapshot["intx_portfolio_uuid"] == "pf-1"

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
