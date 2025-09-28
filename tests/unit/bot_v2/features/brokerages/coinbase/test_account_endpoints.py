"""Combined account and funding tests for Coinbase integration."""

import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase import client as client_mod
from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import MarketType, Product


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def _adapter(api_mode: str = "advanced") -> CoinbaseBrokerage:
    config = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode=api_mode,
        sandbox=False,
    )
    return CoinbaseBrokerage(config)


# ---------------------------------------------------------------------------
# CoinbaseClient account endpoints
# ---------------------------------------------------------------------------


def test_get_accounts_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"accounts": []})

    client.set_transport_for_testing(fake_transport)
    out = client.get_accounts()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/accounts")
    assert "accounts" in out


def test_get_account_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"account": {"uuid": "acc-123"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_account("acc-123")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/accounts/acc-123")
    assert "account" in out


def test_list_portfolios_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"portfolios": []})

    client.set_transport_for_testing(fake_transport)
    out = client.list_portfolios()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios")
    assert "portfolios" in out


def test_get_portfolio_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"portfolio": {"uuid": "port-456"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_portfolio("port-456")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios/port-456")
    assert "portfolio" in out


def test_get_portfolio_breakdown_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"breakdown": {"total_balance": "10000"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_portfolio_breakdown("port-789")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios/port-789/breakdown")
    assert "breakdown" in out


def test_move_funds_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"transfer_id": "mv-1", "status": "completed"})

    client.set_transport_for_testing(transport)
    payload = {
        "from_portfolio_uuid": "port-A",
        "to_portfolio_uuid": "port-B",
        "amount": "1000",
        "currency": "USD",
    }
    out = client.move_funds(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/portfolios/move_funds")
    assert sent == payload
    assert out["status"] == "completed"


# ---------------------------------------------------------------------------
# Adapter funding, balances, and positions
# ---------------------------------------------------------------------------


def test_list_balances_advanced_shape(monkeypatch):
    adapter = _adapter()
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


def test_list_balances_exchange_shape(monkeypatch):
    adapter = _adapter(api_mode="exchange")
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


def test_funding_enrichment_uses_product_catalog(monkeypatch):
    adapter = _adapter()

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


def make_position_broker() -> CoinbaseBrokerage:
    broker = CoinbaseBrokerage(
        APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
    )
    broker.connect()
    return broker


def test_list_positions_maps_from_cfm(monkeypatch):
    broker = make_position_broker()

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


# ---------------------------------------------------------------------------
# CoinbaseAccountManager helpers
# ---------------------------------------------------------------------------


class StubBroker:
    def __init__(self):
        self.calls = []

    def get_key_permissions(self):
        self.calls.append("key_permissions")
        return {"can_trade": True}

    def get_fee_schedule(self):
        self.calls.append("fee_schedule")
        return {"tier": "Advanced"}

    def get_account_limits(self):
        self.calls.append("limits")
        return {"max_order": "100000"}

    def get_transaction_summary(self):
        self.calls.append("transaction_summary")
        return {"total_volume": "12345"}

    def list_payment_methods(self):
        self.calls.append("payment_methods")
        return [{"id": "pm-1"}]

    def list_portfolios(self):
        self.calls.append("portfolios")
        return [{"uuid": "pf-1"}]

    def create_convert_quote(self, payload):
        self.calls.append(("convert_quote", payload))
        return {"trade_id": "trade-1", "quote_id": "q-1"}

    def commit_convert_trade(self, trade_id, payload):
        self.calls.append(("commit_trade", trade_id, payload))
        return {"trade_id": trade_id, "status": "pending"}

    def move_portfolio_funds(self, payload):
        self.calls.append(("move_funds", payload))
        return {"status": "ok", **payload}


class StubEventStore:
    def __init__(self):
        self.metrics = []

    def append_metric(self, bot_id, metrics):
        self.metrics.append((bot_id, metrics))


def test_account_manager_snapshot_collects_all_sections():
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


def test_account_manager_convert_commits_when_requested():
    broker = StubBroker()
    store = StubEventStore()
    manager = CoinbaseAccountManager(broker, event_store=store)

    result = manager.convert({"from": "USD", "to": "USDC", "amount": "100"}, commit=True)

    assert result["trade_id"] == "trade-1"
    assert any(call[0] == "convert_quote" for call in broker.calls)
    assert any(call[0] == "commit_trade" for call in broker.calls)
    assert any(metric[1].get("event_type") == "convert_commit" for metric in store.metrics)


def test_account_manager_move_funds_delegates_to_broker():
    broker = StubBroker()
    store = StubEventStore()
    manager = CoinbaseAccountManager(broker, event_store=store)

    payload = {"from_portfolio": "pf-1", "to_portfolio": "pf-2", "amount": "5"}
    result = manager.move_funds(payload)

    assert result["status"] == "ok"
    assert ("move_funds", payload) in broker.calls
    assert any(metric[1].get("event_type") == "portfolio_move" for metric in store.metrics)
