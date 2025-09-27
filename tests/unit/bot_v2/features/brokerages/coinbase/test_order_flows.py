from decimal import Decimal

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce, OrderStatus
from bot_v2.features.brokerages.coinbase import client as client_mod


def make_broker():
    # Avoid real network; do not call connect()
    return CoinbaseBrokerage(APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api"))


def test_cancel_order_success(monkeypatch):
    broker = make_broker()

    def fake_cancel(self, ids):
        return {"results": [{"order_id": ids[0], "success": True}]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "cancel_orders", fake_cancel)
    assert broker.cancel_order("abc123") is True


def test_get_order_maps(monkeypatch):
    broker = make_broker()

    def fake_get(self, order_id):
        return {
            "id": order_id,
            "product_id": "BTC-USD",
            "side": "buy",
            "type": "limit",
            "size": "1",
            "price": "100",
            "status": "filled",
            "filled_size": "1",
            "average_filled_price": "100",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:01:00",
        }

    monkeypatch.setattr(client_mod.CoinbaseClient, "get_order_historical", fake_get)
    o = broker.get_order("OID")
    assert o.id == "OID"
    assert o.status == OrderStatus.FILLED


def test_list_orders_maps(monkeypatch):
    broker = make_broker()

    def fake_list(self, **params):
        return {"orders": [
            {"id": "o1", "product_id": "BTC-USD", "side": "buy", "type": "market", "size": "1", "status": "filled", "filled_size": "1", "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:10"},
            {"id": "o2", "product_id": "ETH-USD", "side": "sell", "type": "limit", "size": "2", "price": "2000", "status": "open", "filled_size": "0", "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T00:00:10"}
        ]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_orders", fake_list)
    out = broker.list_orders()
    assert len(out) == 2
    assert out[0].id == "o1" and out[1].id == "o2"


def test_list_orders_returns_empty_on_error(monkeypatch):
    broker = make_broker()

    def boom(self, **params):
        raise RuntimeError("501 method not allowed")

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_orders", boom)
    assert broker.list_orders() == []


def test_list_fills(monkeypatch):
    broker = make_broker()

    def fake_fills(self, **params):
        return {"fills": [{"trade_id": "t1"}, {"trade_id": "t2"}]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_fills", fake_fills)
    fills = broker.list_fills("BTC-USD", limit=2)
    assert [f["trade_id"] for f in fills] == ["t1", "t2"]


def test_list_fills_returns_empty_on_error(monkeypatch, caplog):
    broker = make_broker()

    def boom(self, **params):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_fills", boom)
    caplog.set_level("ERROR")
    assert broker.list_fills("BTC-USD") == []
    assert any("Failed to list fills" in rec.message for rec in caplog.records)
