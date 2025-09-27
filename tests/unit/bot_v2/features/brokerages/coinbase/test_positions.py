from decimal import Decimal

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase import client as client_mod


def make_broker():
    b = CoinbaseBrokerage(APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api"))
    b.connect()
    return b


def test_list_positions_maps_from_cfm(monkeypatch):
    broker = make_broker()

    def fake_positions(self):
        return {"positions": [
            {"product_id": "BTC-USD-PERP", "size": "1.5", "entry_price": "100", "mark_price": "110", "unrealized_pnl": "15", "realized_pnl": "2", "leverage": 5, "side": "long"},
            {"product_id": "ETH-USD-PERP", "contracts": "2", "avg_entry_price": "2000", "index_price": "1950", "unrealizedPnl": "-100", "realizedPnl": "5", "leverage": 3, "side": "short"}
        ]}

    monkeypatch.setattr(client_mod.CoinbaseClient, "list_positions", fake_positions)
    pos = broker.list_positions()
    assert len(pos) == 2
    assert pos[0].symbol == "BTC-USD-PERP" and pos[0].qty == Decimal("1.5") and pos[0].side == "long"
    assert pos[1].symbol == "ETH-USD-PERP" and pos[1].qty == Decimal("2") and pos[1].side == "short"
