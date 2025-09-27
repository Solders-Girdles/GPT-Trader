from collections import deque

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig


class DummyWS:
    def __init__(self, msgs):
        self.msgs = deque(msgs)
        self.subs = []

    def subscribe(self, sub):
        self.subs.append(sub)

    def stream_messages(self):
        while self.msgs:
            yield self.msgs.popleft()


def test_adapter_stream_trades_wires_subscription(monkeypatch):
    msgs = [{"type": "trade", "product_id": "BTC-USD"}, {"type": "trade", "product_id": "ETH-USD"}]
    ws = DummyWS(msgs)

    def fake_factory(self):
        return ws

    b = CoinbaseBrokerage(APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api"))
    # Inject ws factory
    monkeypatch.setattr(CoinbaseBrokerage, "_create_ws", fake_factory)
    out = []
    for m in b.stream_trades(["BTC-USD", "ETH-USD"]):
        out.append(m)
    assert len(ws.subs) == 1
    sub = ws.subs[0]
    assert sorted(sub.product_ids) == ["BTC-USD", "ETH-USD"]
    assert [m["product_id"] for m in out] == ["BTC-USD", "ETH-USD"]

