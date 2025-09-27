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


def test_user_stream_gap_detection(monkeypatch):
    ws = DummyWS([
        {"sequence": 1, "type": "order"},
        {"sequence": 2, "type": "fill"},
        {"sequence": 4, "type": "fill"},  # gap (missing 3)
    ])

    def fake_factory(*args, **kwargs):
        return ws

    b = CoinbaseBrokerage(APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api"))
    monkeypatch.setattr(CoinbaseBrokerage, "_create_ws", fake_factory)

    out = list(b.stream_user_events())
    assert len(ws.subs) == 1
    assert out[-1].get("gap_detected") is True
    assert out[-1].get("last_seq") == 2
