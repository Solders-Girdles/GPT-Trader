from collections import deque

from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription


class FakeTransport:
    def __init__(self, batches):
        # batches is a list of (messages_list, should_error) tuples
        self.batches = deque(batches)
        self.subscriptions = []
        self.connected = False

    def connect(self, url):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, payload):
        self.subscriptions.append(payload)

    def stream(self):
        if not self.batches:
            return iter(())
        msgs, err = self.batches.popleft()
        for m in msgs:
            yield m
        if err:
            raise RuntimeError("simulated disconnect")


def test_ws_reconnect_and_resubscribe(monkeypatch):
    ws = CoinbaseWebSocket("wss://example", max_retries=3, base_delay=0)
    fake = FakeTransport([
        ([{"seq": 1}, {"seq": 2}], True),  # first stream yields 2 msgs then error
        ([{"seq": 3}], False),              # second stream yields 1 msg then ends
    ])
    ws.set_transport(fake)
    ws.subscribe(WSSubscription(channels=["market_trades"], product_ids=["BTC-USD"]))

    out = list(ws.stream_messages())
    assert [m["seq"] for m in out] == [1, 2, 3]
    # Should have subscribed at least twice (initial + after reconnect)
    assert len(fake.subscriptions) >= 2

