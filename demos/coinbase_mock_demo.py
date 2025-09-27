"""
Coinbase Brokerage Mock Demo (no network).

Demonstrates end-to-end usage of the Coinbase adapter with mocked HTTP and WS
transports: list products, list balances, place order, stream trades, cancel.
"""

import json
from collections import deque
from decimal import Decimal

from bot_v2.orchestration.broker_factory import create_brokerage


def main():
    broker = create_brokerage()  # Expect BROKER=coinbase in env

    # Inject mock HTTP transport
    def http_transport(method, url, headers, body, timeout):
        # Very small router by URL suffix
        if url.endswith("/api/v3/brokerage/products"):
            return 200, {"content-type": "application/json"}, json.dumps({
                "products": [
                    {
                        "id": "BTC-USD",
                        "base_currency": "BTC",
                        "quote_currency": "USD",
                        "base_min_size": "0.001",
                        "base_increment": "0.001",
                        "quote_increment": "0.1",
                        "min_notional": "10",
                    }
                ]
            })
        if url.endswith("/api/v3/brokerage/accounts"):
            return 200, {}, json.dumps({
                "accounts": [
                    {"currency": "USD", "balance": "1000", "available": "1000", "hold": "0"}
                ]
            })
        if "/orders" in url and method == "POST":
            payload = json.loads(body.decode("utf-8")) if body else {}
            resp = {
                "id": "demo-order-1",
                "product_id": payload.get("product_id"),
                "side": payload.get("side"),
                "type": payload.get("type"),
                "size": payload.get("size"),
                "price": payload.get("price"),
                "status": "open",
                "filled_size": "0",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
            return 200, {}, json.dumps(resp)
        if url.endswith("/api/v3/brokerage/orders/batch_cancel"):
            return 200, {}, json.dumps({"results": [{"order_id": "demo-order-1", "success": True}]})
        if "/ticker" in url:
            return 200, {}, json.dumps({"price": "100.0", "best_bid": "99.9", "best_ask": "100.1", "time": "2024-01-01T00:00:00"})
        # default empty
        return 200, {}, json.dumps({})

    broker.set_http_transport_for_testing(http_transport)

    # Inject mock WS factory
    class FakeWSTransport:
        def __init__(self, msgs):
            self.msgs = deque(msgs)
        def connect(self, url):
            pass
        def disconnect(self):
            pass
        def subscribe(self, payload):
            pass
        def stream(self):
            while self.msgs:
                yield self.msgs.popleft()

    def ws_factory():
        from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket
        ws = CoinbaseWebSocket("wss://demo")
        ws.set_transport(FakeWSTransport([{"type": "trade", "product_id": "BTC-USD", "price": "100.0"}]))
        return ws

    broker.set_ws_factory_for_testing(ws_factory)

    # Run demo actions
    broker.connect()
    balances = broker.list_balances()
    print("Balances:", [(b.asset, str(b.available)) for b in balances])

    products = broker.list_products()
    print("Products:", [p.symbol for p in products])

    # Import enums for order placement
    from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType

    order = broker.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.001"),
        price=Decimal("100.1"),
    )
    print("Order:", order)

    trades = list(broker.stream_trades(["BTC-USD"]))
    print("Trades:", trades)

    cancelled = broker.cancel_order(order.id)
    print("Cancelled:", cancelled)


if __name__ == "__main__":
    main()
