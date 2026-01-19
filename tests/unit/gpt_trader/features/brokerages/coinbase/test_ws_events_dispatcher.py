"""Tests for Coinbase WebSocket EventDispatcher."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws_events import EventDispatcher


class TestEventDispatcher:
    """Tests for EventDispatcher."""

    def test_dispatch_ticker(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_ticker(lambda e: received.append(e))

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1
        assert received[0].product_id == "BTC-USD"

    def test_dispatch_trades(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_trade(lambda e: received.append(e))

        message = {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "ETH-USD",
                            "trade_id": "t1",
                            "price": "2500",
                            "size": "1",
                            "side": "buy",
                        }
                    ]
                }
            ],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1
        assert received[0].product_id == "ETH-USD"

    def test_dispatch_orderbook(self) -> None:
        dispatcher = EventDispatcher()
        received = []

        dispatcher.on_orderbook(lambda e: received.append(e))

        message = {
            "channel": "l2_data",
            "events": [{"updates": [{"side": "bid", "price_level": "100", "new_quantity": "10"}]}],
        }
        dispatcher.dispatch(message)

        assert len(received) == 1

    def test_dispatch_raw_always_called(self) -> None:
        dispatcher = EventDispatcher()
        raw_messages = []

        dispatcher.on_raw(lambda m: raw_messages.append(m))

        message = {"channel": "ticker", "data": "test"}
        dispatcher.dispatch(message)

        assert len(raw_messages) == 1
        assert raw_messages[0] == message

    def test_dispatch_error_event(self) -> None:
        dispatcher = EventDispatcher()
        errors = []

        dispatcher.on_error(lambda e: errors.append(e))

        message = {"type": "error", "message": "subscription failed"}
        dispatcher.dispatch(message)

        assert len(errors) == 1
        assert errors[0]["message"] == "subscription failed"

    def test_handler_exception_does_not_stop_dispatch(self) -> None:
        dispatcher = EventDispatcher()
        results = []

        def failing_handler(e):
            raise ValueError("handler error")

        def working_handler(e):
            results.append(e)

        dispatcher.on_ticker(failing_handler)
        dispatcher.on_ticker(working_handler)

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)  # Should not raise

        # Working handler should still be called
        assert len(results) == 1

    def test_multiple_handlers(self) -> None:
        dispatcher = EventDispatcher()
        results1 = []
        results2 = []

        dispatcher.on_ticker(lambda e: results1.append(e))
        dispatcher.on_ticker(lambda e: results2.append(e))

        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "BTC-USD", "price": "50000"}]}],
        }
        dispatcher.dispatch(message)

        assert len(results1) == 1
        assert len(results2) == 1
