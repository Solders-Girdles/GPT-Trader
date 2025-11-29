"""Tests for WebSocket event dispatcher."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.ws_events import (
    EventDispatcher,
    EventType,
    OrderbookUpdate,
    OrderUpdateEvent,
    TickerEvent,
    TradeEvent,
)


class TestTickerEvent:
    """Tests for TickerEvent parsing."""

    def test_from_message_basic(self) -> None:
        message = {
            "channel": "ticker",
            "timestamp": "2024-01-15T10:30:00Z",
            "events": [
                {
                    "type": "update",
                    "tickers": [
                        {
                            "product_id": "BTC-USD",
                            "price": "45000.50",
                            "best_bid": "44999.00",
                            "best_ask": "45001.00",
                            "volume_24_h": "1234.56",
                        }
                    ],
                }
            ],
        }

        event = TickerEvent.from_message(message)

        assert event.product_id == "BTC-USD"
        assert event.price == Decimal("45000.50")
        assert event.bid == Decimal("44999.00")
        assert event.ask == Decimal("45001.00")
        assert event.volume_24h == Decimal("1234.56")

    def test_from_message_missing_optional(self) -> None:
        message = {
            "channel": "ticker",
            "events": [{"tickers": [{"product_id": "ETH-USD", "price": "2500"}]}],
        }

        event = TickerEvent.from_message(message)

        assert event.product_id == "ETH-USD"
        assert event.price == Decimal("2500")
        assert event.bid is None
        assert event.ask is None


class TestTradeEvent:
    """Tests for TradeEvent parsing."""

    def test_from_message_single_trade(self) -> None:
        message = {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "BTC-USD",
                            "trade_id": "12345",
                            "price": "45000",
                            "size": "0.5",
                            "side": "buy",
                            "time": "2024-01-15T10:30:00Z",
                        }
                    ]
                }
            ],
        }

        events = TradeEvent.from_message(message)

        assert len(events) == 1
        assert events[0].product_id == "BTC-USD"
        assert events[0].trade_id == "12345"
        assert events[0].price == Decimal("45000")
        assert events[0].size == Decimal("0.5")
        assert events[0].side == "buy"

    def test_from_message_multiple_trades(self) -> None:
        message = {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "BTC-USD",
                            "trade_id": "1",
                            "price": "100",
                            "size": "1",
                            "side": "buy",
                        },
                        {
                            "product_id": "BTC-USD",
                            "trade_id": "2",
                            "price": "101",
                            "size": "2",
                            "side": "sell",
                        },
                    ]
                }
            ],
        }

        events = TradeEvent.from_message(message)

        assert len(events) == 2
        assert events[0].trade_id == "1"
        assert events[1].trade_id == "2"


class TestOrderbookUpdate:
    """Tests for OrderbookUpdate parsing."""

    def test_from_message_bids_and_asks(self) -> None:
        message = {
            "channel": "l2_data",
            "timestamp": "2024-01-15T10:30:00Z",
            "events": [
                {
                    "updates": [
                        {
                            "side": "bid",
                            "price_level": "44999",
                            "new_quantity": "10",
                            "product_id": "BTC-USD",
                        },
                        {
                            "side": "bid",
                            "price_level": "44998",
                            "new_quantity": "20",
                            "product_id": "BTC-USD",
                        },
                        {
                            "side": "offer",
                            "price_level": "45001",
                            "new_quantity": "5",
                            "product_id": "BTC-USD",
                        },
                    ]
                }
            ],
        }

        update = OrderbookUpdate.from_message(message)

        assert update.product_id == "BTC-USD"
        assert len(update.bids) == 2
        assert len(update.asks) == 1
        assert update.bids[0] == (Decimal("44999"), Decimal("10"))
        assert update.asks[0] == (Decimal("45001"), Decimal("5"))


class TestOrderUpdateEvent:
    """Tests for OrderUpdateEvent parsing."""

    def test_from_message_order_update(self) -> None:
        message = {
            "channel": "user",
            "events": [
                {
                    "type": "update",
                    "orders": [
                        {
                            "order_id": "order-123",
                            "client_order_id": "client-456",
                            "product_id": "BTC-USD",
                            "status": "OPEN",
                            "order_side": "BUY",
                            "order_type": "LIMIT",
                            "filled_size": "0",
                            "order_configuration": {"size": "1.5", "limit_price": "45000"},
                            "creation_time": "2024-01-15T10:30:00Z",
                        }
                    ],
                }
            ],
        }

        events = OrderUpdateEvent.from_message(message)

        assert len(events) == 1
        assert events[0].order_id == "order-123"
        assert events[0].status == "OPEN"
        assert events[0].side == "BUY"
        assert events[0].price == Decimal("45000")


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


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_values(self) -> None:
        assert EventType.TICKER.value == "ticker"
        assert EventType.LEVEL2.value == "l2_data"
        assert EventType.USER.value == "user"
        assert EventType.ERROR.value == "error"
