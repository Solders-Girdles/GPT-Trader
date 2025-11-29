"""Tests for EventStore persistence functionality.

NOTE: The EventStore is currently a minimal in-memory implementation.
See docs/archive/PROJECT_ROADMAP_NOV_2025.md "Future Activation" section - "EventStore need production hardening".
"""

from pathlib import Path

from gpt_trader.persistence.event_store import EventStore


class TestEventStore:
    """Test the EventStore class."""

    def test_event_store_init_default_path(self) -> None:
        """Test EventStore initialization with default path."""
        store = EventStore()
        assert store.events == []
        assert store.path is None

    def test_event_store_init_custom_path(self) -> None:
        """Test EventStore initialization with custom path."""
        custom_root = Path("/tmp/test_events")
        store = EventStore(root=custom_root)

        expected_path = custom_root / "events.jsonl"
        assert store.path == expected_path
        assert store.events == []

    def test_append_adds_to_memory(self) -> None:
        """Test append method adds to in-memory list."""
        store = EventStore()
        store.append("test_type", {"key": "value"})

        assert len(store.events) == 1
        assert store.events[0] == {"type": "test_type", "data": {"key": "value"}}

    def test_append_trade(self) -> None:
        """Test append_trade method."""
        store = EventStore()
        trade_data = {"symbol": "BTC-USD", "side": "buy", "size": 0.1}
        store.append_trade("bot123", trade_data)

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "trade"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["symbol"] == "BTC-USD"
        assert event["data"]["side"] == "buy"

    def test_append_trade_dict_only(self) -> None:
        """Test append_trade method with dict only."""
        store = EventStore()
        trade_data = {"bot_id": "bot123", "symbol": "BTC-USD", "side": "buy"}
        store.append_trade(trade_data)

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "trade"
        assert event["data"] == trade_data

    def test_append_position(self) -> None:
        """Test append_position method."""
        store = EventStore()
        position_data = {
            "symbol": "BTC-USD",
            "quantity": 0.5,
            "mark_price": 51000,
        }
        store.append_position("bot123", position_data)

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "position"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["position"] == position_data

    def test_append_metric(self) -> None:
        """Test append_metric method."""
        store = EventStore()
        metric_data = {"portfolio_value": 10000.0}
        store.append_metric("bot123", metric_data)

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "metric"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["metrics"] == metric_data

    def test_append_error_without_context(self) -> None:
        """Test append_error method without context."""
        store = EventStore()
        store.append_error(error="Connection failed", bot_id="bot123")

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "error"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["error"] == "Connection failed"
        assert event["data"]["details"] == {}

    def test_append_error_with_context(self) -> None:
        """Test append_error method with context."""
        store = EventStore()
        context = {"retry_count": 3}
        store.append_error(error="Connection failed", details=context, bot_id="bot123")

        assert len(store.events) == 1
        event = store.events[0]
        assert event["type"] == "error"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["error"] == "Connection failed"
        assert event["data"]["details"] == context

    def test_store_event_alias(self) -> None:
        """Test store_event alias."""
        store = EventStore()
        store.store_event("custom", {"foo": "bar"})

        assert len(store.events) == 1
        assert store.events[0] == {"type": "custom", "data": {"foo": "bar"}}
