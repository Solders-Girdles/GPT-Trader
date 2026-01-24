"""Tests for EventStore persistence functionality.

EventStore supports both in-memory mode (`root=None`) and SQLite persistence when a `root` is provided.
"""

from pathlib import Path

from gpt_trader.persistence.event_store import EventStore


class TestEventStore:
    """Test the EventStore class."""

    def test_event_store_init_default_path(self) -> None:
        """Test EventStore initialization with default path."""
        store = EventStore()
        assert store.list_events() == []
        assert store.root is None

    def test_event_store_init_custom_path(self) -> None:
        """Test EventStore initialization with custom path."""
        custom_root = Path("/tmp/test_events")
        store = EventStore(root=custom_root)

        assert store.root == custom_root
        assert store.list_events() == []

    def test_append_adds_to_memory(self) -> None:
        """Test append method adds to in-memory list."""
        store = EventStore()
        store.append("test_type", {"key": "value"})

        events = store.list_events()
        assert len(events) == 1
        assert events[0] == {"type": "test_type", "data": {"key": "value"}}

    def test_append_trade(self) -> None:
        """Test append_trade method."""
        store = EventStore()
        trade_data = {"symbol": "BTC-USD", "side": "buy", "size": 0.1}
        store.append_trade("bot123", trade_data)

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "trade"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["symbol"] == "BTC-USD"
        assert event["data"]["side"] == "buy"

    def test_append_trade_dict_only(self) -> None:
        """Test append_trade method with dict only."""
        store = EventStore()
        trade_data = {"bot_id": "bot123", "symbol": "BTC-USD", "side": "buy"}
        store.append_trade(trade_data)

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
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

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "position"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["position"] == position_data

    def test_append_metric(self) -> None:
        """Test append_metric method."""
        store = EventStore()
        metric_data = {"portfolio_value": 10000.0}
        store.append_metric("bot123", metric_data)

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "metric"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["metrics"] == metric_data

    def test_append_error_without_context(self) -> None:
        """Test append_error method without context."""
        store = EventStore()
        store.append_error(error="Connection failed", bot_id="bot123")

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "error"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["error"] == "Connection failed"
        assert event["data"]["details"] == {}

    def test_append_error_with_context(self) -> None:
        """Test append_error method with context."""
        store = EventStore()
        context = {"retry_count": 3}
        store.append_error(error="Connection failed", details=context, bot_id="bot123")

        events = store.list_events()
        assert len(events) == 1
        event = events[0]
        assert event["type"] == "error"
        assert event["data"]["bot_id"] == "bot123"
        assert event["data"]["error"] == "Connection failed"
        assert event["data"]["details"] == context

    def test_store_event_alias(self) -> None:
        """Test store_event alias."""
        store = EventStore()
        store.store_event("custom", {"foo": "bar"})

        events = store.list_events()
        assert len(events) == 1
        assert events[0] == {"type": "custom", "data": {"foo": "bar"}}


class TestEventStorePruning:
    """Test the EventStore pruning functionality."""

    def test_prune_memory_only_is_noop(self) -> None:
        """Prune on in-memory store returns 0."""
        store = EventStore()
        for i in range(10):
            store.append("test", {"i": i})

        # Memory-only stores don't have a database to prune
        pruned = store.prune(max_rows=5)
        assert pruned == 0
        # Events still in memory (limited by deque maxlen, not prune)
        assert len(store.list_events()) == 10

    def test_prune_persistent_removes_oldest(self, tmp_path: Path) -> None:
        """Prune on persistent store removes oldest events."""
        store = EventStore(root=tmp_path, max_cache_size=100)

        # Add 20 events
        for i in range(20):
            store.append("test", {"index": i})

        # Prune to keep only 10
        pruned = store.prune(max_rows=10)
        assert pruned == 10

        # Verify by reading from database directly
        assert store._database is not None
        remaining = store._database.event_count()
        assert remaining == 10

        # Verify the oldest events were removed
        events = store._database.read_all_events()
        indices = [e["data"]["index"] for e in events]
        # Should keep indices 10-19 (newest)
        assert indices == list(range(10, 20))

        store.close()

    def test_prune_when_under_limit_is_noop(self, tmp_path: Path) -> None:
        """Prune when under limit returns 0."""
        store = EventStore(root=tmp_path)

        for i in range(5):
            store.append("test", {"i": i})

        # Prune to 10, but we only have 5
        pruned = store.prune(max_rows=10)
        assert pruned == 0

        assert store._database is not None
        assert store._database.event_count() == 5

        store.close()

    def test_prune_with_zero_max_rows(self, tmp_path: Path) -> None:
        """Prune with max_rows=0 is a no-op (safety)."""
        store = EventStore(root=tmp_path)

        for i in range(5):
            store.append("test", {"i": i})

        # max_rows=0 should not delete anything (edge case protection)
        pruned = store.prune(max_rows=0)
        assert pruned == 0

        store.close()
