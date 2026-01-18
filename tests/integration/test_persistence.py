"""Integration tests for EventStore SQLite persistence."""

import tempfile
import threading
from pathlib import Path

import pytest

from gpt_trader.persistence.event_store import EventStore

pytestmark = pytest.mark.integration


class TestEventStorePersistence:
    """Test durable persistence across restarts."""

    def test_events_persist_across_restart(self) -> None:
        """Test that events survive EventStore restart."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            # First session: write events
            with EventStore(root=root) as store1:
                store1.append("trade", {"symbol": "BTC-USD", "side": "buy"})
                store1.append_metric("bot1", {"pnl": 100.0})
                store1.append_error(error="test error", bot_id="bot1")
                assert len(store1.list_events()) == 3

            # Second session: verify events loaded
            with EventStore(root=root) as store2:
                events = store2.list_events()
                assert len(events) == 3
                assert events[0]["type"] == "trade"
                assert events[0]["data"]["symbol"] == "BTC-USD"
                assert events[1]["type"] == "metric"
                assert events[2]["type"] == "error"

    def test_in_memory_mode_unchanged(self) -> None:
        """Test that in-memory mode works as before (no persistence)."""
        store = EventStore()  # No root = in-memory
        store.append("test", {"key": "value"})

        events = store.list_events()
        assert len(events) == 1
        assert events[0] == {"type": "test", "data": {"key": "value"}}
        assert store.root is None
        assert store._database is None

    def test_get_recent_from_persistent_store(self) -> None:
        """Test get_recent works with persistent storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with EventStore(root=root) as store:
                for i in range(10):
                    store.append("event", {"index": i})

                recent = store.get_recent(3)
                assert len(recent) == 3
                assert recent[0]["data"]["index"] == 7
                assert recent[2]["data"]["index"] == 9

    def test_large_event_persistence(self) -> None:
        """Test persistence of events with large payloads."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with EventStore(root=root) as store:
                large_data = {"items": list(range(10000))}
                store.append("large", large_data)

            with EventStore(root=root) as store2:
                assert store2.list_events()[0]["data"] == large_data

    def test_concurrent_writes(self) -> None:
        """Test thread safety of event writes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = EventStore(root=root)

            def write_events(thread_id: int) -> None:
                for i in range(100):
                    store.append("event", {"thread": thread_id, "index": i})

            threads = [threading.Thread(target=write_events, args=(i,)) for i in range(5)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(store.list_events()) == 500
            store.close()

            # Verify all persisted
            with EventStore(root=root) as store2:
                assert len(store2.list_events()) == 500

    def test_context_manager(self) -> None:
        """Test context manager support."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with EventStore(root=root) as store:
                store.append("test", {"value": 1})
                assert len(store.list_events()) == 1

            # After context, should be able to reopen
            with EventStore(root=root) as store2:
                assert len(store2.list_events()) == 1

    def test_cache_eviction(self) -> None:
        """Test that deque evicts oldest events when full."""
        store = EventStore(max_cache_size=5)

        for i in range(10):
            store.append("event", {"index": i})

        # Only last 5 should be in cache
        events = store.list_events()
        assert len(events) == 5
        assert events[0]["data"]["index"] == 5
        assert events[4]["data"]["index"] == 9

    def test_database_has_all_events_after_eviction(self) -> None:
        """Test that DB retains all events even when cache evicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with EventStore(root=root, max_cache_size=5) as store:
                for i in range(10):
                    store.append("event", {"index": i})

                # DB has all 10 (list_events reads from persistence in this mode)
                assert len(store.list_events()) == 10
                # get_recent provides the latest window
                assert len(store.get_recent(5)) == 5
                # DB has all 10
                assert store._database is not None
                assert store._database.event_count() == 10

    def test_root_property_with_root(self) -> None:
        """Test root property returns expected value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with EventStore(root=root) as store:
                assert store.root == root


class TestEventStoreSnapshots:
    """Test event list snapshots."""

    def test_list_events_empty_and_populated(self) -> None:
        """Test list_events snapshot behavior."""
        store = EventStore()
        assert store.list_events() == []
        store.append("test", {"key": "value"})
        assert store.list_events() == [{"type": "test", "data": {"key": "value"}}]

    def test_append_trade_both_signatures(self) -> None:
        """Test both append_trade signatures work."""
        store = EventStore()
        store.append_trade("bot1", {"symbol": "BTC-USD"})
        assert store.list_events()[0]["data"]["bot_id"] == "bot1"
        store.append_trade({"bot_id": "bot2", "symbol": "ETH-USD"})
        assert store.list_events()[1]["data"]["bot_id"] == "bot2"

    def test_bot_id_extraction(self) -> None:
        """Test bot_id extraction for database indexing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with EventStore(root=root) as store:
                store.append_metric("bot1", {"pnl": 100})
                store.append_position("bot2", {"symbol": "BTC-USD"})
                assert store._database is not None
                assert len(store._database.read_events_by_bot("bot1")) == 1
                assert len(store._database.read_events_by_bot("bot2")) == 1
