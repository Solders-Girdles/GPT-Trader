"""Tests for EventStore - append-only event logging for audit and analysis.

This module tests the EventStore's ability to record trading events in JSONL
format (one JSON object per line) for audit trails, debugging, and analysis.
Tests verify:

- Event recording (trades, positions, metrics, errors)
- JSONL file format correctness
- Thread-safe concurrent writes
- Query functionality (tail, filtering by type)
- Dataclass serialization
- Timestamp handling

Audit Context:
    The EventStore provides an immutable audit trail of all trading activity.
    This is critical for:
    - Regulatory compliance (MiFID II, SEC Rule 17a-4)
    - Post-trade analysis and strategy debugging
    - Incident investigation when things go wrong
    - Performance attribution

    Loss of events or corrupted event logs could result in:
    - Regulatory violations and fines
    - Inability to reconstruct trading decisions
    - Hidden bugs that can't be debugged without event history
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from bot_v2.persistence.event_store import EventStore


@pytest.fixture
def temp_store(tmp_path: Path) -> EventStore:
    """Create an EventStore with temporary storage."""
    return EventStore(root=tmp_path)


@dataclass
class SamplePosition:
    """Sample dataclass for testing serialization."""

    symbol: str
    size: float
    entry_price: float


class TestEventStoreInitialization:
    """Test EventStore initialization and file creation."""

    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        """EventStore creates directory structure on initialization.

        Ensures the store can bootstrap itself in new environments without
        manual directory setup.
        """
        store_root = tmp_path / "nested" / "path"
        assert not store_root.exists()

        store = EventStore(root=store_root)

        assert store_root.exists()
        assert store.path == store_root / "events.jsonl"

    def test_creates_empty_event_file(self, tmp_path: Path) -> None:
        """EventStore creates empty events.jsonl on first initialization.

        The file should exist and be empty (not containing any events),
        ready to receive appended events.
        """
        store = EventStore(root=tmp_path)

        assert store.path.exists()
        content = store.path.read_text()
        assert content == ""  # Empty file initially

    def test_reuses_existing_file(self, tmp_path: Path) -> None:
        """EventStore reuses existing events.jsonl without truncating.

        Critical: Existing event history must be preserved across restarts.
        Truncating the file would destroy the audit trail.
        """
        events_file = tmp_path / "events.jsonl"
        existing_event = '{"type": "trade", "bot_id": "bot-1", "time": "2024-01-01T00:00:00"}\n'
        events_file.write_text(existing_event)

        store = EventStore(root=tmp_path)
        events = store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "trade"


class TestEventStoreTradeEvents:
    """Test trade event recording."""

    def test_append_trade_records_event(self, temp_store: EventStore) -> None:
        """append_trade() successfully records a trade event.

        Basic append operation - trade should be written to JSONL file
        with correct type and bot_id.
        """
        trade = {"symbol": "BTC-USD", "side": "buy", "quantity": 0.1, "price": 50000.0}

        temp_store.append_trade(bot_id="bot-1", trade=trade)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "trade"
        assert events[0]["bot_id"] == "bot-1"
        assert events[0]["symbol"] == "BTC-USD"
        assert events[0]["side"] == "buy"

    def test_append_trade_adds_timestamp(self, temp_store: EventStore) -> None:
        """append_trade() automatically adds timestamp if not present.

        Ensures all events have a 'time' field for chronological ordering,
        even if the caller doesn't provide one.
        """
        trade = {"symbol": "ETH-USD", "side": "sell", "quantity": 1.0}

        temp_store.append_trade(bot_id="bot-1", trade=trade)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert "time" in events[0]
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(events[0]["time"])

    def test_append_trade_preserves_provided_timestamp(self, temp_store: EventStore) -> None:
        """append_trade() preserves caller-provided timestamp.

        Allows backdating events for replay scenarios or when importing
        historical data.
        """
        custom_time = "2024-01-15T12:00:00"
        trade = {"symbol": "BTC-USD", "side": "buy", "quantity": 0.5, "time": custom_time}

        temp_store.append_trade(bot_id="bot-1", trade=trade)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert events[0]["time"] == custom_time


class TestEventStorePositionEvents:
    """Test position event recording."""

    def test_append_position_records_event(self, temp_store: EventStore) -> None:
        """append_position() successfully records a position snapshot.

        Position events capture the current state of holdings at a point
        in time, essential for P&L calculation and risk monitoring.
        """
        position = {"symbol": "BTC-USD", "size": 0.5, "entry_price": 50000.0, "pnl": 1250.0}

        temp_store.append_position(bot_id="bot-1", position=position)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "position"
        assert events[0]["bot_id"] == "bot-1"
        assert events[0]["size"] == 0.5

    def test_append_position_with_dataclass(self, temp_store: EventStore) -> None:
        """append_position() correctly serializes dataclass objects.

        Production code often uses dataclasses for type safety. The store
        must handle these correctly without requiring manual conversion.
        """
        position = SamplePosition(symbol="ETH-USD", size=2.0, entry_price=3000.0)

        temp_store.append_position(bot_id="bot-1", position={"position": position})
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert events[0]["position"]["symbol"] == "ETH-USD"
        assert events[0]["position"]["size"] == 2.0


class TestEventStoreMetricEvents:
    """Test metric event recording."""

    def test_append_metric_records_event(self, temp_store: EventStore) -> None:
        """append_metric() successfully records system metrics.

        Metrics provide operational visibility into bot performance,
        enabling monitoring dashboards and alerting.
        """
        metrics = {
            "latency_ms": 125.5,
            "success_rate": 0.98,
            "memory_mb": 512,
            "cpu_percent": 15.2,
        }

        temp_store.append_metric(bot_id="bot-1", metrics=metrics)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "metric"
        assert events[0]["latency_ms"] == 125.5
        assert events[0]["success_rate"] == 0.98


class TestEventStoreErrorEvents:
    """Test error event recording."""

    def test_append_error_records_event(self, temp_store: EventStore) -> None:
        """append_error() successfully records an error.

        Error events are critical for debugging production issues and
        alerting when bots encounter problems.
        """
        temp_store.append_error(
            bot_id="bot-1",
            message="Order rejected: insufficient funds",
            context={"order_id": "12345", "symbol": "BTC-USD"},
        )

        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["message"] == "Order rejected: insufficient funds"
        assert events[0]["order_id"] == "12345"

    def test_append_error_without_context(self, temp_store: EventStore) -> None:
        """append_error() works without optional context parameter.

        Ensures simple error logging without additional context is supported.
        """
        temp_store.append_error(bot_id="bot-1", message="Connection timeout")

        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["message"] == "Connection timeout"


class TestEventStoreQueryFiltering:
    """Test event querying and filtering."""

    def test_tail_returns_latest_events(self, temp_store: EventStore) -> None:
        """tail() returns the most recent events up to the limit.

        Critical for dashboards and monitoring that need to display
        recent activity without loading the entire history.
        """
        # Add 10 trades
        for i in range(10):
            temp_store.append_trade(bot_id="bot-1", trade={"index": i})

        events = temp_store.tail(bot_id="bot-1", limit=5)

        assert len(events) == 5
        # Should get the last 5 (indices 5-9)
        assert events[0]["index"] == 5
        assert events[-1]["index"] == 9

    def test_tail_filters_by_bot_id(self, temp_store: EventStore) -> None:
        """tail() only returns events for the specified bot_id.

        Essential for multi-bot systems where each bot needs to see
        only its own events, not other bots' activity.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})
        temp_store.append_trade(bot_id="bot-2", trade={"symbol": "ETH-USD"})
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "SOL-USD"})

        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert len(events) == 2
        assert all(e["bot_id"] == "bot-1" for e in events)

    def test_tail_filters_by_event_type(self, temp_store: EventStore) -> None:
        """tail() filters events by type when types parameter is provided.

        Allows focused queries like "show me only trades" or "show me
        only errors", reducing noise in analysis.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})
        temp_store.append_position(bot_id="bot-1", position={"size": 1.0})
        temp_store.append_error(bot_id="bot-1", message="Error 1")
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "ETH-USD"})

        events = temp_store.tail(bot_id="bot-1", limit=10, types=["trade"])

        assert len(events) == 2
        assert all(e["type"] == "trade" for e in events)

    def test_tail_handles_multiple_types(self, temp_store: EventStore) -> None:
        """tail() can filter by multiple event types simultaneously.

        Supports queries like "show me trades and errors but not metrics",
        useful for focused analysis.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})
        temp_store.append_position(bot_id="bot-1", position={"size": 1.0})
        temp_store.append_error(bot_id="bot-1", message="Error 1")
        temp_store.append_metric(bot_id="bot-1", metrics={"cpu": 50})

        events = temp_store.tail(bot_id="bot-1", limit=10, types=["trade", "error"])

        assert len(events) == 2
        assert {e["type"] for e in events} == {"trade", "error"}

    def test_tail_handles_corrupted_lines(self, temp_store: EventStore) -> None:
        """tail() skips corrupted JSON lines gracefully.

        Defensive behavior: If the file has corrupted lines (power failure,
        disk corruption), the query should skip bad lines and return valid
        events rather than crashing.
        """
        # Write a valid event, then corrupt line, then another valid event
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})

        with temp_store.path.open("a") as f:
            f.write("{invalid json line}\n")

        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "ETH-USD"})

        events = temp_store.tail(bot_id="bot-1", limit=10)

        # Should get 2 valid events, skipping the corrupted line
        assert len(events) == 2
        assert events[0]["symbol"] == "BTC-USD"
        assert events[1]["symbol"] == "ETH-USD"

    def test_tail_returns_empty_for_nonexistent_bot(self, temp_store: EventStore) -> None:
        """tail() returns empty list for bot_id with no events.

        Safe default: Querying a non-existent bot should return empty list,
        not raise an error.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})

        events = temp_store.tail(bot_id="nonexistent", limit=10)

        assert events == []


class TestEventStoreFileFormat:
    """Test JSONL file format correctness."""

    def test_writes_one_json_per_line(self, temp_store: EventStore) -> None:
        """Each event is written as a single line of JSON (JSONL format).

        JSONL format is critical for streaming processing and tools like
        grep, awk, and log aggregators that process line by line.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "BTC-USD"})
        temp_store.append_trade(bot_id="bot-1", trade={"symbol": "ETH-USD"})

        lines = temp_store.path.read_text().strip().split("\n")

        assert len(lines) == 2
        # Each line should be valid JSON
        event1 = json.loads(lines[0])
        event2 = json.loads(lines[1])
        assert event1["symbol"] == "BTC-USD"
        assert event2["symbol"] == "ETH-USD"

    def test_file_is_append_only(self, temp_store: EventStore) -> None:
        """New events are appended without modifying existing lines.

        Append-only ensures immutability - past events cannot be altered,
        maintaining audit trail integrity.
        """
        temp_store.append_trade(bot_id="bot-1", trade={"index": 1})
        first_content = temp_store.path.read_text()

        temp_store.append_trade(bot_id="bot-1", trade={"index": 2})
        second_content = temp_store.path.read_text()

        # New content should be strictly additive
        assert second_content.startswith(first_content)
        assert len(second_content) > len(first_content)


class TestEventStoreThreadSafety:
    """Test EventStore thread safety."""

    def test_concurrent_writes_preserve_all_events(self, temp_store: EventStore) -> None:
        """Multiple threads can write concurrently without data loss.

        Critical for production: Multiple bots or monitoring threads may
        write events simultaneously. All events must be recorded without
        corruption or loss.
        """

        def write_events(bot_id: str, count: int) -> None:
            for i in range(count):
                temp_store.append_trade(bot_id=bot_id, trade={"index": i})

        threads = [
            threading.Thread(target=write_events, args=("bot-1", 20)),
            threading.Thread(target=write_events, args=("bot-2", 20)),
            threading.Thread(target=write_events, args=("bot-3", 20)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All events should be present
        bot1_events = temp_store.tail(bot_id="bot-1", limit=100)
        bot2_events = temp_store.tail(bot_id="bot-2", limit=100)
        bot3_events = temp_store.tail(bot_id="bot-3", limit=100)

        assert len(bot1_events) == 20
        assert len(bot2_events) == 20
        assert len(bot3_events) == 20

    def test_concurrent_mixed_operations(self, temp_store: EventStore) -> None:
        """Concurrent writes of different event types work correctly.

        Ensures the locking mechanism protects all write operations
        (trades, positions, metrics, errors) uniformly.
        """
        errors: list[Exception] = []

        def write_trades() -> None:
            try:
                for i in range(10):
                    temp_store.append_trade(bot_id="bot-1", trade={"index": i})
            except Exception as e:
                errors.append(e)

        def write_positions() -> None:
            try:
                for i in range(10):
                    temp_store.append_position(bot_id="bot-1", position={"size": i})
            except Exception as e:
                errors.append(e)

        def write_errors() -> None:
            try:
                for i in range(10):
                    temp_store.append_error(bot_id="bot-1", message=f"Error {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_trades),
            threading.Thread(target=write_positions),
            threading.Thread(target=write_errors),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur
        assert len(errors) == 0

        # All events should be recorded
        all_events = temp_store.tail(bot_id="bot-1", limit=100)
        assert len(all_events) == 30

        trades = [e for e in all_events if e["type"] == "trade"]
        positions = [e for e in all_events if e["type"] == "position"]
        error_events = [e for e in all_events if e["type"] == "error"]

        assert len(trades) == 10
        assert len(positions) == 10
        assert len(error_events) == 10


class TestEventStoreDatetimeSerialization:
    """Test datetime object serialization."""

    def test_datetime_objects_are_serialized(self, temp_store: EventStore) -> None:
        """datetime objects in events are serialized to ISO format.

        Python datetime objects can't be directly JSON-serialized. The
        store must convert them automatically to ISO strings.
        """
        trade_time = datetime(2024, 1, 15, 12, 30, 45)
        trade = {"symbol": "BTC-USD", "executed_at": trade_time}

        temp_store.append_trade(bot_id="bot-1", trade=trade)
        events = temp_store.tail(bot_id="bot-1", limit=10)

        assert events[0]["executed_at"] == "2024-01-15T12:30:45"
