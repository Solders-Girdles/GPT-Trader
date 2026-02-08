"""Unit tests for DatabaseEngine SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from gpt_trader.persistence.database import DatabaseEngine


def test_initialize_creates_database(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    engine = DatabaseEngine(database_path)
    try:
        engine.initialize()
        assert database_path.exists()
    finally:
        engine.close()


def test_write_and_read_event(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    engine = DatabaseEngine(database_path)
    try:
        engine.initialize()

        engine.write_event("trade", {"symbol": "BTC-USD"}, "bot1")
        events = engine.read_all_events()

        assert len(events) == 1
        assert events[0]["type"] == "trade"
        assert events[0]["data"]["symbol"] == "BTC-USD"
    finally:
        engine.close()


def test_read_recent_events(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    engine = DatabaseEngine(database_path)
    try:
        engine.initialize()

        for i in range(10):
            engine.write_event("event", {"index": i}, None)

        recent = engine.read_recent_events(3)
        assert len(recent) == 3
        assert [e["data"]["index"] for e in recent] == [7, 8, 9]
    finally:
        engine.close()


def test_read_events_by_symbol(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    engine = DatabaseEngine(database_path)
    try:
        engine.initialize()

        engine.write_event("price_tick", {"symbol": "BTC-USD", "price": "100"}, None)
        engine.write_event("price_tick", {"symbol": "ETH-USD", "price": "50"}, None)
        engine.write_event("price_tick", {"symbol": "BTC-USD", "price": "101"}, None)
        engine.write_event("trade", {"amount": 5}, None)  # no symbol

        btc_events = engine.read_events_by_symbol("BTC-USD")
        assert len(btc_events) == 2
        assert btc_events[0]["data"]["price"] == "100"
        assert btc_events[1]["data"]["price"] == "101"

        eth_events = engine.read_events_by_symbol("ETH-USD")
        assert len(eth_events) == 1
        assert eth_events[0]["data"]["price"] == "50"

        none_events = engine.read_events_by_symbol("SOL-USD")
        assert len(none_events) == 0
    finally:
        engine.close()


def test_read_events_by_bot(tmp_path: Path) -> None:
    database_path = tmp_path / "events.db"
    engine = DatabaseEngine(database_path)
    try:
        engine.initialize()

        engine.write_event("trade", {"amount": 100}, "bot1")
        engine.write_event("trade", {"amount": 200}, "bot2")
        engine.write_event("trade", {"amount": 300}, "bot1")

        bot1_events = engine.read_events_by_bot("bot1")
        assert [e["data"]["amount"] for e in bot1_events] == [100, 300]
    finally:
        engine.close()
