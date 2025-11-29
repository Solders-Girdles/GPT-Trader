"""Integration tests for DatabaseEngine SQLite functionality."""

import tempfile
from pathlib import Path

from gpt_trader.persistence.database import DatabaseEngine


class TestDatabaseEngine:
    """Test DatabaseEngine directly."""

    def test_initialize_creates_database(self) -> None:
        """Test that initialize creates the database file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "events.db"
            engine = DatabaseEngine(database_path)
            engine.initialize()

            assert database_path.exists()
            engine.close()

    def test_write_and_read_event(self) -> None:
        """Test writing and reading events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "events.db"
            engine = DatabaseEngine(database_path)
            engine.initialize()

            engine.write_event("trade", {"symbol": "BTC-USD"}, "bot1")
            events = engine.read_all_events()

            assert len(events) == 1
            assert events[0]["type"] == "trade"
            assert events[0]["data"]["symbol"] == "BTC-USD"
            engine.close()

    def test_read_recent_events(self) -> None:
        """Test reading recent events."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "events.db"
            engine = DatabaseEngine(database_path)
            engine.initialize()

            for i in range(10):
                engine.write_event("event", {"index": i}, None)

            recent = engine.read_recent_events(3)
            assert len(recent) == 3
            # Should be in chronological order (7, 8, 9)
            assert recent[0]["data"]["index"] == 7
            assert recent[1]["data"]["index"] == 8
            assert recent[2]["data"]["index"] == 9
            engine.close()

    def test_read_events_by_bot(self) -> None:
        """Test reading events filtered by bot_id."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "events.db"
            engine = DatabaseEngine(database_path)
            engine.initialize()

            engine.write_event("trade", {"amount": 100}, "bot1")
            engine.write_event("trade", {"amount": 200}, "bot2")
            engine.write_event("trade", {"amount": 300}, "bot1")

            bot1_events = engine.read_events_by_bot("bot1")
            assert len(bot1_events) == 2
            assert bot1_events[0]["data"]["amount"] == 100
            assert bot1_events[1]["data"]["amount"] == 300
            engine.close()
