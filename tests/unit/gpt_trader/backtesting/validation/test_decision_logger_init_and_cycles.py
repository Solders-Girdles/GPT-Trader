"""Tests for DecisionLogger initialization and cycles."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gpt_trader.backtesting.validation.decision_logger import DecisionLogger


class TestDecisionLoggerInit:
    """Tests for DecisionLogger initialization."""

    def test_default_initialization(self) -> None:
        logger = DecisionLogger()

        assert logger.storage_path is None
        assert logger.max_memory_decisions == 10000
        assert logger.decision_count == 0

    def test_with_storage_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(storage_path=tmpdir)

            assert logger.storage_path == Path(tmpdir)
            assert logger.storage_path.exists()

    def test_with_custom_max_memory(self) -> None:
        logger = DecisionLogger(max_memory_decisions=1000)

        assert logger.max_memory_decisions == 1000

    def test_creates_storage_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "decisions" / "logs"
            logger = DecisionLogger(storage_path=new_path)

            assert logger.storage_path.exists()


class TestDecisionLoggerCycles:
    """Tests for DecisionLogger cycle management."""

    def test_start_cycle_generates_id(self) -> None:
        logger = DecisionLogger()

        cycle_id = logger.start_cycle()

        assert cycle_id is not None
        assert len(cycle_id) == 12
        assert logger.current_cycle_id == cycle_id

    def test_start_cycle_uses_provided_id(self) -> None:
        logger = DecisionLogger()

        cycle_id = logger.start_cycle("my-custom-cycle")

        assert cycle_id == "my-custom-cycle"
        assert logger.current_cycle_id == "my-custom-cycle"
