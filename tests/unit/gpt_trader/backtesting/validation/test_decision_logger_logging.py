"""Tests for DecisionLogger logging and retrieval APIs."""

from __future__ import annotations

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gpt_trader.backtesting.validation.decision_logger import (
    DecisionLogger,
    StrategyDecision,
)

DEFAULT_EQUITY = Decimal("100000")
DEFAULT_MARK_PRICE = Decimal("50000")
DEFAULT_TIMESTAMP = datetime(2024, 1, 1, 12, 0, 0)


def make_decision(
    decision_id: str,
    *,
    cycle_id: str = "cycle-001",
    timestamp: datetime = DEFAULT_TIMESTAMP,
    symbol: str = "BTC-USD",
) -> StrategyDecision:
    return StrategyDecision(
        decision_id=decision_id,
        cycle_id=cycle_id,
        timestamp=timestamp,
        symbol=symbol,
        equity=DEFAULT_EQUITY,
        position_quantity=Decimal("0"),
        position_side=None,
        mark_price=DEFAULT_MARK_PRICE,
        recent_marks=[DEFAULT_MARK_PRICE],
    )


class TestDecisionLoggerLogDecision:
    """Tests for DecisionLogger.log_decision()."""

    def test_log_decision_increments_count(self) -> None:
        logger = DecisionLogger()
        decision = make_decision("dec-001")

        logger.log_decision(decision)

        assert logger.decision_count == 1

    def test_log_multiple_decisions(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = make_decision(
                f"dec-{i:03d}",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
            )
            logger.log_decision(decision)

        assert logger.decision_count == 5

    def test_log_decision_trims_when_over_limit(self) -> None:
        logger = DecisionLogger(max_memory_decisions=5)

        for i in range(10):
            decision = make_decision(
                f"dec-{i:03d}",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
            )
            logger.log_decision(decision)

        assert logger.decision_count == 5
        decisions = logger.get_decisions()
        assert decisions[0].decision_id == "dec-005"
        assert decisions[-1].decision_id == "dec-009"

    def test_log_decision_persists_to_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(storage_path=tmpdir)
            decision = make_decision("dec-001")

            logger.log_decision(decision)

            files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(files) == 1
            assert "20240101_cycle-001" in files[0].name


class TestDecisionLoggerGetDecisions:
    """Tests for DecisionLogger.get_decisions()."""

    def test_get_all_decisions(self) -> None:
        logger = DecisionLogger()

        for i in range(3):
            decision = make_decision(
                f"dec-{i:03d}",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions()

        assert len(decisions) == 3

    def test_filter_by_cycle_id(self) -> None:
        logger = DecisionLogger()

        for cycle in ["cycle-001", "cycle-001", "cycle-002"]:
            decision = make_decision(f"dec-{cycle}", cycle_id=cycle)
            logger.log_decision(decision)

        decisions = logger.get_decisions(cycle_id="cycle-001")

        assert len(decisions) == 2
        assert all(d.cycle_id == "cycle-001" for d in decisions)

    def test_filter_by_symbol(self) -> None:
        logger = DecisionLogger()

        for symbol in ["BTC-USD", "BTC-USD", "ETH-USD"]:
            decision = make_decision(f"dec-{symbol}", symbol=symbol)
            logger.log_decision(decision)

        decisions = logger.get_decisions(symbol="ETH-USD")

        assert len(decisions) == 1
        assert decisions[0].symbol == "ETH-USD"

    def test_filter_by_start_time(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = make_decision(
                f"dec-{i:03d}",
                timestamp=datetime(2024, 1, 1, i, 0, 0),
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(start_time=datetime(2024, 1, 1, 3, 0, 0))

        assert len(decisions) == 2
        assert all(d.timestamp >= datetime(2024, 1, 1, 3, 0, 0) for d in decisions)

    def test_filter_by_end_time(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = make_decision(
                f"dec-{i:03d}",
                timestamp=datetime(2024, 1, 1, i, 0, 0),
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(end_time=datetime(2024, 1, 1, 3, 0, 0))

        assert len(decisions) == 3
        assert all(d.timestamp < datetime(2024, 1, 1, 3, 0, 0) for d in decisions)


class TestDecisionLoggerGetDecisionById:
    """Tests for DecisionLogger.get_decision_by_id()."""

    def test_get_existing_decision(self) -> None:
        logger = DecisionLogger()
        decision = make_decision("target-dec")
        logger.log_decision(decision)

        result = logger.get_decision_by_id("target-dec")

        assert result is not None
        assert result.decision_id == "target-dec"

    def test_get_nonexistent_decision(self) -> None:
        logger = DecisionLogger()

        result = logger.get_decision_by_id("nonexistent")

        assert result is None
