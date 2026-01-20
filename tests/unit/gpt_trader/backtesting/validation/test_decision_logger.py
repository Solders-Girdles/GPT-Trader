"""Tests for DecisionLogger initialization, cycles, clearing, export/import, and properties."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from gpt_trader.backtesting.validation.decision_logger import (
    DecisionLogger,
    StrategyDecision,
)


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


class TestDecisionLoggerClear:
    """Tests for DecisionLogger.clear()."""

    def test_clear_removes_all_decisions(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = StrategyDecision(
                decision_id=f"dec-{i:03d}",
                cycle_id="cycle-001",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                symbol="BTC-USD",
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        assert logger.decision_count == 5

        logger.clear()

        assert logger.decision_count == 0


class TestDecisionLoggerExportImport:
    """Tests for DecisionLogger export/import functionality."""

    def test_export_to_json(self) -> None:
        logger = DecisionLogger()

        for i in range(3):
            decision = StrategyDecision(
                decision_id=f"dec-{i:03d}",
                cycle_id="cycle-001",
                timestamp=datetime(2024, 1, 1, 12, i, 0),
                symbol="BTC-USD",
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        count = logger.export_to_json(filepath)

        assert count == 3

        with open(filepath) as f:
            data = json.load(f)
            assert len(data) == 3

    def test_import_from_json(self) -> None:
        data = [
            {
                "decision_id": "dec-001",
                "cycle_id": "cycle-001",
                "timestamp": "2024-01-01T12:00:00",
                "symbol": "BTC-USD",
                "equity": "100000",
                "position_quantity": "0",
                "position_side": None,
                "mark_price": "50000",
                "recent_marks": ["50000"],
                "action": "HOLD",
                "target_quantity": "0",
                "target_price": None,
                "order_type": "MARKET",
                "reason": "",
                "risk_checks_passed": True,
                "risk_check_failures": [],
                "strategy_name": "",
                "strategy_params": {},
            },
            {
                "decision_id": "dec-002",
                "cycle_id": "cycle-001",
                "timestamp": "2024-01-01T12:01:00",
                "symbol": "BTC-USD",
                "equity": "100000",
                "position_quantity": "0",
                "position_side": None,
                "mark_price": "50000",
                "recent_marks": ["50000"],
                "action": "BUY",
                "target_quantity": "1.0",
                "target_price": "50000",
                "order_type": "MARKET",
                "reason": "Signal",
                "risk_checks_passed": True,
                "risk_check_failures": [],
                "strategy_name": "",
                "strategy_params": {},
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            filepath = f.name

        logger = DecisionLogger()
        count = logger.import_from_json(filepath)

        assert count == 2
        assert logger.decision_count == 2
        decisions = logger.get_decisions()
        assert decisions[0].decision_id == "dec-001"
        assert decisions[1].action == "BUY"


class TestDecisionLoggerProperties:
    """Tests for DecisionLogger properties."""

    def test_current_cycle_id_initially_none(self) -> None:
        logger = DecisionLogger()

        assert logger.current_cycle_id is None

    def test_current_cycle_id_after_start(self) -> None:
        logger = DecisionLogger()
        logger.start_cycle("test-cycle")

        assert logger.current_cycle_id == "test-cycle"

    def test_decision_count_reflects_logged_decisions(self) -> None:
        logger = DecisionLogger()

        assert logger.decision_count == 0

        decision = StrategyDecision(
            decision_id="dec-001",
            cycle_id="cycle-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )
        logger.log_decision(decision)

        assert logger.decision_count == 1
