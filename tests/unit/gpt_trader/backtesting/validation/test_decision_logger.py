"""Tests for decision logging module."""

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
from gpt_trader.utilities.datetime_helpers import utc_now


class TestStrategyDecisionCreate:
    """Tests for StrategyDecision.create() class method."""

    def test_create_generates_decision_id(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )
        assert decision.decision_id is not None
        assert len(decision.decision_id) == 12

    def test_create_sets_timestamp(self) -> None:
        before = utc_now()
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )
        after = utc_now()
        assert before <= decision.timestamp <= after

    def test_create_preserves_parameters(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="ETH-USD",
            equity=Decimal("50000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("3000"),
            recent_marks=[Decimal("2990"), Decimal("3000"), Decimal("3010")],
        )
        assert decision.cycle_id == "cycle-001"
        assert decision.symbol == "ETH-USD"
        assert decision.equity == Decimal("50000")
        assert decision.position_quantity == Decimal("1.5")
        assert decision.position_side == "long"
        assert decision.mark_price == Decimal("3000")
        assert len(decision.recent_marks) == 3


class TestStrategyDecisionBuilderMethods:
    """Tests for StrategyDecision builder methods."""

    def test_with_market_data(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_market_data(
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            volume=Decimal("1000"),
        )

        assert result is decision  # Returns self
        assert decision.bid == Decimal("49990")
        assert decision.ask == Decimal("50010")
        assert decision.volume == Decimal("1000")

    def test_with_market_data_partial(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_market_data(bid=Decimal("49990"))

        assert decision.bid == Decimal("49990")
        assert decision.ask is None

    def test_with_strategy(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_strategy(
            name="momentum_strategy",
            params={"lookback": 20, "threshold": 0.02},
        )

        assert result is decision
        assert decision.strategy_name == "momentum_strategy"
        assert decision.strategy_params == {"lookback": 20, "threshold": 0.02}

    def test_with_strategy_no_params(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_strategy(name="simple_strategy")

        assert decision.strategy_name == "simple_strategy"
        assert decision.strategy_params == {}

    def test_with_action(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_action(
            action="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            order_type="LIMIT",
            reason="Signal triggered",
        )

        assert result is decision
        assert decision.action == "BUY"
        assert decision.target_quantity == Decimal("1.0")
        assert decision.target_price == Decimal("50000")
        assert decision.order_type == "LIMIT"
        assert decision.reason == "Signal triggered"

    def test_with_action_defaults(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_action(action="HOLD")

        assert decision.action == "HOLD"
        assert decision.target_quantity == Decimal("0")
        assert decision.target_price is None
        assert decision.order_type == "MARKET"

    def test_with_risk_result_passed(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_risk_result(passed=True)

        assert result is decision
        assert decision.risk_checks_passed is True
        assert decision.risk_check_failures == []

    def test_with_risk_result_failed(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_risk_result(
            passed=False,
            failures=["Position too large", "Daily loss limit exceeded"],
        )

        assert decision.risk_checks_passed is False
        assert len(decision.risk_check_failures) == 2
        assert "Position too large" in decision.risk_check_failures

    def test_with_execution(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_execution(
            order_id="order-12345",
            fill_price=Decimal("50005"),
            fill_quantity=Decimal("0.98"),
            slippage_bps=Decimal("1.0"),
        )

        assert result is decision
        assert decision.order_id == "order-12345"
        assert decision.fill_price == Decimal("50005")
        assert decision.fill_quantity == Decimal("0.98")
        assert decision.slippage_bps == Decimal("1.0")


class TestStrategyDecisionSerialization:
    """Tests for StrategyDecision serialization."""

    def test_to_dict_converts_decimals(self) -> None:
        decision = StrategyDecision(
            decision_id="dec-001",
            cycle_id="cycle-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("49900"), Decimal("50000")],
        )

        data = decision.to_dict()

        assert data["equity"] == "100000"
        assert data["mark_price"] == "50000"
        assert data["recent_marks"] == ["49900", "50000"]

    def test_to_dict_converts_datetime(self) -> None:
        ts = datetime(2024, 1, 1, 12, 30, 45)
        decision = StrategyDecision(
            decision_id="dec-001",
            cycle_id="cycle-001",
            timestamp=ts,
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        data = decision.to_dict()

        assert data["timestamp"] == "2024-01-01T12:30:45"

    def test_from_dict_restores_decimals(self) -> None:
        data = {
            "decision_id": "dec-001",
            "cycle_id": "cycle-001",
            "timestamp": "2024-01-01T12:00:00",
            "symbol": "BTC-USD",
            "equity": "100000",
            "position_quantity": "1.5",
            "position_side": "long",
            "mark_price": "50000",
            "recent_marks": ["49900", "50000"],
            "action": "BUY",
            "target_quantity": "1.0",
            "target_price": "50000",
            "order_type": "MARKET",
            "reason": "",
            "risk_checks_passed": True,
            "risk_check_failures": [],
            "strategy_name": "",
            "strategy_params": {},
        }

        decision = StrategyDecision.from_dict(data)

        assert decision.equity == Decimal("100000")
        assert decision.position_quantity == Decimal("1.5")
        assert decision.mark_price == Decimal("50000")
        assert decision.recent_marks == [Decimal("49900"), Decimal("50000")]

    def test_from_dict_restores_timestamp(self) -> None:
        data = {
            "decision_id": "dec-001",
            "cycle_id": "cycle-001",
            "timestamp": "2024-01-01T12:30:45",
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
        }

        decision = StrategyDecision.from_dict(data)

        assert decision.timestamp == datetime(2024, 1, 1, 12, 30, 45)

    def test_roundtrip_serialization(self) -> None:
        original = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("49900"), Decimal("50000")],
        )
        original.with_action("BUY", Decimal("2.0"), Decimal("50100"))
        original.with_market_data(Decimal("50095"), Decimal("50105"))

        data = original.to_dict()
        restored = StrategyDecision.from_dict(data)

        assert restored.decision_id == original.decision_id
        assert restored.symbol == original.symbol
        assert restored.equity == original.equity
        assert restored.action == original.action
        assert restored.target_quantity == original.target_quantity


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


class TestDecisionLoggerLogDecision:
    """Tests for DecisionLogger.log_decision()."""

    def test_log_decision_increments_count(self) -> None:
        logger = DecisionLogger()
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

    def test_log_multiple_decisions(self) -> None:
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

    def test_log_decision_trims_when_over_limit(self) -> None:
        logger = DecisionLogger(max_memory_decisions=5)

        for i in range(10):
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
        # Should keep the most recent 5 (dec-005 through dec-009)
        decisions = logger.get_decisions()
        assert decisions[0].decision_id == "dec-005"
        assert decisions[-1].decision_id == "dec-009"

    def test_log_decision_persists_to_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = DecisionLogger(storage_path=tmpdir)
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

            # Check file was created
            files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(files) == 1
            assert "20240101_cycle-001" in files[0].name


class TestDecisionLoggerGetDecisions:
    """Tests for DecisionLogger.get_decisions()."""

    def test_get_all_decisions(self) -> None:
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

        decisions = logger.get_decisions()

        assert len(decisions) == 3

    def test_filter_by_cycle_id(self) -> None:
        logger = DecisionLogger()

        # Add decisions for different cycles
        for cycle in ["cycle-001", "cycle-001", "cycle-002"]:
            decision = StrategyDecision(
                decision_id=f"dec-{cycle}",
                cycle_id=cycle,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                symbol="BTC-USD",
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(cycle_id="cycle-001")

        assert len(decisions) == 2
        assert all(d.cycle_id == "cycle-001" for d in decisions)

    def test_filter_by_symbol(self) -> None:
        logger = DecisionLogger()

        for symbol in ["BTC-USD", "BTC-USD", "ETH-USD"]:
            decision = StrategyDecision(
                decision_id=f"dec-{symbol}",
                cycle_id="cycle-001",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                symbol=symbol,
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(symbol="ETH-USD")

        assert len(decisions) == 1
        assert decisions[0].symbol == "ETH-USD"

    def test_filter_by_start_time(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = StrategyDecision(
                decision_id=f"dec-{i:03d}",
                cycle_id="cycle-001",
                timestamp=datetime(2024, 1, 1, i, 0, 0),
                symbol="BTC-USD",
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(start_time=datetime(2024, 1, 1, 3, 0, 0))

        assert len(decisions) == 2
        assert all(d.timestamp >= datetime(2024, 1, 1, 3, 0, 0) for d in decisions)

    def test_filter_by_end_time(self) -> None:
        logger = DecisionLogger()

        for i in range(5):
            decision = StrategyDecision(
                decision_id=f"dec-{i:03d}",
                cycle_id="cycle-001",
                timestamp=datetime(2024, 1, 1, i, 0, 0),
                symbol="BTC-USD",
                equity=Decimal("100000"),
                position_quantity=Decimal("0"),
                position_side=None,
                mark_price=Decimal("50000"),
                recent_marks=[Decimal("50000")],
            )
            logger.log_decision(decision)

        decisions = logger.get_decisions(end_time=datetime(2024, 1, 1, 3, 0, 0))

        assert len(decisions) == 3
        assert all(d.timestamp < datetime(2024, 1, 1, 3, 0, 0) for d in decisions)


class TestDecisionLoggerGetDecisionById:
    """Tests for DecisionLogger.get_decision_by_id()."""

    def test_get_existing_decision(self) -> None:
        logger = DecisionLogger()
        decision = StrategyDecision(
            decision_id="target-dec",
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

        result = logger.get_decision_by_id("target-dec")

        assert result is not None
        assert result.decision_id == "target-dec"

    def test_get_nonexistent_decision(self) -> None:
        logger = DecisionLogger()

        result = logger.get_decision_by_id("nonexistent")

        assert result is None


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

        # Verify file content
        with open(filepath) as f:
            data = json.load(f)
            assert len(data) == 3

    def test_import_from_json(self) -> None:
        # Create export file
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
