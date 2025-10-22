"""Tests for decision logger."""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.features.optimize.decision_logger import (
    DecisionLogger,
    compare_decision_logs,
    load_decision_log,
)
from bot_v2.features.optimize.types import BacktestMetrics
from bot_v2.features.optimize.types_v2 import (
    BacktestConfig,
    BacktestResult,
    DecisionContext,
    ExecutionResult,
)


@pytest.fixture
def sample_context():
    """Create a sample decision context."""
    return DecisionContext(
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        symbol="BTC-USD",
        current_mark=Decimal("42350.50"),
        recent_marks=[Decimal("42320.10"), Decimal("42340.25")],
        position_state=None,
        equity=Decimal("10500.00"),
        signal_label="bullish",
    )


@pytest.fixture
def sample_decision():
    """Create a sample decision."""
    return Decision(
        action=Action.BUY,
        quantity=Decimal("0.1"),
        target_notional=Decimal("4235.05"),
        reason="Bullish MA crossover",
    )


@pytest.fixture
def sample_execution():
    """Create a sample execution result."""
    return ExecutionResult(
        filled=True,
        fill_price=Decimal("42352.00"),
        filled_quantity=Decimal("0.1"),
        commission=Decimal("4.24"),
        slippage=Decimal("1.50"),
    )


def test_decision_logger_initialization():
    """Test DecisionLogger initialization."""
    logger = DecisionLogger(enabled=True, base_directory="test_logs")

    assert logger.enabled is True
    assert logger.base_directory == Path("test_logs")
    assert len(logger.decisions) == 0


def test_decision_logger_disabled():
    """Test that disabled logger doesn't log."""
    logger = DecisionLogger(enabled=False)
    context = DecisionContext(
        timestamp=datetime.now(),
        symbol="BTC-USD",
        current_mark=Decimal("100"),
        recent_marks=[],
        position_state=None,
        equity=Decimal("10000"),
    )
    decision = Decision(action=Action.HOLD, reason="Test")
    execution = ExecutionResult(filled=False)

    logger.log_decision(context=context, decision=decision, execution=execution)

    # Should not log when disabled
    assert len(logger.decisions) == 0


def test_decision_logger_logs_decision(sample_context, sample_decision, sample_execution):
    """Test logging a decision."""
    logger = DecisionLogger(enabled=True)

    logger.log_decision(context=sample_context, decision=sample_decision, execution=sample_execution)

    assert logger.get_decision_count() == 1
    assert logger.get_execution_count() == 1

    record = logger.decisions[0]
    assert record.context.symbol == "BTC-USD"
    assert record.decision.action == Action.BUY
    assert record.execution.filled is True


def test_decision_logger_save_and_load(tmp_path, sample_context, sample_decision, sample_execution):
    """Test saving and loading decision logs."""
    logger = DecisionLogger(enabled=True, base_directory=str(tmp_path))

    # Log a decision
    logger.log_decision(context=sample_context, decision=sample_decision, execution=sample_execution)

    # Create a backtest result
    result = BacktestResult(
        run_id="test_run_123",
        strategy_name="BaselinePerpsStrategy",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 31),
        config=BacktestConfig(),
        decisions=logger.decisions,
        metrics=BacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.6,
            profit_factor=1.5,
            total_trades=10,
            avg_trade=0.015,
            best_trade=0.05,
            worst_trade=-0.02,
            recovery_factor=3.0,
            calmar_ratio=3.0,
        ),
        equity_curve=[(datetime(2024, 1, 1), Decimal("10000"))],
    )

    # Save
    filepath = logger.save(result)
    assert filepath.exists()

    # Load
    loaded_result = logger.load(filepath)

    assert loaded_result.run_id == "test_run_123"
    assert loaded_result.symbol == "BTC-USD"
    assert len(loaded_result.decisions) == 1
    assert loaded_result.metrics.total_return == 0.15


def test_decision_context_serialization(sample_context):
    """Test DecisionContext serialization/deserialization."""
    # Serialize
    data = sample_context.to_dict()

    assert data["symbol"] == "BTC-USD"
    assert data["current_mark"] == "42350.50"
    assert data["equity"] == "10500.00"

    # Deserialize
    restored = DecisionContext.from_dict(data)

    assert restored.symbol == sample_context.symbol
    assert restored.current_mark == sample_context.current_mark
    assert restored.equity == sample_context.equity


def test_execution_result_serialization(sample_execution):
    """Test ExecutionResult serialization/deserialization."""
    # Serialize
    data = sample_execution.to_dict()

    assert data["filled"] is True
    assert data["fill_price"] == "42352.00"
    assert data["commission"] == "4.24"

    # Deserialize
    restored = ExecutionResult.from_dict(data)

    assert restored.filled == sample_execution.filled
    assert restored.fill_price == sample_execution.fill_price
    assert restored.commission == sample_execution.commission


def test_decision_logger_clear():
    """Test clearing decision log."""
    logger = DecisionLogger(enabled=True)

    # Add some decisions
    for i in range(5):
        context = DecisionContext(
            timestamp=datetime.now(),
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            recent_marks=[],
            position_state=None,
            equity=Decimal("10000"),
        )
        decision = Decision(action=Action.HOLD, reason="Test")
        execution = ExecutionResult(filled=False)
        logger.log_decision(context=context, decision=decision, execution=execution)

    assert logger.get_decision_count() == 5

    # Clear
    logger.clear()

    assert logger.get_decision_count() == 0


def test_compare_decision_logs_identical(tmp_path):
    """Test comparing identical decision logs."""
    logger1 = DecisionLogger(enabled=True, base_directory=str(tmp_path / "log1"))
    logger2 = DecisionLogger(enabled=True, base_directory=str(tmp_path / "log2"))

    # Create identical decisions
    for i in range(3):
        context = DecisionContext(
            timestamp=datetime(2024, 1, 1, i, 0, 0),
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            recent_marks=[],
            position_state=None,
            equity=Decimal("10000"),
        )
        decision = Decision(action=Action.BUY, reason="Test")
        execution = ExecutionResult(filled=True)

        logger1.log_decision(context=context, decision=decision, execution=execution)
        logger2.log_decision(context=context, decision=decision, execution=execution)

    # Save both
    result1 = BacktestResult(
        run_id="test1",
        strategy_name="Test",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        config=BacktestConfig(),
        decisions=logger1.decisions,
    )
    result2 = BacktestResult(
        run_id="test2",
        strategy_name="Test",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        config=BacktestConfig(),
        decisions=logger2.decisions,
    )

    path1 = logger1.save(result1)
    path2 = logger2.save(result2)

    # Compare
    comparison = compare_decision_logs(path1, path2)

    assert comparison["decision_count_match"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["parity_rate"] == 1.0


def test_compare_decision_logs_with_differences(tmp_path):
    """Test comparing decision logs with differences."""
    logger1 = DecisionLogger(enabled=True, base_directory=str(tmp_path / "log1"))
    logger2 = DecisionLogger(enabled=True, base_directory=str(tmp_path / "log2"))

    # Create different decisions
    for i in range(3):
        context = DecisionContext(
            timestamp=datetime(2024, 1, 1, i, 0, 0),
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            recent_marks=[],
            position_state=None,
            equity=Decimal("10000"),
        )

        # Different actions
        decision1 = Decision(action=Action.BUY, reason="Test")
        decision2 = Decision(action=Action.HOLD, reason="Different")
        execution = ExecutionResult(filled=False)

        logger1.log_decision(context=context, decision=decision1, execution=execution)
        logger2.log_decision(context=context, decision=decision2, execution=execution)

    # Save both
    result1 = BacktestResult(
        run_id="test1",
        strategy_name="Test",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        config=BacktestConfig(),
        decisions=logger1.decisions,
    )
    result2 = BacktestResult(
        run_id="test2",
        strategy_name="Test",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        config=BacktestConfig(),
        decisions=logger2.decisions,
    )

    path1 = logger1.save(result1)
    path2 = logger2.save(result2)

    # Compare
    comparison = compare_decision_logs(path1, path2)

    assert comparison["decision_count_match"] is True
    assert comparison["mismatch_count"] == 3  # All 3 differ
    assert comparison["parity_rate"] == 0.0  # 0% match


def test_load_decision_log_convenience_function(tmp_path, sample_context, sample_decision, sample_execution):
    """Test the load_decision_log convenience function."""
    logger = DecisionLogger(enabled=True, base_directory=str(tmp_path))
    logger.log_decision(context=sample_context, decision=sample_decision, execution=sample_execution)

    result = BacktestResult(
        run_id="test",
        strategy_name="Test",
        symbol="BTC-USD",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        config=BacktestConfig(),
        decisions=logger.decisions,
    )

    filepath = logger.save(result)

    # Load using convenience function
    loaded = load_decision_log(filepath)

    assert loaded.run_id == "test"
    assert len(loaded.decisions) == 1
