"""Tests for experiment tracking module."""

from datetime import datetime

import pytest

from gpt_trader.features.strategy_dev.lab.experiment import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
)


class TestExperimentResult:
    """Tests for ExperimentResult."""

    def test_create_result(self):
        """Test creating an experiment result."""
        result = ExperimentResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            average_win=50.0,
            average_loss=40.0,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 1),
            duration_seconds=86400 * 152,
        )

        assert result.total_return == 0.15
        assert result.sharpe_ratio == 1.5
        assert result.total_trades == 100

    def test_to_dict(self):
        """Test serialization to dict."""
        result = ExperimentResult(
            total_return=0.10,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.60,
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            average_win=100.0,
            average_loss=75.0,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 3, 1),
            duration_seconds=86400 * 60,
        )

        data = result.to_dict()

        assert data["metrics"]["total_return"] == 0.10
        assert data["trades"]["total"] == 50

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "metrics": {
                "total_return": 0.20,
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.10,
                "win_rate": 0.65,
            },
            "trades": {
                "total": 80,
                "winning": 52,
                "losing": 28,
                "average_win": 120.0,
                "average_loss": 90.0,
            },
            "timing": {
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-06-01T00:00:00",
                "duration_seconds": 13132800,
            },
        }

        result = ExperimentResult.from_dict(data)

        assert result.total_return == 0.20
        assert result.total_trades == 80

    def test_compare_to(self):
        """Test comparing two results."""
        result1 = ExperimentResult(
            total_return=0.20,
            sharpe_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.60,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            average_win=100.0,
            average_loss=75.0,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 1),
            duration_seconds=86400,
        )

        result2 = ExperimentResult(
            total_return=0.10,
            sharpe_ratio=1.0,
            max_drawdown=0.15,
            win_rate=0.55,
            total_trades=80,
            winning_trades=44,
            losing_trades=36,
            average_win=90.0,
            average_loss=70.0,
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 6, 1),
            duration_seconds=86400,
        )

        comparison = result1.compare_to(result2)

        # result1 has better return (100% improvement from 0.10 to 0.20)
        assert comparison["return_improvement_percent"] == pytest.approx(100.0)
        # result1 has better sharpe (100% improvement from 1.0 to 2.0)
        assert comparison["sharpe_improvement_percent"] == pytest.approx(100.0)


class TestExperiment:
    """Tests for Experiment."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        exp = Experiment(
            name="test_experiment",
            description="Testing parameters",
            strategy_name="momentum",
            parameters={"period": 20, "threshold": 0.6},
        )

        assert exp.name == "test_experiment"
        assert exp.strategy_name == "momentum"
        assert exp.status == ExperimentStatus.PENDING
        assert exp.parameters["period"] == 20

    def test_parameter_hash(self):
        """Test parameter hash generation."""
        exp1 = Experiment(
            name="exp1",
            description="",
            strategy_name="test",
            parameters={"a": 1, "b": 2},
        )

        exp2 = Experiment(
            name="exp2",
            description="Different",
            strategy_name="test",
            parameters={"a": 1, "b": 2},
        )

        exp3 = Experiment(
            name="exp3",
            description="",
            strategy_name="test",
            parameters={"a": 1, "b": 3},
        )

        # Same parameters should have same hash
        assert exp1.parameter_hash == exp2.parameter_hash
        # Different parameters should have different hash
        assert exp1.parameter_hash != exp3.parameter_hash

    def test_lifecycle(self):
        """Test experiment lifecycle."""
        exp = Experiment(
            name="lifecycle_test",
            description="",
            strategy_name="test",
            parameters={},
        )

        assert exp.status == ExperimentStatus.PENDING

        # Start
        exp.start()
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

        # Complete
        result = ExperimentResult(
            total_return=0.10,
            sharpe_ratio=1.0,
            max_drawdown=0.05,
            win_rate=0.50,
            total_trades=10,
            winning_trades=5,
            losing_trades=5,
            average_win=100.0,
            average_loss=100.0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=100.0,
        )

        exp.complete(result)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.completed_at is not None
        assert exp.result is not None

    def test_fail(self):
        """Test failing an experiment."""
        exp = Experiment(
            name="fail_test",
            description="",
            strategy_name="test",
            parameters={},
        )

        exp.start()
        exp.fail("Test error")

        assert exp.status == ExperimentStatus.FAILED
        assert exp.result.error_message == "Test error"

    def test_clone(self):
        """Test cloning an experiment."""
        exp = Experiment(
            name="original",
            description="Original experiment",
            strategy_name="test",
            parameters={"a": 1},
            tags=["tag1"],
        )

        cloned = exp.clone({"a": 2, "b": 3})

        assert cloned.name == "original_v2"
        assert cloned.parameters == {"a": 2, "b": 3}
        assert cloned.version == 2
        assert cloned.parent_experiment_id == exp.experiment_id

    def test_to_from_dict(self):
        """Test serialization round-trip."""
        exp = Experiment(
            name="serialize_test",
            description="Test serialization",
            strategy_name="test_strategy",
            parameters={"x": 10, "y": 20},
            tags=["test", "serialization"],
        )

        data = exp.to_dict()
        restored = Experiment.from_dict(data)

        assert restored.name == exp.name
        assert restored.parameters == exp.parameters
        assert restored.tags == exp.tags
