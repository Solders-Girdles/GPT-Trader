"""Tests for ExperimentResult."""

from datetime import datetime

import pytest

from gpt_trader.features.strategy_dev.lab.experiment import ExperimentResult


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
