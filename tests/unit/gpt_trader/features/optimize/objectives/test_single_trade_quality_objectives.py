"""Unit tests for single-metric trade quality objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.single import (
    CostAdjustedReturnObjective,
    ExecutionQualityObjective,
    StreakConsistencyObjective,
)


@pytest.fixture
def mock_results():
    """Create mock results for testing."""
    result = Mock(spec=BacktestResult)
    result.total_return = Decimal("0.1")
    result.total_return_usd = Decimal("1000")
    result.fees_paid = Decimal("50")

    risk = Mock(spec=RiskMetrics)
    risk.sharpe_ratio = Decimal("1.5")
    risk.sortino_ratio = Decimal("2.0")
    risk.var_95_daily = Decimal("2.5")
    risk.var_99_daily = Decimal("4.0")
    risk.drawdown_duration_days = 5
    risk.downside_volatility = Decimal("15.0")
    risk.total_return_pct = Decimal("10.0")
    risk.time_in_market_pct = Decimal("50.0")
    risk.avg_leverage_used = Decimal("2.0")

    stats = Mock(spec=TradeStatistics)
    stats.total_trades = 20
    stats.max_consecutive_losses = 3
    stats.avg_slippage_bps = Decimal("5.0")
    stats.limit_fill_rate = Decimal("95.0")
    stats.avg_hold_time_minutes = Decimal("60.0")

    return result, risk, stats


class TestStreakConsistencyObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = StreakConsistencyObjective()
        assert objective.name == "streak_consistency"
        assert objective.direction == "minimize"

    def test_calculate(self, mock_results):
        """Test streak consistency calculation."""
        result, risk, stats = mock_results
        objective = StreakConsistencyObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 3

    def test_feasibility_with_streak_constraint(self, mock_results):
        """Test feasibility with max streak constraint."""
        result, risk, stats = mock_results
        objective = StreakConsistencyObjective(min_trades=10, max_allowed_consecutive_losses=5)

        assert objective.is_feasible(result, risk, stats)

        stats.max_consecutive_losses = 6
        assert not objective.is_feasible(result, risk, stats)


class TestCostAdjustedReturnObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = CostAdjustedReturnObjective()
        assert objective.name == "cost_adjusted_return"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test cost adjusted return calculation."""
        result, risk, stats = mock_results
        objective = CostAdjustedReturnObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 950.0


class TestExecutionQualityObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = ExecutionQualityObjective()
        assert objective.name == "execution_quality"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test execution quality calculation."""
        result, risk, stats = mock_results
        objective = ExecutionQualityObjective(slippage_weight=0.5, fill_rate_weight=0.5)

        value = objective.calculate(result, risk, stats)
        assert value == 95.0
