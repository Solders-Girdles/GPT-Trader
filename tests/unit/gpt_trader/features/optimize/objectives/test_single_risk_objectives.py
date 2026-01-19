"""Unit tests for single-metric risk-focused objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.single import (
    DownsideVolatilityObjective,
    DrawdownRecoveryObjective,
    TailRiskAdjustedReturnObjective,
    ValueAtRisk95Objective,
    ValueAtRisk99Objective,
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


class TestValueAtRisk95Objective:
    def test_properties(self):
        """Test objective properties."""
        objective = ValueAtRisk95Objective()
        assert objective.name == "var_95_daily"
        assert objective.direction == "minimize"

    def test_calculate(self, mock_results):
        """Test VaR 95 calculation."""
        result, risk, stats = mock_results
        objective = ValueAtRisk95Objective()

        value = objective.calculate(result, risk, stats)
        assert value == 2.5

    def test_feasibility(self, mock_results):
        """Test feasibility check."""
        result, risk, stats = mock_results
        objective = ValueAtRisk95Objective(min_trades=10)

        assert objective.is_feasible(result, risk, stats)
        stats.total_trades = 5
        assert not objective.is_feasible(result, risk, stats)


class TestValueAtRisk99Objective:
    def test_properties(self):
        """Test objective properties."""
        objective = ValueAtRisk99Objective()
        assert objective.name == "var_99_daily"
        assert objective.direction == "minimize"

    def test_calculate(self, mock_results):
        """Test VaR 99 calculation."""
        result, risk, stats = mock_results
        objective = ValueAtRisk99Objective()

        value = objective.calculate(result, risk, stats)
        assert value == 4.0


class TestDrawdownRecoveryObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = DrawdownRecoveryObjective()
        assert objective.name == "drawdown_recovery"
        assert objective.direction == "minimize"

    def test_calculate(self, mock_results):
        """Test drawdown recovery calculation."""
        result, risk, stats = mock_results
        objective = DrawdownRecoveryObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 5

    def test_feasibility_with_duration_constraint(self, mock_results):
        """Test feasibility with max duration constraint."""
        result, risk, stats = mock_results
        objective = DrawdownRecoveryObjective(min_trades=10, max_duration_days=10)

        assert objective.is_feasible(result, risk, stats)

        risk.drawdown_duration_days = 15
        assert not objective.is_feasible(result, risk, stats)


class TestDownsideVolatilityObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = DownsideVolatilityObjective()
        assert objective.name == "downside_volatility"
        assert objective.direction == "minimize"

    def test_calculate(self, mock_results):
        """Test downside volatility calculation."""
        result, risk, stats = mock_results
        objective = DownsideVolatilityObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 15.0


class TestTailRiskAdjustedReturnObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = TailRiskAdjustedReturnObjective()
        assert objective.name == "tail_risk_adjusted_return"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test tail risk adjusted return calculation."""
        result, risk, stats = mock_results
        objective = TailRiskAdjustedReturnObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 2.5

    def test_calculate_zero_var(self, mock_results):
        """Test calculation when VaR is zero."""
        result, risk, stats = mock_results
        risk.var_99_daily = Decimal("0")

        objective = TailRiskAdjustedReturnObjective()
        value = objective.calculate(result, risk, stats)
        assert value == float("-inf")
