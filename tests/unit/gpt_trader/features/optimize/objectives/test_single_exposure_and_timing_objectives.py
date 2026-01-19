"""Unit tests for single-metric exposure and timing objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.single import (
    HoldDurationObjective,
    LeverageAdjustedReturnObjective,
    TimeEfficiencyObjective,
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


class TestTimeEfficiencyObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = TimeEfficiencyObjective()
        assert objective.name == "time_efficiency"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test time efficiency calculation."""
        result, risk, stats = mock_results
        objective = TimeEfficiencyObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 0.2

    def test_calculate_low_exposure(self, mock_results):
        """Test calculation with very low market exposure."""
        result, risk, stats = mock_results
        risk.time_in_market_pct = Decimal("2.0")  # Below default min of 5%

        objective = TimeEfficiencyObjective()
        value = objective.calculate(result, risk, stats)
        assert value == float("-inf")


class TestHoldDurationObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = HoldDurationObjective()
        assert objective.name == "hold_duration"
        assert objective.direction == "minimize"

    def test_calculate_at_target(self, mock_results):
        """Test calculation when at target."""
        result, risk, stats = mock_results
        objective = HoldDurationObjective(target_minutes=60.0)

        value = objective.calculate(result, risk, stats)
        assert value == 0.0

    def test_calculate_deviation(self, mock_results):
        """Test calculation with deviation from target."""
        result, risk, stats = mock_results
        stats.avg_hold_time_minutes = Decimal("90.0")
        objective = HoldDurationObjective(target_minutes=60.0)

        value = objective.calculate(result, risk, stats)
        assert value == 30.0


class TestLeverageAdjustedReturnObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = LeverageAdjustedReturnObjective()
        assert objective.name == "leverage_adjusted_return"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test leverage adjusted return calculation."""
        result, risk, stats = mock_results
        objective = LeverageAdjustedReturnObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 5.0

    def test_calculate_no_leverage(self, mock_results):
        """Test calculation when no leverage tracked."""
        result, risk, stats = mock_results
        risk.avg_leverage_used = Decimal("0")

        objective = LeverageAdjustedReturnObjective()
        value = objective.calculate(result, risk, stats)
        assert value == 10.0  # 10 / 1
