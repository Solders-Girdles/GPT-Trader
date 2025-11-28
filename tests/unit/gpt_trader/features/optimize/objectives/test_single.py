"""Unit tests for single-metric objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest
from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.single import (
    CostAdjustedReturnObjective,
    DownsideVolatilityObjective,
    DrawdownRecoveryObjective,
    ExecutionQualityObjective,
    HoldDurationObjective,
    LeverageAdjustedReturnObjective,
    SharpeRatioObjective,
    SortinoRatioObjective,
    StreakConsistencyObjective,
    TailRiskAdjustedReturnObjective,
    TimeEfficiencyObjective,
    TotalReturnObjective,
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


class TestSharpeRatioObjective:
    def test_calculate(self, mock_results):
        """Test Sharpe ratio calculation."""
        result, risk, stats = mock_results
        objective = SharpeRatioObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 1.5

    def test_calculate_none(self, mock_results):
        """Test calculation when Sharpe is None."""
        result, risk, stats = mock_results
        risk.sharpe_ratio = None

        objective = SharpeRatioObjective()
        value = objective.calculate(result, risk, stats)
        assert value == float("-inf")

    def test_feasibility(self, mock_results):
        """Test feasibility check."""
        result, risk, stats = mock_results
        objective = SharpeRatioObjective(min_trades=10)

        assert objective.is_feasible(result, risk, stats)

        stats.total_trades = 5
        assert not objective.is_feasible(result, risk, stats)


class TestTotalReturnObjective:
    def test_calculate(self, mock_results):
        """Test total return calculation."""
        result, risk, stats = mock_results
        objective = TotalReturnObjective()

        value = objective.calculate(result, risk, stats)
        assert value == 0.1


# =============================================================================
# Risk-Focused Objectives Tests
# =============================================================================


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

        # Should pass: 5 days < 10 days max
        assert objective.is_feasible(result, risk, stats)

        # Should fail: duration exceeds max
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

        # total_return_pct (10.0) / var_99_daily (4.0) = 2.5
        value = objective.calculate(result, risk, stats)
        assert value == 2.5

    def test_calculate_zero_var(self, mock_results):
        """Test calculation when VaR is zero."""
        result, risk, stats = mock_results
        risk.var_99_daily = Decimal("0")

        objective = TailRiskAdjustedReturnObjective()
        value = objective.calculate(result, risk, stats)
        assert value == float("-inf")


# =============================================================================
# Trade Quality Objectives Tests
# =============================================================================


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
        objective = StreakConsistencyObjective(
            min_trades=10, max_allowed_consecutive_losses=5
        )

        # Should pass: 3 < 5
        assert objective.is_feasible(result, risk, stats)

        # Should fail: streak exceeds max
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

        # total_return_usd (1000) - fees_paid (50) = 950
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
        objective = ExecutionQualityObjective(
            slippage_weight=0.5, fill_rate_weight=0.5
        )

        # slippage_score = 100 - 5 = 95
        # fill_rate_score = 95
        # 0.5 * 95 + 0.5 * 95 = 95
        value = objective.calculate(result, risk, stats)
        assert value == 95.0


# =============================================================================
# Exposure/Timing Objectives Tests
# =============================================================================


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

        # total_return_pct (10.0) / time_in_market_pct (50.0) = 0.2
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

        # avg_hold_time_minutes = 60, target = 60, deviation = 0
        value = objective.calculate(result, risk, stats)
        assert value == 0.0

    def test_calculate_deviation(self, mock_results):
        """Test calculation with deviation from target."""
        result, risk, stats = mock_results
        stats.avg_hold_time_minutes = Decimal("90.0")
        objective = HoldDurationObjective(target_minutes=60.0)

        # deviation = |90 - 60| = 30
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

        # total_return_pct (10.0) / avg_leverage_used (2.0) = 5.0
        value = objective.calculate(result, risk, stats)
        assert value == 5.0

    def test_calculate_no_leverage(self, mock_results):
        """Test calculation when no leverage tracked."""
        result, risk, stats = mock_results
        risk.avg_leverage_used = Decimal("0")

        objective = LeverageAdjustedReturnObjective()
        # Should default to 1x leverage
        value = objective.calculate(result, risk, stats)
        assert value == 10.0  # 10 / 1
