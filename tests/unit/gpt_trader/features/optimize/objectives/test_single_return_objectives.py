"""Unit tests for single-metric return objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.single import (
    SharpeRatioObjective,
    TotalReturnObjective,
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
