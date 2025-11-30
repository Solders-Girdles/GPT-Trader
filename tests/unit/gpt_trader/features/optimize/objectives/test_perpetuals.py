"""Unit tests for perpetuals-specific objectives."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.perpetuals import (
    FundingAdjustedReturnObjective,
    FundingEfficiencyObjective,
)


@pytest.fixture
def mock_results():
    """Create mock results for testing."""
    result = Mock(spec=BacktestResult)
    result.total_return_usd = Decimal("1000")
    result.realized_pnl = Decimal("1100")
    result.funding_pnl = Decimal("-100")  # Negative = paid funding

    risk = Mock(spec=RiskMetrics)

    stats = Mock(spec=TradeStatistics)
    stats.total_trades = 20

    return result, risk, stats


class TestFundingAdjustedReturnObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = FundingAdjustedReturnObjective()
        assert objective.name == "funding_adjusted_return"
        assert objective.direction == "maximize"

    def test_calculate(self, mock_results):
        """Test funding adjusted return calculation."""
        result, risk, stats = mock_results
        objective = FundingAdjustedReturnObjective()

        # total_return_usd includes funding already
        value = objective.calculate(result, risk, stats)
        assert value == 1000.0

    def test_feasibility(self, mock_results):
        """Test feasibility check."""
        result, risk, stats = mock_results
        objective = FundingAdjustedReturnObjective(min_trades=10)

        assert objective.is_feasible(result, risk, stats)

        stats.total_trades = 5
        assert not objective.is_feasible(result, risk, stats)


class TestFundingEfficiencyObjective:
    def test_properties(self):
        """Test objective properties."""
        objective = FundingEfficiencyObjective()
        assert objective.name == "funding_efficiency"
        assert objective.direction == "maximize"

    def test_calculate_negative_funding(self, mock_results):
        """Test calculation when paying funding."""
        result, risk, stats = mock_results
        # realized_pnl = 1100, funding_pnl = -100
        # ratio = 1100 / 100 = 11.0

        objective = FundingEfficiencyObjective()
        value = objective.calculate(result, risk, stats)
        assert value == 11.0

    def test_calculate_positive_funding(self, mock_results):
        """Test calculation when receiving funding."""
        result, risk, stats = mock_results
        result.funding_pnl = Decimal("50")  # Receiving funding

        objective = FundingEfficiencyObjective()
        # When receiving funding, just return realized PnL
        value = objective.calculate(result, risk, stats)
        assert value == 1100.0  # realized_pnl

    def test_calculate_negligible_funding(self, mock_results):
        """Test calculation when funding is negligible."""
        result, risk, stats = mock_results
        result.funding_pnl = Decimal("-0.005")  # Very small

        objective = FundingEfficiencyObjective(min_funding_threshold=0.01)
        # Funding below threshold, just return realized PnL
        value = objective.calculate(result, risk, stats)
        assert value == 1100.0

    def test_feasibility(self, mock_results):
        """Test feasibility check."""
        result, risk, stats = mock_results
        objective = FundingEfficiencyObjective(min_trades=10)

        assert objective.is_feasible(result, risk, stats)

        stats.total_trades = 5
        assert not objective.is_feasible(result, risk, stats)
