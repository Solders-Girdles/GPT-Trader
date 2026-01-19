"""Unit tests for advanced constraint types."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.backtesting.metrics.risk import RiskMetrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics
from gpt_trader.backtesting.types import BacktestResult
from gpt_trader.features.optimize.objectives.constraints import (
    CircuitBreakerConstraint,
    ConditionalConstraint,
    RangeConstraint,
    ReduceOnlyConstraint,
)


@pytest.fixture
def mock_results():
    """Create mock results for testing."""
    result = Mock(spec=BacktestResult)
    result.circuit_breaker_triggers = 0
    result.reduce_only_periods = 0

    risk = Mock(spec=RiskMetrics)
    risk.max_drawdown_pct = Decimal("10.0")
    risk.var_95_daily = Decimal("3.0")

    stats = Mock(spec=TradeStatistics)
    stats.total_trades = 25
    stats.win_rate = Decimal("55.0")
    stats.loss_rate = Decimal("45.0")
    stats.profit_factor = Decimal("1.8")

    return result, risk, stats


@pytest.fixture
def loss_rate_profit_factor_constraint() -> ConditionalConstraint:
    return ConditionalConstraint(
        name="high_loss_requires_high_pf",
        condition_metric="loss_rate",
        condition_operator="gt",
        condition_threshold=50.0,
        constrained_metric="profit_factor",
        constraint_operator="gt",
        constraint_threshold=2.0,
    )


class TestConditionalConstraint:
    def test_validation_invalid_condition_operator(self):
        """Test validation rejects invalid condition operator."""
        with pytest.raises(ValueError, match="condition_operator"):
            ConditionalConstraint(
                name="test",
                condition_metric="loss_rate",
                condition_operator="invalid",  # Invalid
                condition_threshold=40.0,
                constrained_metric="profit_factor",
                constraint_operator="gt",
                constraint_threshold=2.0,
            )

    def test_validation_invalid_constraint_operator(self):
        """Test validation rejects invalid constraint operator."""
        with pytest.raises(ValueError, match="constraint_operator"):
            ConditionalConstraint(
                name="test",
                condition_metric="loss_rate",
                condition_operator="gt",
                condition_threshold=40.0,
                constrained_metric="profit_factor",
                constraint_operator="invalid",  # Invalid
                constraint_threshold=2.0,
            )

    def test_satisfied_condition_not_met(self, mock_results, loss_rate_profit_factor_constraint):
        """Test constraint passes when condition is not met."""
        result, risk, stats = mock_results
        assert loss_rate_profit_factor_constraint.is_satisfied(result, risk, stats)

    def test_satisfied_condition_met_constraint_satisfied(
        self, mock_results, loss_rate_profit_factor_constraint
    ):
        """Test constraint passes when condition is met and constraint is satisfied."""
        result, risk, stats = mock_results
        stats.loss_rate = Decimal("55.0")  # Now loss_rate > 50%
        stats.profit_factor = Decimal("2.5")  # And profit_factor > 2.0

        # Condition met AND constraint satisfied
        assert loss_rate_profit_factor_constraint.is_satisfied(result, risk, stats)

    def test_not_satisfied_condition_met_constraint_not_satisfied(
        self, mock_results, loss_rate_profit_factor_constraint
    ):
        """Test constraint fails when condition is met but constraint is not satisfied."""
        result, risk, stats = mock_results
        stats.loss_rate = Decimal("55.0")  # loss_rate > 50%
        stats.profit_factor = Decimal("1.5")  # But profit_factor < 2.0

        # Condition met BUT constraint not satisfied
        assert not loss_rate_profit_factor_constraint.is_satisfied(result, risk, stats)


class TestRangeConstraint:
    def test_validation_invalid_bounds(self):
        """Test validation rejects lower > upper."""
        with pytest.raises(ValueError, match="lower_bound"):
            RangeConstraint(
                name="test",
                metric="win_rate",
                lower_bound=70.0,  # Greater than upper
                upper_bound=40.0,
            )

    def test_satisfied_within_range_inclusive(self, mock_results):
        """Test constraint passes when value is within inclusive range."""
        result, risk, stats = mock_results
        constraint = RangeConstraint(
            name="balanced_win_rate",
            metric="win_rate",
            lower_bound=40.0,
            upper_bound=70.0,
            inclusive=True,
        )

        # win_rate = 55%, which is in [40, 70]
        assert constraint.is_satisfied(result, risk, stats)

    def test_satisfied_at_bounds_inclusive(self, mock_results):
        """Test constraint passes when value equals bounds (inclusive)."""
        result, risk, stats = mock_results
        stats.win_rate = Decimal("40.0")  # At lower bound

        constraint = RangeConstraint(
            name="balanced_win_rate",
            metric="win_rate",
            lower_bound=40.0,
            upper_bound=70.0,
            inclusive=True,
        )

        assert constraint.is_satisfied(result, risk, stats)

    def test_not_satisfied_at_bounds_exclusive(self, mock_results):
        """Test constraint fails when value equals bounds (exclusive)."""
        result, risk, stats = mock_results
        stats.win_rate = Decimal("40.0")  # At lower bound

        constraint = RangeConstraint(
            name="balanced_win_rate",
            metric="win_rate",
            lower_bound=40.0,
            upper_bound=70.0,
            inclusive=False,
        )

        # Value at bound should fail for exclusive
        assert not constraint.is_satisfied(result, risk, stats)

    def test_not_satisfied_outside_range(self, mock_results):
        """Test constraint fails when value is outside range."""
        result, risk, stats = mock_results
        stats.win_rate = Decimal("80.0")  # Above upper bound

        constraint = RangeConstraint(
            name="balanced_win_rate",
            metric="win_rate",
            lower_bound=40.0,
            upper_bound=70.0,
        )

        assert not constraint.is_satisfied(result, risk, stats)


class TestCircuitBreakerConstraint:
    def test_name_property(self):
        """Test name property."""
        constraint = CircuitBreakerConstraint()
        assert constraint.name == "circuit_breaker_limit"

    def test_satisfied_no_triggers(self, mock_results):
        """Test constraint passes with no circuit breaker triggers."""
        result, risk, stats = mock_results
        constraint = CircuitBreakerConstraint(max_triggers=0)

        # No triggers
        assert constraint.is_satisfied(result, risk, stats)

    def test_satisfied_within_limit(self, mock_results):
        """Test constraint passes when triggers within limit."""
        result, risk, stats = mock_results
        result.circuit_breaker_triggers = 2

        constraint = CircuitBreakerConstraint(max_triggers=5)
        assert constraint.is_satisfied(result, risk, stats)

    def test_not_satisfied_exceeds_limit(self, mock_results):
        """Test constraint fails when triggers exceed limit."""
        result, risk, stats = mock_results
        result.circuit_breaker_triggers = 3

        constraint = CircuitBreakerConstraint(max_triggers=2)
        assert not constraint.is_satisfied(result, risk, stats)


class TestReduceOnlyConstraint:
    def test_name_property(self):
        """Test name property."""
        constraint = ReduceOnlyConstraint()
        assert constraint.name == "reduce_only_limit"

    def test_satisfied_no_periods(self, mock_results):
        """Test constraint passes with no reduce-only periods."""
        result, risk, stats = mock_results
        constraint = ReduceOnlyConstraint(max_periods=0)

        # No reduce-only periods
        assert constraint.is_satisfied(result, risk, stats)

    def test_not_satisfied_exceeds_limit(self, mock_results):
        """Test constraint fails when periods exceed limit."""
        result, risk, stats = mock_results
        result.reduce_only_periods = 2

        constraint = ReduceOnlyConstraint(max_periods=1)
        assert not constraint.is_satisfied(result, risk, stats)
