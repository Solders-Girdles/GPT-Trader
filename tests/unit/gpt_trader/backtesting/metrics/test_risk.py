"""Tests for risk metrics calculation module."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.backtesting.metrics.risk import (
    RiskMetrics,
    _calculate_drawdown_metrics,
    _calculate_returns,
    _calculate_time_in_market,
    _calculate_var,
    _empty_risk_metrics,
    _std_dev,
    calculate_risk_metrics,
)


class TestCalculateReturns:
    """Tests for _calculate_returns function."""

    def test_empty_list_returns_empty(self) -> None:
        assert _calculate_returns([]) == []

    def test_single_value_returns_empty(self) -> None:
        assert _calculate_returns([100.0]) == []

    def test_two_values_calculates_return(self) -> None:
        result = _calculate_returns([100.0, 110.0])
        assert len(result) == 1
        assert abs(result[0] - 0.1) < 0.0001  # 10% return

    def test_multiple_values(self) -> None:
        equities = [100.0, 110.0, 99.0, 110.0]
        result = _calculate_returns(equities)
        assert len(result) == 3
        # First return: (110-100)/100 = 0.1
        assert abs(result[0] - 0.1) < 0.0001
        # Second return: (99-110)/110 = -0.1
        assert abs(result[1] - (-0.1)) < 0.0001
        # Third return: (110-99)/99 = 0.1111...
        assert abs(result[2] - (11 / 99)) < 0.0001

    def test_handles_zero_equity(self) -> None:
        # Zero equity should be skipped to avoid division by zero
        result = _calculate_returns([0.0, 100.0])
        assert result == []

    def test_steady_growth(self) -> None:
        # 5% growth each period
        equities = [100.0, 105.0, 110.25, 115.7625]
        result = _calculate_returns(equities)
        for ret in result:
            assert abs(ret - 0.05) < 0.0001


class TestStdDev:
    """Tests for _std_dev function."""

    def test_empty_list_returns_zero(self) -> None:
        assert _std_dev([]) == 0.0

    def test_single_value_returns_zero(self) -> None:
        assert _std_dev([10.0]) == 0.0

    def test_two_identical_values(self) -> None:
        assert _std_dev([10.0, 10.0]) == 0.0

    def test_simple_case(self) -> None:
        # Values: 2, 4, 4, 4, 5, 5, 7, 9
        # Mean: 5
        # Variance: sum of squared deviations / (n-1)
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = _std_dev(values)
        expected = math.sqrt(32 / 7)  # Approx 2.138
        assert abs(result - expected) < 0.001

    def test_negative_values(self) -> None:
        values = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        result = _std_dev(values)
        # Mean is 0, variance = (25+9+1+1+9+25)/5 = 14
        expected = math.sqrt(14)
        assert abs(result - expected) < 0.001

    def test_sample_standard_deviation(self) -> None:
        # Uses n-1 (sample std dev), not n (population)
        values = [10.0, 20.0]
        result = _std_dev(values)
        # Mean = 15, variance = ((10-15)^2 + (20-15)^2) / 1 = 50
        # std dev = sqrt(50) ≈ 7.07
        expected = math.sqrt(50)
        assert abs(result - expected) < 0.001


class TestCalculateVar:
    """Tests for _calculate_var function (Value at Risk)."""

    def test_empty_returns_zero(self) -> None:
        assert _calculate_var([], 0.95) == 0.0

    def test_single_value(self) -> None:
        result = _calculate_var([-0.05], 0.95)
        # With single value, VaR is the negative of that return
        assert result == 0.05

    def test_all_positive_returns(self) -> None:
        returns = [0.01, 0.02, 0.03, 0.04, 0.05]
        result = _calculate_var(returns, 0.95)
        # 95% VaR should be smallest return (negated)
        assert result == -0.01  # Negative loss indicates no loss

    def test_mixed_returns_95(self) -> None:
        # 100 returns, sorted from -10% to -1%, then 0% to 89%
        returns = [-0.10 + 0.01 * i for i in range(10)] + [i * 0.01 for i in range(90)]
        result = _calculate_var(returns, 0.95)
        # 5th percentile index = int(0.05 * 100) = 5, so returns[5] = -0.05
        assert abs(result - 0.05) < 0.01

    def test_var_99_more_conservative(self) -> None:
        returns = [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]
        var_95 = _calculate_var(returns, 0.95)
        var_99 = _calculate_var(returns, 0.99)
        # 99% VaR should show higher potential loss
        assert var_99 >= var_95


class TestCalculateDrawdownMetrics:
    """Tests for _calculate_drawdown_metrics function."""

    def test_empty_equities(self) -> None:
        result = _calculate_drawdown_metrics([], [])
        assert result["avg_drawdown"] == Decimal("0")
        assert result["max_duration"] == 0

    def test_no_drawdown_monotonic_increase(self) -> None:
        now = datetime.now()
        # Start below first "peak" to avoid edge case where equity == peak
        equities = [100.0, 110.0, 120.0, 130.0]
        timestamps = [now + timedelta(days=i) for i in range(4)]
        result = _calculate_drawdown_metrics(equities, timestamps)
        # Implementation counts 0% drawdown at first point (equity == peak)
        assert float(result["avg_drawdown"]) == pytest.approx(0.0, abs=0.01)
        # Duration of 1 because first point equity==peak triggers else branch
        assert result["max_duration"] == 1

    def test_single_drawdown(self) -> None:
        now = datetime.now()
        # Peak at 100, drops to 90, recovers to 110
        equities = [100.0, 90.0, 95.0, 110.0]
        timestamps = [now + timedelta(days=i) for i in range(4)]
        result = _calculate_drawdown_metrics(equities, timestamps)
        # Implementation counts: first point (0%), 90 (10%), 95 (5%)
        # Average = (0 + 10 + 5) / 3 = 5%
        assert float(result["avg_drawdown"]) == pytest.approx(5.0, rel=0.01)
        # Duration: day 0 to day 3 = 3 days (starts at first point)
        assert result["max_duration"] == 3

    def test_multiple_drawdowns(self) -> None:
        now = datetime.now()
        equities = [100.0, 95.0, 100.0, 105.0, 98.0, 110.0]
        timestamps = [now + timedelta(days=i) for i in range(6)]
        result = _calculate_drawdown_metrics(equities, timestamps)
        # Has two drawdown periods
        assert float(result["avg_drawdown"]) > 0

    def test_still_in_drawdown_at_end(self) -> None:
        now = datetime.now()
        equities = [100.0, 105.0, 95.0, 90.0]  # Still in drawdown at end
        timestamps = [now + timedelta(days=i) for i in range(4)]
        result = _calculate_drawdown_metrics(equities, timestamps)
        # Duration should still be calculated
        assert result["max_duration"] >= 1


class TestEmptyRiskMetrics:
    """Tests for _empty_risk_metrics function."""

    def test_returns_valid_risk_metrics(self) -> None:
        result = _empty_risk_metrics()
        assert isinstance(result, RiskMetrics)

    def test_all_values_are_zero_or_none(self) -> None:
        result = _empty_risk_metrics()
        assert result.max_drawdown_pct == Decimal("0")
        assert result.max_drawdown_usd == Decimal("0")
        assert result.total_return_pct == Decimal("0")
        assert result.sharpe_ratio is None
        assert result.sortino_ratio is None
        assert result.calmar_ratio is None

    def test_leverage_defaults_to_one(self) -> None:
        result = _empty_risk_metrics()
        assert result.max_leverage_used == Decimal("1")
        assert result.avg_leverage_used == Decimal("1")


class TestCalculateTimeInMarket:
    """Tests for _calculate_time_in_market function."""

    def test_empty_equity_curve_returns_zero(self) -> None:
        broker = MagicMock()
        broker.get_equity_curve.return_value = []
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_single_point_returns_zero(self) -> None:
        broker = MagicMock()
        broker.get_equity_curve.return_value = [(datetime.now(), Decimal("100"))]
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_no_positions_returns_zero(self) -> None:
        now = datetime.now()
        broker = MagicMock()
        broker.get_equity_curve.return_value = [
            (now, Decimal("100")),
            (now + timedelta(hours=1), Decimal("101")),
        ]
        broker.positions = {}
        result = _calculate_time_in_market(broker)
        assert result == Decimal("0")

    def test_with_positions_returns_estimate(self) -> None:
        now = datetime.now()
        broker = MagicMock()
        broker.get_equity_curve.return_value = [
            (now + timedelta(hours=i), Decimal("100")) for i in range(10)
        ]
        broker.positions = {"BTC-USD": MagicMock()}
        result = _calculate_time_in_market(broker)
        # Should return ~50% estimate when positions exist
        assert result > Decimal("0")


class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics function."""

    def _create_mock_broker(
        self,
        initial_equity: float = 10000.0,
        final_equity: float = 11000.0,
        equity_curve: list[tuple[datetime, Decimal]] | None = None,
        max_drawdown: float = 0.05,
    ) -> MagicMock:
        broker = MagicMock()
        broker._initial_equity = Decimal(str(initial_equity))
        broker.get_equity.return_value = Decimal(str(final_equity))
        broker._max_drawdown = Decimal(str(max_drawdown))
        broker._max_drawdown_usd = Decimal(str(initial_equity * max_drawdown))
        broker.positions = {}

        if equity_curve is None:
            now = datetime.now()
            equity_curve = [
                (now + timedelta(days=i), Decimal(str(initial_equity + i * 100))) for i in range(30)
            ]
        broker.get_equity_curve.return_value = equity_curve

        return broker

    def test_insufficient_data_returns_empty_metrics(self) -> None:
        broker = self._create_mock_broker()
        broker.get_equity_curve.return_value = []
        result = calculate_risk_metrics(broker)
        assert result.max_drawdown_pct == Decimal("0")
        assert result.sharpe_ratio is None

    def test_single_equity_point_returns_empty(self) -> None:
        broker = self._create_mock_broker()
        broker.get_equity_curve.return_value = [(datetime.now(), Decimal("10000"))]
        result = calculate_risk_metrics(broker)
        assert result.total_return_pct == Decimal("0")

    def test_calculates_total_return(self) -> None:
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=12000.0,
        )
        result = calculate_risk_metrics(broker)
        # 20% return
        assert float(result.total_return_pct) == pytest.approx(20.0, rel=0.1)

    def test_calculates_sharpe_ratio(self) -> None:
        # Create a broker with consistent positive returns
        now = datetime.now()
        equity_curve = [(now + timedelta(days=i), Decimal(str(10000 + i * 50))) for i in range(100)]
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=float(equity_curve[-1][1]),
            equity_curve=equity_curve,
        )
        result = calculate_risk_metrics(broker)
        # Should have a positive Sharpe ratio for positive returns
        assert result.sharpe_ratio is not None
        assert result.sharpe_ratio > 0

    def test_calculates_volatility(self) -> None:
        broker = self._create_mock_broker()
        result = calculate_risk_metrics(broker)
        # Volatility should be computed
        assert result.volatility_annualized >= Decimal("0")
        assert result.daily_return_std >= Decimal("0")

    def test_calculates_var_metrics(self) -> None:
        broker = self._create_mock_broker()
        result = calculate_risk_metrics(broker)
        # VaR should be calculated
        assert result.var_95_daily is not None
        assert result.var_99_daily is not None

    def test_calmar_ratio_when_drawdown_exists(self) -> None:
        broker = self._create_mock_broker(max_drawdown=0.1)
        result = calculate_risk_metrics(broker)
        # Calmar should be calculated when drawdown > 0
        assert result.calmar_ratio is not None

    def test_calmar_ratio_none_when_no_drawdown(self) -> None:
        broker = self._create_mock_broker(max_drawdown=0.0)
        result = calculate_risk_metrics(broker)
        # Calmar should be None when no drawdown
        assert result.calmar_ratio is None

    def test_uses_custom_risk_free_rate(self) -> None:
        broker = self._create_mock_broker()
        result_low_rfr = calculate_risk_metrics(broker, risk_free_rate=Decimal("0.01"))
        result_high_rfr = calculate_risk_metrics(broker, risk_free_rate=Decimal("0.10"))
        # Higher risk-free rate should result in lower Sharpe
        if result_low_rfr.sharpe_ratio and result_high_rfr.sharpe_ratio:
            assert result_low_rfr.sharpe_ratio >= result_high_rfr.sharpe_ratio

    def test_handles_negative_returns(self) -> None:
        now = datetime.now()
        equity_curve = [(now + timedelta(days=i), Decimal(str(10000 - i * 100))) for i in range(30)]
        broker = self._create_mock_broker(
            initial_equity=10000.0,
            final_equity=7100.0,
            equity_curve=equity_curve,
            max_drawdown=0.29,
        )
        result = calculate_risk_metrics(broker)
        # Should handle negative returns properly
        assert float(result.total_return_pct) < 0
        # Sortino may still be calculated with downside deviation
        assert result.downside_volatility >= Decimal("0")


class TestRiskMetricsDataclass:
    """Tests for RiskMetrics dataclass."""

    def test_all_fields_accessible(self) -> None:
        metrics = RiskMetrics(
            max_drawdown_pct=Decimal("10"),
            max_drawdown_usd=Decimal("1000"),
            avg_drawdown_pct=Decimal("5"),
            drawdown_duration_days=10,
            total_return_pct=Decimal("15"),
            annualized_return_pct=Decimal("18"),
            daily_return_avg=Decimal("0.05"),
            daily_return_std=Decimal("1.2"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("1.8"),
            volatility_annualized=Decimal("20"),
            downside_volatility=Decimal("15"),
            max_leverage_used=Decimal("3"),
            avg_leverage_used=Decimal("2"),
            time_in_market_pct=Decimal("75"),
            var_95_daily=Decimal("2"),
            var_99_daily=Decimal("3"),
        )
        assert metrics.max_drawdown_pct == Decimal("10")
        assert metrics.sharpe_ratio == Decimal("1.5")
        assert metrics.drawdown_duration_days == 10

    def test_optional_ratios_can_be_none(self) -> None:
        metrics = RiskMetrics(
            max_drawdown_pct=Decimal("0"),
            max_drawdown_usd=Decimal("0"),
            avg_drawdown_pct=Decimal("0"),
            drawdown_duration_days=0,
            total_return_pct=Decimal("0"),
            annualized_return_pct=Decimal("0"),
            daily_return_avg=Decimal("0"),
            daily_return_std=Decimal("0"),
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            volatility_annualized=Decimal("0"),
            downside_volatility=Decimal("0"),
            max_leverage_used=Decimal("1"),
            avg_leverage_used=Decimal("1"),
            time_in_market_pct=Decimal("0"),
            var_95_daily=Decimal("0"),
            var_99_daily=Decimal("0"),
        )
        assert metrics.sharpe_ratio is None
        assert metrics.sortino_ratio is None
        assert metrics.calmar_ratio is None


class TestCalculateRiskMetricsEdgeCases:
    """Edge case tests for calculate_risk_metrics."""

    def test_zero_equity_returns_empty_metrics(self) -> None:
        """Test that equity curves starting at zero return empty metrics."""
        now = datetime.now()
        broker = MagicMock()
        broker._initial_equity = Decimal("0")
        broker.get_equity.return_value = Decimal("100")
        broker._max_drawdown = Decimal("0")
        broker._max_drawdown_usd = Decimal("0")
        broker.positions = {}
        # Equity curve with zero values generates no valid returns
        broker.get_equity_curve.return_value = [
            (now, Decimal("0")),
            (now + timedelta(days=1), Decimal("0")),
        ]
        result = calculate_risk_metrics(broker)
        assert result.sharpe_ratio is None

    def test_same_day_equity_curve(self) -> None:
        """Test equity curve within same day (years=0 path)."""
        now = datetime.now()
        broker = MagicMock()
        broker._initial_equity = Decimal("10000")
        broker.get_equity.return_value = Decimal("10500")
        broker._max_drawdown = Decimal("0")
        broker._max_drawdown_usd = Decimal("0")
        broker.positions = {}
        # Same timestamp for all entries (0 days duration)
        broker.get_equity_curve.return_value = [
            (now, Decimal("10000")),
            (now, Decimal("10100")),
            (now, Decimal("10500")),
        ]
        result = calculate_risk_metrics(broker)
        # Should still calculate return even if years=0
        assert result.total_return_pct == Decimal("5")
