"""Tests for returns, volatility, and VaR helper functions."""

from __future__ import annotations

import math

from gpt_trader.backtesting.metrics.risk import _calculate_returns, _calculate_var, _std_dev


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
