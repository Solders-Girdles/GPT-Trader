"""Tests for technical indicators module."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.indicators import (
    compute_ma_series,
    detect_crossover,
    exponential_moving_average,
    mean_decimal,
    relative_strength_index,
    simple_moving_average,
    to_decimal,
)


class TestMeanDecimal:
    """Tests for mean_decimal function."""

    def test_empty_list_returns_zero(self) -> None:
        assert mean_decimal([]) == Decimal("0")

    def test_single_value(self) -> None:
        assert mean_decimal([Decimal("100")]) == Decimal("100")

    def test_multiple_values(self) -> None:
        values = [Decimal("10"), Decimal("20"), Decimal("30")]
        assert mean_decimal(values) == Decimal("20")

    def test_decimal_precision(self) -> None:
        values = [Decimal("1"), Decimal("2"), Decimal("3")]
        assert mean_decimal(values) == Decimal("2")


class TestSimpleMovingAverage:
    """Tests for simple_moving_average function."""

    def test_insufficient_data_returns_none(self) -> None:
        values = [Decimal("100"), Decimal("101")]
        assert simple_moving_average(values, period=5) is None

    def test_exact_period_length(self) -> None:
        values = [Decimal("100"), Decimal("102"), Decimal("104"), Decimal("106"), Decimal("108")]
        sma = simple_moving_average(values, period=5)
        assert sma == Decimal("104")  # (100+102+104+106+108)/5

    def test_longer_than_period(self) -> None:
        # SMA uses the last N values
        values = [Decimal("50"), Decimal("100"), Decimal("102"), Decimal("104")]
        sma = simple_moving_average(values, period=3)
        # Should use last 3: 100, 102, 104
        assert sma == Decimal("102")


class TestExponentialMovingAverage:
    """Tests for exponential_moving_average function."""

    def test_insufficient_data_returns_none(self) -> None:
        values = [Decimal("100"), Decimal("101")]
        assert exponential_moving_average(values, period=5) is None

    def test_exact_period_returns_sma(self) -> None:
        # With exactly period values, EMA starts as SMA
        values = [Decimal("100"), Decimal("102"), Decimal("104")]
        ema = exponential_moving_average(values, period=3)
        expected_sma = Decimal("102")
        assert ema == expected_sma

    def test_ema_responds_to_price_changes(self) -> None:
        # EMA should be closer to recent values than SMA
        values = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("150")]
        ema = exponential_moving_average(values, period=3)
        sma = simple_moving_average(values, period=3)
        # EMA should be higher than SMA because it weights recent (150) more
        assert ema is not None
        assert sma is not None
        assert ema > sma


class TestRelativeStrengthIndex:
    """Tests for relative_strength_index function."""

    def test_insufficient_data_returns_none(self) -> None:
        values = [Decimal("100"), Decimal("101")]
        assert relative_strength_index(values, period=14) is None

    def test_minimum_data_for_period(self) -> None:
        # Need period + 1 values
        values = [Decimal(str(i)) for i in range(16)]  # 16 values for period 14
        rsi = relative_strength_index(values, period=14)
        assert rsi is not None
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_all_gains_returns_100(self) -> None:
        # Strictly increasing prices should give RSI near 100
        values = [Decimal(str(100 + i)) for i in range(20)]
        rsi = relative_strength_index(values, period=14)
        assert rsi is not None
        assert rsi == Decimal("100")

    def test_all_losses_returns_near_zero(self) -> None:
        # Strictly decreasing prices should give RSI near 0
        values = [Decimal(str(200 - i)) for i in range(20)]
        rsi = relative_strength_index(values, period=14)
        assert rsi is not None
        assert rsi < Decimal("10")  # Should be very low

    def test_mixed_prices_in_range(self) -> None:
        # Mixed gains and losses should give RSI between 30 and 70 typically
        values = [
            Decimal("100"),
            Decimal("102"),
            Decimal("101"),
            Decimal("103"),
            Decimal("102"),
            Decimal("104"),
            Decimal("103"),
            Decimal("105"),
            Decimal("104"),
            Decimal("106"),
            Decimal("105"),
            Decimal("107"),
            Decimal("106"),
            Decimal("108"),
            Decimal("107"),
            Decimal("109"),
        ]
        rsi = relative_strength_index(values, period=14)
        assert rsi is not None
        assert Decimal("30") < rsi < Decimal("80")


class TestCrossoverDetection:
    """Tests for detect_crossover function."""

    def test_insufficient_data_returns_none(self) -> None:
        fast = [Decimal("100")]
        slow = [Decimal("100")]
        assert detect_crossover(fast, slow, lookback=2) is None

    def test_bullish_crossover(self) -> None:
        # Fast crosses above slow
        fast = [Decimal("98"), Decimal("102")]  # Was below, now above
        slow = [Decimal("100"), Decimal("100")]
        signal = detect_crossover(fast, slow, lookback=2)
        assert signal is not None
        assert signal.crossed is True
        assert signal.direction == "bullish"

    def test_bearish_crossover(self) -> None:
        # Fast crosses below slow
        fast = [Decimal("102"), Decimal("98")]  # Was above, now below
        slow = [Decimal("100"), Decimal("100")]
        signal = detect_crossover(fast, slow, lookback=2)
        assert signal is not None
        assert signal.crossed is True
        assert signal.direction == "bearish"

    def test_no_crossover(self) -> None:
        # Fast stays above slow
        fast = [Decimal("102"), Decimal("104")]
        slow = [Decimal("100"), Decimal("100")]
        signal = detect_crossover(fast, slow, lookback=2)
        assert signal is not None
        assert signal.crossed is False
        assert signal.direction == "none"

    def test_crossover_signal_contains_values(self) -> None:
        fast = [Decimal("98"), Decimal("102")]
        slow = [Decimal("100"), Decimal("100")]
        signal = detect_crossover(fast, slow, lookback=2)
        assert signal is not None
        assert signal.fast_value == Decimal("102")
        assert signal.slow_value == Decimal("100")


class TestComputeMASeries:
    """Tests for compute_ma_series function."""

    def test_sma_series_length_matches_input(self) -> None:
        values = [Decimal(str(i)) for i in range(10)]
        result = compute_ma_series(values, period=3, ma_type="sma")
        assert len(result) == len(values)

    def test_sma_series_first_values_are_simple_avg(self) -> None:
        values = [Decimal("100"), Decimal("102"), Decimal("104"), Decimal("106")]
        result = compute_ma_series(values, period=3, ma_type="sma")
        # First value should be just the first value
        assert result[0] == Decimal("100")
        # Second value should be average of first two
        assert result[1] == Decimal("101")

    def test_ema_series(self) -> None:
        values = [Decimal(str(100 + i)) for i in range(10)]
        result = compute_ma_series(values, period=3, ma_type="ema")
        assert len(result) == len(values)


class TestToDecimal:
    """Tests for to_decimal function."""

    def test_int_conversion(self) -> None:
        assert to_decimal(100) == Decimal("100")

    def test_float_conversion(self) -> None:
        assert to_decimal(100.5) == Decimal("100.5")

    def test_string_conversion(self) -> None:
        assert to_decimal("100.50") == Decimal("100.50")

    def test_decimal_passthrough(self) -> None:
        d = Decimal("100.123")
        assert to_decimal(d) == d
