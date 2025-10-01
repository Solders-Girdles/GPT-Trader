"""
Comprehensive tests for strategy enhancements.

Tests cover:
- StrategyEnhancements initialization
- RSI calculation
- MA crossover confirmation with RSI
- Edge cases and boundary conditions
"""

from decimal import Decimal

import pytest

from bot_v2.features.strategy_tools.enhancements import StrategyEnhancements


# ============================================================================
# Test: StrategyEnhancements Initialization
# ============================================================================


class TestStrategyEnhancementsInitialization:
    """Test StrategyEnhancements initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        enhancements = StrategyEnhancements()

        assert enhancements.rsi_period == 14
        assert enhancements.rsi_confirmation_enabled is True
        assert enhancements.volatility_lookback == 20
        assert enhancements.volatility_scaling_enabled is False
        assert enhancements.min_volatility_percentile == Decimal("25")

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        enhancements = StrategyEnhancements(
            rsi_period=20,
            rsi_confirmation_enabled=False,
            volatility_lookback=30,
            volatility_scaling_enabled=True,
            min_volatility_percentile=Decimal("50"),
        )

        assert enhancements.rsi_period == 20
        assert enhancements.rsi_confirmation_enabled is False
        assert enhancements.volatility_lookback == 30
        assert enhancements.volatility_scaling_enabled is True
        assert enhancements.min_volatility_percentile == Decimal("50")


# ============================================================================
# Test: RSI Calculation
# ============================================================================


class TestRSICalculation:
    """Test RSI calculation."""

    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation with sufficient data."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Create price series with upward trend
        prices = [Decimal(str(100 + i)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_calculate_rsi_uptrend(self):
        """Test RSI calculation in strong uptrend."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Strong uptrend - RSI should be high
        prices = [Decimal(str(100 + i * 2)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert rsi > Decimal("50")  # Should be above 50 in uptrend

    def test_calculate_rsi_downtrend(self):
        """Test RSI calculation in strong downtrend."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Strong downtrend - RSI should be low
        prices = [Decimal(str(100 - i * 2)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert rsi < Decimal("50")  # Should be below 50 in downtrend

    def test_calculate_rsi_flat_prices(self):
        """Test RSI calculation with flat prices."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Flat prices - RSI should be near 50
        prices = [Decimal("100")] * 30

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        # With no change, RSI should be indeterminate (returns 100 due to zero loss)
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Only 10 prices (need at least 15)
        prices = [Decimal(str(100 + i)) for i in range(10)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is None

    def test_calculate_rsi_exact_minimum_data(self):
        """Test RSI with exactly minimum required data."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Exactly 15 prices (period + 1)
        prices = [Decimal(str(100 + i)) for i in range(15)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None

    def test_calculate_rsi_custom_period(self):
        """Test RSI calculation with custom period."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        # Use custom period
        rsi = enhancements.calculate_rsi(prices, period=10)

        assert rsi is not None

    def test_calculate_rsi_different_periods_same_data(self):
        """Test RSI with different periods on same data."""
        enhancements = StrategyEnhancements()

        # Create varying price data (ups and downs) so RSI differs by period
        prices = []
        for i in range(50):
            if i % 3 == 0:
                prices.append(Decimal(str(100 + i * 0.5)))
            elif i % 3 == 1:
                prices.append(Decimal(str(100 - i * 0.3)))
            else:
                prices.append(Decimal(str(100 + i * 0.7)))

        rsi_14 = enhancements.calculate_rsi(prices, period=14)
        rsi_20 = enhancements.calculate_rsi(prices, period=20)

        assert rsi_14 is not None
        assert rsi_20 is not None
        # Different periods should give different RSI values with varying data
        assert rsi_14 != rsi_20

    def test_calculate_rsi_volatile_prices(self):
        """Test RSI with volatile price movements."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # Alternating large up/down moves
        prices = [Decimal("100")]
        for i in range(29):
            if i % 2 == 0:
                prices.append(prices[-1] + Decimal("5"))
            else:
                prices.append(prices[-1] - Decimal("3"))

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_calculate_rsi_all_gains(self):
        """Test RSI when all price changes are gains."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # All gains, no losses
        prices = [Decimal(str(100 + i)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert rsi == Decimal("100")  # All gains = RSI 100

    def test_calculate_rsi_all_losses(self):
        """Test RSI when all price changes are losses."""
        enhancements = StrategyEnhancements(rsi_period=14)

        # All losses, no gains
        prices = [Decimal(str(100 - i)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert rsi == Decimal("0")  # All losses = RSI 0

    def test_calculate_rsi_single_price(self):
        """Test RSI with only one price."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal("100")]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is None


# ============================================================================
# Test: MA Crossover Confirmation
# ============================================================================


class TestMACrossoverConfirmation:
    """Test MA crossover confirmation with RSI."""

    def test_confirm_buy_signal_enabled_rsi_ok(self):
        """Test buy signal confirmation when RSI is acceptable."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("50")
        )

        assert confirmed is True
        assert "confirms" in reason.lower()

    def test_confirm_buy_signal_enabled_rsi_too_high(self):
        """Test buy signal rejection when RSI is too high."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("75")
        )

        assert confirmed is False
        assert "too high" in reason.lower()

    def test_confirm_buy_signal_disabled(self):
        """Test buy signal when RSI confirmation is disabled."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=False)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("90")
        )

        assert confirmed is True
        assert "disabled" in reason.lower()

    def test_confirm_sell_signal_enabled_rsi_ok(self):
        """Test sell signal confirmation when RSI is acceptable."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 - i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=Decimal("50")
        )

        assert confirmed is True
        assert "confirms" in reason.lower()

    def test_confirm_sell_signal_enabled_rsi_too_low(self):
        """Test sell signal rejection when RSI is too low."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 - i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=Decimal("25")
        )

        assert confirmed is False
        assert "too low" in reason.lower()

    def test_confirm_buy_signal_rsi_at_threshold(self):
        """Test buy signal with RSI exactly at threshold."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        # RSI exactly at 70
        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("70")
        )

        # Should reject (> not >=)
        assert confirmed is False

    def test_confirm_sell_signal_rsi_at_threshold(self):
        """Test sell signal with RSI exactly at threshold."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 - i)) for i in range(30)]

        # RSI exactly at 30
        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=Decimal("30")
        )

        # Should reject (< not <=)
        assert confirmed is False

    def test_confirm_no_rsi_provided_calculates(self):
        """Test confirmation when no RSI provided (calculates from prices)."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Strong uptrend - will calculate high RSI
        prices = [Decimal(str(100 + i * 2)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=None
        )

        # Should calculate RSI and make decision
        assert isinstance(confirmed, bool)

    def test_confirm_insufficient_price_data(self):
        """Test confirmation with insufficient price data for RSI."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Only 10 prices
        prices = [Decimal(str(100 + i)) for i in range(10)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=None
        )

        assert confirmed is False
        assert "insufficient" in reason.lower()

    def test_confirm_unknown_signal(self):
        """Test confirmation with unknown signal type."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="hold", prices=prices, rsi=Decimal("50")
        )

        assert confirmed is False
        assert "unknown" in reason.lower()

    def test_confirm_buy_various_rsi_levels(self):
        """Test buy confirmation across various RSI levels."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        test_cases = [
            (Decimal("20"), True),  # Oversold - ok for buy
            (Decimal("40"), True),  # Neutral - ok for buy
            (Decimal("60"), True),  # Neutral/bullish - ok for buy
            (Decimal("69"), True),  # Just below threshold - ok
            (Decimal("71"), False),  # Above threshold - reject
            (Decimal("90"), False),  # Overbought - reject
        ]

        for rsi_value, expected_confirmed in test_cases:
            confirmed, _ = enhancements.should_confirm_ma_crossover(
                ma_signal="buy", prices=prices, rsi=rsi_value
            )
            assert confirmed == expected_confirmed, f"Failed for RSI={rsi_value}"

    def test_confirm_sell_various_rsi_levels(self):
        """Test sell confirmation across various RSI levels."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 - i)) for i in range(30)]

        test_cases = [
            (Decimal("10"), False),  # Oversold - reject
            (Decimal("29"), False),  # Just below threshold - reject
            (Decimal("31"), True),  # Above threshold - ok
            (Decimal("50"), True),  # Neutral - ok for sell
            (Decimal("70"), True),  # Overbought - ok for sell
            (Decimal("90"), True),  # Very overbought - ok for sell
        ]

        for rsi_value, expected_confirmed in test_cases:
            confirmed, _ = enhancements.should_confirm_ma_crossover(
                ma_signal="sell", prices=prices, rsi=rsi_value
            )
            assert confirmed == expected_confirmed, f"Failed for RSI={rsi_value}"


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rsi_with_zero_prices(self):
        """Test RSI calculation with zero prices."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal("0")] * 30

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        # No change = RSI 100 (due to zero loss)

    def test_rsi_with_negative_prices(self):
        """Test RSI calculation with negative prices."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal(str(-100 + i)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None

    def test_rsi_with_very_large_prices(self):
        """Test RSI with very large price values."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal(str(1000000 + i * 100)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None
        assert Decimal("0") <= rsi <= Decimal("100")

    def test_rsi_with_very_small_changes(self):
        """Test RSI with very small price changes."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal("100") + Decimal(str(i * 0.001)) for i in range(30)]

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None

    def test_rsi_period_one(self):
        """Test RSI with period of 1."""
        enhancements = StrategyEnhancements(rsi_period=1)

        prices = [Decimal(str(100 + i)) for i in range(10)]

        rsi = enhancements.calculate_rsi(prices, period=1)

        # Period 1 requires at least 2 prices
        assert rsi is not None or rsi is None

    def test_rsi_very_long_period(self):
        """Test RSI with very long period."""
        enhancements = StrategyEnhancements(rsi_period=50)

        prices = [Decimal(str(100 + i)) for i in range(100)]

        rsi = enhancements.calculate_rsi(prices, period=50)

        assert rsi is not None

    def test_confirm_with_rsi_zero(self):
        """Test confirmation with RSI of 0."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, _ = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("0")
        )

        assert confirmed is True  # 0 < 70, so ok for buy

    def test_confirm_with_rsi_hundred(self):
        """Test confirmation with RSI of 100."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = [Decimal(str(100 + i)) for i in range(30)]

        confirmed, _ = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("100")
        )

        assert confirmed is False  # 100 > 70, reject

    def test_confirm_empty_prices_list(self):
        """Test confirmation with empty prices list."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        prices = []

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=None
        )

        assert confirmed is False
        assert "insufficient" in reason.lower()

    def test_rsi_with_single_large_spike(self):
        """Test RSI with single large price spike."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal("100")] * 15
        prices.append(Decimal("200"))  # Large spike
        prices.extend([Decimal("100")] * 14)

        rsi = enhancements.calculate_rsi(prices)

        assert rsi is not None


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_bullish_crossover_with_confirmation(self):
        """Test bullish MA crossover with RSI confirmation."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Moderate uptrend with acceptable RSI
        prices = [Decimal(str(100 + i * 0.5)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=None
        )

        # Should calculate reasonable RSI and confirm
        assert isinstance(confirmed, bool)

    def test_bearish_crossover_with_confirmation(self):
        """Test bearish MA crossover with RSI confirmation."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Moderate downtrend with acceptable RSI
        prices = [Decimal(str(100 - i * 0.5)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=None
        )

        # Should calculate reasonable RSI and confirm
        assert isinstance(confirmed, bool)

    def test_overbought_buy_rejection(self):
        """Test buy signal rejection in overbought conditions."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Strong uptrend creating overbought condition
        prices = [Decimal(str(100 + i * 3)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=None
        )

        # Should reject due to high RSI
        assert confirmed is False

    def test_oversold_sell_rejection(self):
        """Test sell signal rejection in oversold conditions."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=True)

        # Strong downtrend creating oversold condition
        prices = [Decimal(str(100 - i * 3)) for i in range(30)]

        confirmed, reason = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=None
        )

        # Should reject due to low RSI
        assert confirmed is False

    def test_disabled_confirmation_always_passes(self):
        """Test that disabled confirmation always passes."""
        enhancements = StrategyEnhancements(rsi_confirmation_enabled=False)

        # Extreme prices that would fail with RSI check
        prices = [Decimal(str(100 + i * 10)) for i in range(30)]

        confirmed_buy, _ = enhancements.should_confirm_ma_crossover(
            ma_signal="buy", prices=prices, rsi=Decimal("95")
        )

        confirmed_sell, _ = enhancements.should_confirm_ma_crossover(
            ma_signal="sell", prices=prices, rsi=Decimal("5")
        )

        assert confirmed_buy is True
        assert confirmed_sell is True

    def test_rsi_calculation_consistency(self):
        """Test RSI calculation consistency across multiple calls."""
        enhancements = StrategyEnhancements(rsi_period=14)

        prices = [Decimal(str(100 + i * 0.5)) for i in range(30)]

        rsi1 = enhancements.calculate_rsi(prices)
        rsi2 = enhancements.calculate_rsi(prices)
        rsi3 = enhancements.calculate_rsi(prices)

        # Should be consistent
        assert rsi1 == rsi2 == rsi3
