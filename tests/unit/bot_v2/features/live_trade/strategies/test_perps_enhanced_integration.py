"""
Integration tests for PerpsBaselineEnhancedStrategy with signal calculator.

Verifies that the strategy correctly uses the extracted signal calculator.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.live_trade.strategies.perps_baseline_enhanced import (
    PerpsBaselineEnhancedStrategy,
    StrategyConfig,
)
from bot_v2.features.live_trade.strategies.strategy_signals import SignalSnapshot


class TestSignalCalculatorIntegration:
    """Test strategy integration with signal calculator."""

    def test_strategy_instantiates_calculator_automatically(self):
        """Should create signal calculator during initialization."""
        strategy = PerpsBaselineEnhancedStrategy()

        assert strategy.signal_calculator is not None
        assert strategy.signal_calculator.short_ma_period == 5
        assert strategy.signal_calculator.long_ma_period == 20

    def test_strategy_accepts_injected_calculator(self):
        """Should allow dependency injection of signal calculator."""
        mock_calc = Mock()
        mock_calc.calculate_signals.return_value = SignalSnapshot(
            short_ma=Decimal("105"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=True,
            bearish_cross=False,
            rsi=None,
        )

        strategy = PerpsBaselineEnhancedStrategy(signal_calculator=mock_calc)

        assert strategy.signal_calculator is mock_calc

    def test_calculate_signals_delegates_to_calculator(self):
        """Should delegate signal calculation to calculator."""
        mock_calc = Mock()
        mock_calc.calculate_signals.return_value = SignalSnapshot(
            short_ma=Decimal("105"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=True,
            bearish_cross=False,
            rsi=None,
        )

        strategy = PerpsBaselineEnhancedStrategy(signal_calculator=mock_calc)
        marks = [Decimal("100")] * 25

        signal = strategy._calculate_trading_signals(marks, Decimal("105"))

        assert signal.bullish_cross
        mock_calc.calculate_signals.assert_called_once_with(
            marks, Decimal("105"), require_rsi=False
        )

    def test_calculate_signals_requests_rsi_when_configured(self):
        """Should request RSI when filters require it."""
        from bot_v2.features.live_trade.strategies.perps_baseline_enhanced import (
            StrategyFiltersConfig,
        )

        mock_calc = Mock()
        mock_calc.calculate_signals.return_value = SignalSnapshot(
            short_ma=Decimal("105"),
            long_ma=Decimal("100"),
            epsilon=Decimal("1"),
            bullish_cross=True,
            bearish_cross=False,
            rsi=Decimal("50"),
        )

        config = StrategyConfig(filters_config=StrategyFiltersConfig(require_rsi_confirmation=True))
        strategy = PerpsBaselineEnhancedStrategy(config=config, signal_calculator=mock_calc)
        marks = [Decimal("100")] * 25

        signal = strategy._calculate_trading_signals(marks, Decimal("105"))

        mock_calc.calculate_signals.assert_called_once_with(marks, Decimal("105"), require_rsi=True)

    def test_real_calculator_produces_valid_signals(self):
        """Integration test with real calculator (no mocks)."""
        strategy = PerpsBaselineEnhancedStrategy()

        # Create bullish crossover scenario
        marks = [Decimal("100")] * 20  # Stable base
        marks.extend([Decimal("95"), Decimal("95")])  # Short drops below long
        marks.append(Decimal("125"))  # Sharp move triggers bullish cross

        signal = strategy._calculate_trading_signals(marks, Decimal("125"))

        # Default config: short=5, long=20
        # Short MA: last 5 marks = (100, 95, 95, 125) average, but only 4 marks after base
        # Actually uses (95, 95, 125, 100, 100) = 515/5 = 103
        assert signal.short_ma == Decimal("103")
        assert signal.bullish_cross
        assert not signal.bearish_cross
        assert signal.rsi is None  # No filters configured

    def test_real_calculator_with_epsilon_tolerance(self):
        """Integration test with epsilon tolerance."""
        config = StrategyConfig(ma_cross_epsilon_bps=Decimal("100"))  # 1% tolerance
        strategy = PerpsBaselineEnhancedStrategy(config=config)

        marks = [Decimal("1000")] * 25

        signal = strategy._calculate_trading_signals(marks, Decimal("1000"))

        # No crossover - all marks identical
        assert not signal.bullish_cross
        assert not signal.bearish_cross
        assert signal.epsilon == Decimal("10")  # 1% of 1000
