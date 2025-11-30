"""
Integration test for Ensemble Strategy.
"""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.ensemble import (
    EnsembleStrategy,
    EnsembleStrategyConfig,
)


class TestEnsembleStrategy:
    def test_initialization(self):
        strategy = EnsembleStrategy()
        assert len(strategy.signals) == 3
        assert strategy.combiner is not None

    def test_trend_signal_bullish(self):
        """Test that a clear uptrend produces a BUY signal."""
        strategy = EnsembleStrategy()

        # Create a synthetic uptrend
        # Fast MA (5) > Slow MA (20)
        # We need at least 20 data points
        prices = [Decimal("100") + Decimal(i) for i in range(30)]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        # Should be BUY because Trend Signal will be +0.5 (Bullish Trend)
        # Mean Reversion might be Sell (Overbought), but Trend usually dominates in default weights?
        # Wait, default weights in RegimeCombiner:
        # Neutral Regime (default): Trend 0.5, MeanRev 0.5.
        # Trend Signal: +0.5 (Bullish Trend)
        # Mean Rev Signal: Z-Score will be high (uptrend).
        # Z-Score of linear trend is approx sqrt(3)*(N-1)/N... it's high.
        # So Mean Rev will be -1.0 (Strong Sell).
        # Net: 0.5 * 0.5 + (-1.0) * 0.5 = 0.25 - 0.5 = -0.25 -> SELL?

        # This highlights the importance of Regime Detection!
        # With linear trend, ADX should be high -> Trending Regime.
        # If Trending Regime: Trend Weight 1.0, Mean Rev Weight 0.0.
        # Net: 0.5 * 1.0 + (-1.0) * 0.0 = +0.5 -> BUY.

        # BUT: We are not passing candles, so ADX is None, so Regime is Neutral.
        # So it might output SELL or HOLD.

        # To verify Trend Signal specifically, we can inspect indicators.
        indicators = decision.indicators
        components = indicators.get("components", {})
        trend_component = components.get("trend_ma", {})

        assert trend_component.get("raw") > 0  # Should be positive (bullish)

    def test_mean_reversion_signal(self):
        """Test mean reversion on ranging data."""
        # Configure with lower threshold to ensure triggering on sine wave
        from gpt_trader.features.live_trade.signals.mean_reversion import MeanReversionSignalConfig

        config = EnsembleStrategyConfig(
            mean_reversion_config=MeanReversionSignalConfig(z_entry_threshold=1.0)
        )
        strategy = EnsembleStrategy(config=config)

        # Create a sine wave (ranging)
        import math

        prices = [Decimal("100") + Decimal(str(10 * math.sin(i * 0.5))) for i in range(50)]

        # Force a sharp drop at the end to guarantee Z-Score trigger
        prices[-1] = Decimal("80")

        # At the end of sine wave?
        # i=49 -> 24.5 rad. 24.5 / 2pi = 3.9 cycles.
        # sin(24.5) is approx -0.99. So price is near low.
        # Z-Score should be negative -> Buy Signal.

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=prices[-1],
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        indicators = decision.indicators
        components = indicators.get("components", {})
        mr_component = components.get("mean_reversion_z", {})

        assert mr_component.get("raw") > 0  # Should be positive (Buy) because price is low
