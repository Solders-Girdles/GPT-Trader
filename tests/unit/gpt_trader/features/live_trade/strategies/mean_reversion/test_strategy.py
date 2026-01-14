"""
Unit tests for MeanReversionStrategy.

Tests cover:
1. Z-Score calculation accuracy
2. Signal generation at threshold boundaries
3. Position sizing with volatility targeting
4. Entry and exit logic
5. Kill switch behavior
6. Rehydration (stateless confirmation)
"""

from decimal import Decimal

from gpt_trader.app.config import MeanReversionConfig
from gpt_trader.features.live_trade.strategies.mean_reversion import (
    MeanReversionStrategy,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import Action


class TestZScoreCalculation:
    """Test Z-Score calculation accuracy."""

    def test_z_score_at_mean_is_zero(self):
        """Z-Score should be ~0 when price equals the rolling mean."""
        config = MeanReversionConfig(lookback_window=5)
        strategy = MeanReversionStrategy(config)

        # Create prices where the last price equals the mean
        # Mean of [100, 102, 98, 101, 99] = 100
        prices = [Decimal("100"), Decimal("102"), Decimal("98"), Decimal("101"), Decimal("100")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        z_score = decision.indicators.get("z_score", 0)
        assert abs(z_score) < 0.5, f"Z-Score should be near zero, got {z_score}"

    def test_z_score_positive_when_above_mean(self):
        """Z-Score should be positive when price is above the mean."""
        config = MeanReversionConfig(lookback_window=5)
        strategy = MeanReversionStrategy(config)

        # Create prices with a high final value
        # Mean of [100, 100, 100, 100, 110] ≈ 102
        prices = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("110")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("110"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        z_score = decision.indicators.get("z_score", 0)
        assert z_score > 0, f"Z-Score should be positive, got {z_score}"

    def test_z_score_negative_when_below_mean(self):
        """Z-Score should be negative when price is below the mean."""
        config = MeanReversionConfig(lookback_window=5)
        strategy = MeanReversionStrategy(config)

        # Create prices with a low final value
        prices = [Decimal("100"), Decimal("100"), Decimal("100"), Decimal("100"), Decimal("90")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("90"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        z_score = decision.indicators.get("z_score", 0)
        assert z_score < 0, f"Z-Score should be negative, got {z_score}"


class TestSignalGeneration:
    """Test signal generation at threshold boundaries."""

    def test_long_signal_below_negative_threshold(self):
        """Should generate BUY signal when Z-Score < -threshold."""
        config = MeanReversionConfig(
            z_score_entry_threshold=2.0,
            lookback_window=20,
        )
        strategy = MeanReversionStrategy(config)

        # Create stable prices then a significant drop (Z-Score < -2)
        prices = [Decimal("100")] * 19 + [Decimal("85")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("85"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.BUY, f"Expected BUY, got {decision.action}"
        assert "mean reversion long" in decision.reason.lower()

    def test_short_signal_above_positive_threshold(self):
        """Should generate SELL signal when Z-Score > +threshold."""
        config = MeanReversionConfig(
            z_score_entry_threshold=2.0,
            lookback_window=20,
            enable_shorts=True,
        )
        strategy = MeanReversionStrategy(config)

        # Create stable prices then a significant spike (Z-Score > +2)
        prices = [Decimal("100")] * 19 + [Decimal("115")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("115"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.SELL, f"Expected SELL, got {decision.action}"
        assert "mean reversion short" in decision.reason.lower()

    def test_no_short_when_disabled(self):
        """Should not generate SELL when shorts are disabled."""
        config = MeanReversionConfig(
            z_score_entry_threshold=2.0,
            lookback_window=20,
            enable_shorts=False,
        )
        strategy = MeanReversionStrategy(config)

        prices = [Decimal("100")] * 19 + [Decimal("115")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("115"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.HOLD, (
            f"Expected HOLD (shorts disabled), got {decision.action}"
        )

    def test_hold_in_neutral_zone(self):
        """Should HOLD when Z-Score is within neutral zone."""
        config = MeanReversionConfig(
            z_score_entry_threshold=2.0,
            lookback_window=20,
        )
        strategy = MeanReversionStrategy(config)

        # Create prices with reasonable variance where final price is near mean
        # Mean ≈ 100, std ≈ 5, price at 103 gives Z-Score ≈ 0.6 (in neutral zone)
        prices = [
            Decimal("95"),
            Decimal("98"),
            Decimal("102"),
            Decimal("105"),
            Decimal("100"),
            Decimal("97"),
            Decimal("103"),
            Decimal("99"),
            Decimal("101"),
            Decimal("98"),
            Decimal("104"),
            Decimal("96"),
            Decimal("100"),
            Decimal("102"),
            Decimal("98"),
            Decimal("101"),
            Decimal("99"),
            Decimal("100"),
            Decimal("102"),
            Decimal("100"),
        ]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        # Z-Score should be near 0 (within -2 to +2 range)
        z_score = decision.indicators.get("z_score", 0)
        assert abs(z_score) < 2.0, f"Z-Score {z_score} should be in neutral zone"
        assert decision.action == Action.HOLD, f"Expected HOLD, got {decision.action}"


class TestExitLogic:
    """Test exit conditions."""

    def test_exit_when_z_score_returns_to_mean(self):
        """Should CLOSE position when Z-Score returns near zero."""
        config = MeanReversionConfig(
            z_score_exit_threshold=0.5,
            lookback_window=20,
        )
        strategy = MeanReversionStrategy(config)

        # Z-Score near zero (price at mean)
        prices = [Decimal("100")] * 20

        position_state = {
            "quantity": Decimal("0.1"),
            "side": "long",
            "entry_price": Decimal("95"),
        }

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            position_state=position_state,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.CLOSE, f"Expected CLOSE at mean, got {decision.action}"
        assert "mean reversion complete" in decision.reason.lower()

    def test_stop_loss_triggered(self):
        """Should CLOSE position when stop loss is hit."""
        config = MeanReversionConfig(
            stop_loss_pct=0.03,  # 3% stop loss
            lookback_window=5,
        )
        strategy = MeanReversionStrategy(config)

        # Create varied prices to avoid exit signal
        prices = [Decimal("90"), Decimal("92"), Decimal("94"), Decimal("96"), Decimal("91")]

        position_state = {
            "quantity": Decimal("0.1"),
            "side": "long",
            "entry_price": Decimal("100"),  # Entry at 100, now at 91 = -9% loss
        }

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("91"),
            position_state=position_state,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.CLOSE, (
            f"Expected CLOSE on stop loss, got {decision.action}"
        )
        assert "stop loss" in decision.reason.lower()

    def test_take_profit_triggered(self):
        """Should CLOSE position when take profit is hit."""
        config = MeanReversionConfig(
            take_profit_pct=0.06,  # 6% take profit
            lookback_window=5,
        )
        strategy = MeanReversionStrategy(config)

        # Create varied prices to avoid exit signal
        prices = [Decimal("100"), Decimal("102"), Decimal("104"), Decimal("106"), Decimal("108")]

        position_state = {
            "quantity": Decimal("0.1"),
            "side": "long",
            "entry_price": Decimal("100"),  # Entry at 100, now at 108 = +8% gain
        }

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("108"),
            position_state=position_state,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.CLOSE, (
            f"Expected CLOSE on take profit, got {decision.action}"
        )
        assert "take profit" in decision.reason.lower()


class TestVolatilityTargeting:
    """Test volatility-targeted position sizing."""

    def test_high_volatility_reduces_position(self):
        """Position size should decrease when volatility is high."""
        config = MeanReversionConfig(
            target_daily_volatility=0.02,  # 2% target
            max_position_pct=0.25,
        )
        strategy = MeanReversionStrategy(config)

        equity = Decimal("10000")

        # High volatility (4%) should reduce position size
        high_vol_size = strategy.calculate_position_size(equity, current_volatility=0.04)

        # Low volatility (1%) should increase position size
        low_vol_size = strategy.calculate_position_size(equity, current_volatility=0.01)

        assert high_vol_size < low_vol_size, (
            f"High vol size ({high_vol_size}) should be less than low vol size ({low_vol_size})"
        )

    def test_position_capped_at_max_pct(self):
        """Position size should never exceed max_position_pct of equity."""
        config = MeanReversionConfig(
            target_daily_volatility=0.02,
            max_position_pct=0.25,
        )
        strategy = MeanReversionStrategy(config)

        equity = Decimal("10000")
        max_allowed = float(equity) * 0.25

        # Very low volatility would scale up significantly
        size = strategy.calculate_position_size(equity, current_volatility=0.001)

        assert float(size) <= max_allowed, f"Position {size} exceeds max allowed {max_allowed}"

    def test_zero_volatility_uses_default(self):
        """Should handle zero volatility gracefully."""
        config = MeanReversionConfig()
        strategy = MeanReversionStrategy(config)

        equity = Decimal("10000")

        # Should not raise, should use default volatility
        size = strategy.calculate_position_size(equity, current_volatility=0.0)

        assert size > 0, "Position size should be positive even with zero volatility"


class TestKillSwitch:
    """Test kill switch behavior."""

    def test_kill_switch_prevents_all_actions(self):
        """When kill switch is on, should always HOLD."""
        config = MeanReversionConfig(
            kill_switch_enabled=True,
            z_score_entry_threshold=2.0,
        )
        strategy = MeanReversionStrategy(config)

        # Even with extreme Z-Score, should HOLD
        prices = [Decimal("100")] * 19 + [Decimal("80")]

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("80"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.HOLD
        assert "kill switch" in decision.reason.lower()


class TestRehydration:
    """Test crash recovery / rehydration."""

    def test_rehydrate_returns_zero_for_stateless_strategy(self):
        """Stateless strategy should return 0 from rehydrate."""
        config = MeanReversionConfig()
        strategy = MeanReversionStrategy(config)

        events = [
            {"type": "price_tick", "symbol": "BTC-USD", "price": "100"},
            {"type": "price_tick", "symbol": "BTC-USD", "price": "101"},
        ]

        count = strategy.rehydrate(events)

        assert count == 0, "Stateless strategy should process 0 events"


class TestInsufficientData:
    """Test behavior with insufficient data."""

    def test_hold_when_insufficient_data(self):
        """Should HOLD when not enough price history."""
        config = MeanReversionConfig(lookback_window=20)
        strategy = MeanReversionStrategy(config)

        # Only 5 prices when 20 are needed
        prices = [Decimal("100")] * 5

        decision = strategy.decide(
            symbol="BTC-USD",
            current_mark=Decimal("100"),
            position_state=None,
            recent_marks=prices,
            equity=Decimal("10000"),
            product=None,
        )

        assert decision.action == Action.HOLD
        assert "insufficient data" in decision.reason.lower()
        assert decision.indicators["data_points"] == 5
        assert decision.indicators["required"] == 20
